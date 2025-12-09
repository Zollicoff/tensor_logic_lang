// Tensor Logic Runtime - Einsum Engine
//
// The core insight from Domingos' paper: all tensor logic operations are einsums.
// C[i,k] = A[i,j] B[j,k] is matrix multiplication via einsum
// Ancestor(x,z) = step(Parent(x,y) Ancestor(y,z)) is transitive closure
//
// Key operations:
// - Implicit join: adjacent tensors multiply; shared indices are contracted
// - Implicit projection: indices on RHS but not LHS are summed out
// - step() makes it Boolean (Heaviside function)

const std = @import("std");
const tensor = @import("tensor.zig");
const DenseTensor = tensor.DenseTensor;
const SparseTensor = tensor.SparseTensor;
const Shape = tensor.Shape;
const Tensor = tensor.Tensor;

/// Index specification for einsum
pub const IndexSpec = struct {
    /// Name of the index variable (e.g., "i", "j", "k")
    name: []const u8,
    /// Size of this dimension
    size: usize,
};

/// Einsum operand - a tensor with its index specification
pub const Operand = struct {
    tensor_ptr: *const Tensor,
    indices: []const []const u8, // e.g., ["i", "j"] for A[i,j]
};

/// Result of analyzing an einsum expression
pub const EinsumPlan = struct {
    /// All unique index names in the expression
    all_indices: [][]const u8,
    /// Sizes of each index
    index_sizes: []usize,
    /// Which indices appear in output (preserved)
    output_indices: [][]const u8,
    /// Which indices are contracted (summed out)
    contracted_indices: [][]const u8,

    allocator: std.mem.Allocator,

    pub fn deinit(self: *EinsumPlan) void {
        self.allocator.free(self.all_indices);
        self.allocator.free(self.index_sizes);
        self.allocator.free(self.output_indices);
        self.allocator.free(self.contracted_indices);
    }
};

/// Plan an einsum operation
/// output_indices: the indices that appear on the LHS
/// operands: the tensors on the RHS with their indices
pub fn planEinsum(
    allocator: std.mem.Allocator,
    output_indices: []const []const u8,
    operands: []const Operand,
) !EinsumPlan {
    // Collect all unique indices from operands
    var all_set = std.StringHashMap(usize).init(allocator);
    defer all_set.deinit();

    for (operands) |op| {
        for (op.indices, 0..) |idx_name, dim| {
            if (!all_set.contains(idx_name)) {
                const size = op.tensor_ptr.shape().dims[dim];
                try all_set.put(idx_name, size);
            }
        }
    }

    // Convert to arrays
    var all_indices = std.ArrayList([]const u8).init(allocator);
    var index_sizes = std.ArrayList(usize).init(allocator);

    var iter = all_set.iterator();
    while (iter.next()) |entry| {
        try all_indices.append(entry.key_ptr.*);
        try index_sizes.append(entry.value_ptr.*);
    }

    // Determine contracted indices (in operands but not in output)
    var contracted = std.ArrayList([]const u8).init(allocator);
    for (all_indices.items) |idx| {
        var in_output = false;
        for (output_indices) |out_idx| {
            if (std.mem.eql(u8, idx, out_idx)) {
                in_output = true;
                break;
            }
        }
        if (!in_output) {
            try contracted.append(idx);
        }
    }

    // Copy output indices
    const out_copy = try allocator.alloc([]const u8, output_indices.len);
    @memcpy(out_copy, output_indices);

    return EinsumPlan{
        .all_indices = try all_indices.toOwnedSlice(),
        .index_sizes = try index_sizes.toOwnedSlice(),
        .output_indices = out_copy,
        .contracted_indices = try contracted.toOwnedSlice(),
        .allocator = allocator,
    };
}

/// Execute matrix multiplication: C[i,k] = A[i,j] B[j,k]
/// This is the most common einsum pattern
pub fn matmul(
    allocator: std.mem.Allocator,
    a: *const DenseTensor(f64),
    b: *const DenseTensor(f64),
) !DenseTensor(f64) {
    // A is [m, n], B is [n, p], result is [m, p]
    const m = a.shape.dims[0];
    const n = a.shape.dims[1];
    const p = b.shape.dims[1];

    var c = try DenseTensor(f64).init(allocator, &[_]usize{ m, p });

    for (0..m) |i| {
        for (0..p) |k| {
            var sum: f64 = 0.0;
            for (0..n) |j| {
                sum += a.get(&[_]usize{ i, j }) * b.get(&[_]usize{ j, k });
            }
            c.set(&[_]usize{ i, k }, sum);
        }
    }

    return c;
}

/// Execute general einsum for two operands (most common case)
/// Handles: C[out_indices] = A[a_indices] B[b_indices]
pub fn einsum2(
    allocator: std.mem.Allocator,
    a: *const DenseTensor(f64),
    a_indices: []const []const u8,
    b: *const DenseTensor(f64),
    b_indices: []const []const u8,
    out_indices: []const []const u8,
) !DenseTensor(f64) {
    // Build index -> size mapping
    var index_size = std.StringHashMap(usize).init(allocator);
    defer index_size.deinit();

    for (a_indices, 0..) |idx, dim| {
        try index_size.put(idx, a.shape.dims[dim]);
    }
    for (b_indices, 0..) |idx, dim| {
        if (!index_size.contains(idx)) {
            try index_size.put(idx, b.shape.dims[dim]);
        }
    }

    // Determine output shape
    var out_shape = try allocator.alloc(usize, out_indices.len);
    defer allocator.free(out_shape);
    for (out_indices, 0..) |idx, i| {
        out_shape[i] = index_size.get(idx) orelse 1;
    }

    // Find contracted indices
    var all_indices = std.ArrayListUnmanaged([]const u8){};
    defer all_indices.deinit(allocator);

    var iter = index_size.keyIterator();
    while (iter.next()) |key| {
        try all_indices.append(allocator, key.*);
    }

    // Create output tensor
    var c = try DenseTensor(f64).init(allocator, out_shape);

    // Naive einsum: iterate over all combinations
    // (This is O(product of all dimensions) - not optimal but correct)
    try einsumLoop(allocator, &c, a, a_indices, b, b_indices, out_indices, &index_size, all_indices.items);

    return c;
}

fn einsumLoop(
    allocator: std.mem.Allocator,
    c: *DenseTensor(f64),
    a: *const DenseTensor(f64),
    a_indices: []const []const u8,
    b: *const DenseTensor(f64),
    b_indices: []const []const u8,
    out_indices: []const []const u8,
    index_size: *std.StringHashMap(usize),
    all_indices: []const []const u8,
) !void {
    // Create index state
    var idx_state = std.StringHashMap(usize).init(allocator);
    defer idx_state.deinit();

    for (all_indices) |idx| {
        try idx_state.put(idx, 0);
    }

    // Iterate through all index combinations
    while (true) {
        // Get A indices
        var a_idx = try allocator.alloc(usize, a_indices.len);
        defer allocator.free(a_idx);
        for (a_indices, 0..) |idx, i| {
            a_idx[i] = idx_state.get(idx) orelse 0;
        }

        // Get B indices
        var b_idx = try allocator.alloc(usize, b_indices.len);
        defer allocator.free(b_idx);
        for (b_indices, 0..) |idx, i| {
            b_idx[i] = idx_state.get(idx) orelse 0;
        }

        // Get output indices
        var c_idx = try allocator.alloc(usize, out_indices.len);
        defer allocator.free(c_idx);
        for (out_indices, 0..) |idx, i| {
            c_idx[i] = idx_state.get(idx) orelse 0;
        }

        // Compute product and accumulate
        const a_val = a.get(a_idx);
        const b_val = b.get(b_idx);
        const old_val = c.get(c_idx);
        c.set(c_idx, old_val + a_val * b_val);

        // Increment indices
        var carry = true;
        for (all_indices) |idx| {
            if (!carry) break;
            const current = idx_state.get(idx) orelse 0;
            const max_val = index_size.get(idx) orelse 1;
            if (current + 1 < max_val) {
                try idx_state.put(idx, current + 1);
                carry = false;
            } else {
                try idx_state.put(idx, 0);
            }
        }

        if (carry) break; // All combinations exhausted
    }
}

// ============================================================================
// Sparse Einsum Operations
// ============================================================================

/// Sparse-sparse einsum for two operands
/// C[out_indices] = A[a_indices] B[b_indices]
/// Uses hash-based lookup for O(nnz_a + nnz_b) when there are contracted indices
pub fn sparseEinsum2(
    allocator: std.mem.Allocator,
    a: *const SparseTensor(f64),
    a_indices: []const []const u8,
    b: *const SparseTensor(f64),
    b_indices: []const []const u8,
    out_indices: []const []const u8,
) !SparseTensor(f64) {
    // Build index -> size mapping from shapes
    var index_size = std.StringHashMap(usize).init(allocator);
    defer index_size.deinit();

    for (a_indices, 0..) |idx, dim| {
        try index_size.put(idx, a.shape.dims[dim]);
    }
    for (b_indices, 0..) |idx, dim| {
        if (!index_size.contains(idx)) {
            try index_size.put(idx, b.shape.dims[dim]);
        }
    }

    // Determine output shape
    var out_shape = try allocator.alloc(usize, out_indices.len);
    defer allocator.free(out_shape);
    for (out_indices, 0..) |idx, i| {
        out_shape[i] = index_size.get(idx) orelse 1;
    }

    // Find contracted indices (in both A and B but not in output)
    // Also find their positions in B's index list
    var contracted_names = std.ArrayListUnmanaged([]const u8){};
    defer contracted_names.deinit(allocator);
    var contracted_pos_in_a = std.ArrayListUnmanaged(usize){};
    defer contracted_pos_in_a.deinit(allocator);
    var contracted_pos_in_b = std.ArrayListUnmanaged(usize){};
    defer contracted_pos_in_b.deinit(allocator);

    for (a_indices, 0..) |a_idx, a_pos| {
        for (b_indices, 0..) |b_idx, b_pos| {
            if (std.mem.eql(u8, a_idx, b_idx)) {
                // Check if it's in output
                var in_out = false;
                for (out_indices) |o_idx| {
                    if (std.mem.eql(u8, a_idx, o_idx)) {
                        in_out = true;
                        break;
                    }
                }
                if (!in_out) {
                    try contracted_names.append(allocator, a_idx);
                    try contracted_pos_in_a.append(allocator, a_pos);
                    try contracted_pos_in_b.append(allocator, b_pos);
                }
                break;
            }
        }
    }

    // Create output sparse tensor
    var c = try SparseTensor(f64).init(allocator, out_shape);

    // If there are contracted indices, build a hash map for B entries
    // Key: contracted index values, Value: list of (entry_index, full_indices, value)
    if (contracted_names.items.len > 0) {
        // Build hash map: contracted values -> list of B entry indices
        var b_hash = std.AutoHashMap(u64, std.ArrayListUnmanaged(usize)).init(allocator);
        defer {
            var iter = b_hash.valueIterator();
            while (iter.next()) |list| {
                list.deinit(allocator);
            }
            b_hash.deinit();
        }

        // Populate hash map with B entries
        for (b.indices.items, 0..) |b_idx, entry_idx| {
            const key = computeContractedKey(b_idx, contracted_pos_in_b.items);
            const result = try b_hash.getOrPut(key);
            if (!result.found_existing) {
                result.value_ptr.* = std.ArrayListUnmanaged(usize){};
            }
            try result.value_ptr.append(allocator, entry_idx);
        }

        // For each A entry, look up matching B entries via hash
        for (a.indices.items, a.values.items) |a_idx, a_val| {
            const key = computeContractedKey(a_idx, contracted_pos_in_a.items);

            if (b_hash.get(key)) |matching_b_entries| {
                for (matching_b_entries.items) |b_entry_idx| {
                    const b_idx = b.indices.items[b_entry_idx];
                    const b_val = b.values.items[b_entry_idx];

                    // Compute output index
                    var c_idx = try allocator.alloc(usize, out_indices.len);
                    defer allocator.free(c_idx);

                    for (out_indices, 0..) |idx_name, i| {
                        // Find in A indices first
                        var found = false;
                        for (a_indices, 0..) |a_name, a_dim| {
                            if (std.mem.eql(u8, idx_name, a_name)) {
                                c_idx[i] = a_idx[a_dim];
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            // Find in B indices
                            for (b_indices, 0..) |b_name, b_dim| {
                                if (std.mem.eql(u8, idx_name, b_name)) {
                                    c_idx[i] = b_idx[b_dim];
                                    break;
                                }
                            }
                        }
                    }

                    // Accumulate product
                    const prod = a_val * b_val;
                    const old_val = c.get(c_idx);
                    try c.set(c_idx, old_val + prod);
                }
            }
        }
    } else {
        // No contracted indices - compute outer product (all pairs)
        for (a.indices.items, a.values.items) |a_idx, a_val| {
            for (b.indices.items, b.values.items) |b_idx, b_val| {
                // Compute output index
                var c_idx = try allocator.alloc(usize, out_indices.len);
                defer allocator.free(c_idx);

                for (out_indices, 0..) |idx_name, i| {
                    var found = false;
                    for (a_indices, 0..) |a_name, a_dim| {
                        if (std.mem.eql(u8, idx_name, a_name)) {
                            c_idx[i] = a_idx[a_dim];
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        for (b_indices, 0..) |b_name, b_dim| {
                            if (std.mem.eql(u8, idx_name, b_name)) {
                                c_idx[i] = b_idx[b_dim];
                                break;
                            }
                        }
                    }
                }

                // Accumulate product
                const prod = a_val * b_val;
                const old_val = c.get(c_idx);
                try c.set(c_idx, old_val + prod);
            }
        }
    }

    return c;
}

/// Compute a hash key from contracted index values
fn computeContractedKey(indices: []const usize, positions: []const usize) u64 {
    var key: u64 = 0;
    for (positions) |pos| {
        // Simple hash combining
        key = key *% 31 +% indices[pos];
    }
    return key;
}

/// Sparse-dense einsum: sparse A, dense B -> sparse result
/// Useful when A is a sparse relation and B is a dense tensor
pub fn sparseDenseEinsum2(
    allocator: std.mem.Allocator,
    a: *const SparseTensor(f64),
    a_indices: []const []const u8,
    b: *const DenseTensor(f64),
    b_indices: []const []const u8,
    out_indices: []const []const u8,
) !SparseTensor(f64) {
    // Build index -> size mapping
    var index_size = std.StringHashMap(usize).init(allocator);
    defer index_size.deinit();

    for (a_indices, 0..) |idx, dim| {
        try index_size.put(idx, a.shape.dims[dim]);
    }
    for (b_indices, 0..) |idx, dim| {
        if (!index_size.contains(idx)) {
            try index_size.put(idx, b.shape.dims[dim]);
        }
    }

    // Determine output shape
    var out_shape = try allocator.alloc(usize, out_indices.len);
    defer allocator.free(out_shape);
    for (out_indices, 0..) |idx, i| {
        out_shape[i] = index_size.get(idx) orelse 1;
    }

    // Find indices that are only in B (need to iterate over them)
    var b_only_indices = std.ArrayListUnmanaged([]const u8){};
    defer b_only_indices.deinit(allocator);

    for (b_indices) |b_idx| {
        var in_a = false;
        for (a_indices) |a_idx| {
            if (std.mem.eql(u8, b_idx, a_idx)) {
                in_a = true;
                break;
            }
        }
        if (!in_a) {
            try b_only_indices.append(allocator, b_idx);
        }
    }

    // Create output sparse tensor
    var c = try SparseTensor(f64).init(allocator, out_shape);

    // Iterate over non-zeros in A
    var iter_a = a.iterator();
    while (iter_a.next()) |entry_a| {
        // Extract A's index values
        var idx_map = std.StringHashMap(usize).init(allocator);
        defer idx_map.deinit();
        for (a_indices, 0..) |idx_name, dim| {
            try idx_map.put(idx_name, entry_a.indices[dim]);
        }

        // For indices only in B, we need to iterate over all their values
        // This is a multi-dimensional iteration
        try sparseIterateB(allocator, &c, entry_a.value, &idx_map, b, b_indices, out_indices, &index_size, b_only_indices.items, 0);
    }

    return c;
}

/// Helper for sparseDenseEinsum2: recursively iterate over B-only indices
fn sparseIterateB(
    allocator: std.mem.Allocator,
    c: *SparseTensor(f64),
    a_val: f64,
    idx_map: *std.StringHashMap(usize),
    b: *const DenseTensor(f64),
    b_indices: []const []const u8,
    out_indices: []const []const u8,
    index_size: *std.StringHashMap(usize),
    b_only: []const []const u8,
    depth: usize,
) !void {
    if (depth >= b_only.len) {
        // All B-only indices are set, compute the product
        var b_idx = try allocator.alloc(usize, b_indices.len);
        defer allocator.free(b_idx);
        for (b_indices, 0..) |idx_name, i| {
            b_idx[i] = idx_map.get(idx_name) orelse 0;
        }

        const b_val = b.get(b_idx);
        if (b_val != 0.0) {
            // Compute output index
            var c_idx = try allocator.alloc(usize, out_indices.len);
            defer allocator.free(c_idx);
            for (out_indices, 0..) |idx_name, i| {
                c_idx[i] = idx_map.get(idx_name) orelse 0;
            }

            const old_val = c.get(c_idx);
            try c.set(c_idx, old_val + a_val * b_val);
        }
        return;
    }

    // Iterate over all values of b_only[depth]
    const idx_name = b_only[depth];
    const size = index_size.get(idx_name) orelse 1;

    for (0..size) |val| {
        try idx_map.put(idx_name, val);
        try sparseIterateB(allocator, c, a_val, idx_map, b, b_indices, out_indices, index_size, b_only, depth + 1);
    }
}

/// Apply step function to sparse tensor (keeps sparsity)
pub fn stepSparseTensor(t: *SparseTensor(f64)) void {
    for (t.values.items) |*v| {
        v.* = step(v.*);
    }
    // Note: step(x) returns 0 for x <= 0, so we should compact
    // But for simplicity, we keep all entries (values become 0 or 1)
}

/// Apply ReLU to sparse tensor
pub fn reluSparseTensor(t: *SparseTensor(f64)) void {
    for (t.values.items) |*v| {
        v.* = relu(v.*);
    }
}

/// Apply sigmoid to sparse tensor
pub fn sigmoidSparseTensor(t: *SparseTensor(f64)) void {
    for (t.values.items) |*v| {
        v.* = sigmoid(v.*);
    }
}

// ============================================================================
// Nonlinearity functions (activation functions)
// ============================================================================

/// Heaviside step function: H(x) = 1 if x > 0 else 0
/// This is THE key function that makes tensor logic work for Boolean reasoning
pub fn step(x: f64) f64 {
    return if (x > 0) 1.0 else 0.0;
}

/// Apply step function element-wise to tensor
pub fn stepTensor(t: *DenseTensor(f64)) void {
    for (t.data) |*x| {
        x.* = step(x.*);
    }
}

/// ReLU: max(0, x)
pub fn relu(x: f64) f64 {
    return @max(0.0, x);
}

/// Apply ReLU element-wise
pub fn reluTensor(t: *DenseTensor(f64)) void {
    for (t.data) |*x| {
        x.* = relu(x.*);
    }
}

/// Sigmoid: 1 / (1 + exp(-x))
pub fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

/// Apply sigmoid element-wise
pub fn sigmoidTensor(t: *DenseTensor(f64)) void {
    for (t.data) |*x| {
        x.* = sigmoid(x.*);
    }
}

/// Softmax over last dimension
pub fn softmax(allocator: std.mem.Allocator, t: *DenseTensor(f64)) !void {
    _ = allocator;
    // For 1D tensor, softmax is straightforward
    if (t.shape.rank() == 1) {
        softmax1D(t.data);
    } else if (t.shape.rank() == 2) {
        // For 2D tensor, apply softmax over last dimension (each row)
        const rows = t.shape.dims[0];
        const cols = t.shape.dims[1];
        for (0..rows) |i| {
            const start = i * cols;
            const end = start + cols;
            softmax1D(t.data[start..end]);
        }
    }
    // TODO: handle higher-rank tensors
}

/// Apply softmax to a 1D slice
fn softmax1D(data: []f64) void {
    if (data.len == 0) return;

    // Find max for numerical stability
    var max_val: f64 = data[0];
    for (data) |x| {
        if (x > max_val) max_val = x;
    }

    // Compute exp(x - max) and sum
    var sum: f64 = 0.0;
    for (data) |*x| {
        x.* = @exp(x.* - max_val);
        sum += x.*;
    }

    // Normalize
    for (data) |*x| {
        x.* /= sum;
    }
}

/// Apply softmax to tensor (convenience wrapper)
pub fn softmaxTensor(t: *DenseTensor(f64)) void {
    if (t.shape.rank() == 1) {
        softmax1D(t.data);
    } else if (t.shape.rank() == 2) {
        const rows = t.shape.dims[0];
        const cols = t.shape.dims[1];
        for (0..rows) |i| {
            const start = i * cols;
            const end = start + cols;
            softmax1D(t.data[start..end]);
        }
    }
}

/// Tanh: (e^x - e^-x) / (e^x + e^-x)
pub fn tanh_fn(x: f64) f64 {
    return std.math.tanh(x);
}

/// Apply tanh element-wise
pub fn tanhTensor(t: *DenseTensor(f64)) void {
    for (t.data) |*x| {
        x.* = tanh_fn(x.*);
    }
}

/// Exponential: e^x
pub fn exp_fn(x: f64) f64 {
    return @exp(x);
}

/// Apply exp element-wise
pub fn expTensor(t: *DenseTensor(f64)) void {
    for (t.data) |*x| {
        x.* = exp_fn(x.*);
    }
}

/// Natural logarithm: ln(x)
pub fn log_fn(x: f64) f64 {
    return @log(x);
}

/// Apply log element-wise
pub fn logTensor(t: *DenseTensor(f64)) void {
    for (t.data) |*x| {
        x.* = log_fn(x.*);
    }
}

/// Absolute value: |x|
pub fn abs_fn(x: f64) f64 {
    return @abs(x);
}

/// Apply abs element-wise
pub fn absTensor(t: *DenseTensor(f64)) void {
    for (t.data) |*x| {
        x.* = abs_fn(x.*);
    }
}

/// Square root: sqrt(x)
pub fn sqrt_fn(x: f64) f64 {
    return @sqrt(x);
}

/// Apply sqrt element-wise
pub fn sqrtTensor(t: *DenseTensor(f64)) void {
    for (t.data) |*x| {
        x.* = sqrt_fn(x.*);
    }
}

/// Sine: sin(x)
pub fn sin_fn(x: f64) f64 {
    return @sin(x);
}

/// Apply sin element-wise
pub fn sinTensor(t: *DenseTensor(f64)) void {
    for (t.data) |*x| {
        x.* = sin_fn(x.*);
    }
}

/// Cosine: cos(x)
pub fn cos_fn(x: f64) f64 {
    return @cos(x);
}

/// Apply cos element-wise
pub fn cosTensor(t: *DenseTensor(f64)) void {
    for (t.data) |*x| {
        x.* = cos_fn(x.*);
    }
}

// ============================================================================
// Broadcasting Operations
// ============================================================================

/// Check if shapes are broadcast-compatible and compute result shape
/// Returns null if shapes are incompatible
pub fn broadcastShape(allocator: std.mem.Allocator, shape_a: []const usize, shape_b: []const usize) !?[]usize {
    const max_rank = @max(shape_a.len, shape_b.len);
    var result = try allocator.alloc(usize, max_rank);

    // Align shapes from the right
    var i: usize = 0;
    while (i < max_rank) : (i += 1) {
        const a_idx = if (i < shape_a.len) shape_a.len - 1 - i else null;
        const b_idx = if (i < shape_b.len) shape_b.len - 1 - i else null;

        const a_dim: usize = if (a_idx) |idx| shape_a[idx] else 1;
        const b_dim: usize = if (b_idx) |idx| shape_b[idx] else 1;

        if (a_dim == b_dim) {
            result[max_rank - 1 - i] = a_dim;
        } else if (a_dim == 1) {
            result[max_rank - 1 - i] = b_dim;
        } else if (b_dim == 1) {
            result[max_rank - 1 - i] = a_dim;
        } else {
            // Incompatible shapes
            allocator.free(result);
            return null;
        }
    }

    return result;
}

/// Broadcast add: C = A + B with broadcasting
pub fn broadcastAdd(
    allocator: std.mem.Allocator,
    a: *const DenseTensor(f64),
    b: *const DenseTensor(f64),
) !DenseTensor(f64) {
    const result_shape = try broadcastShape(allocator, a.shape.dims, b.shape.dims) orelse
        return error.IncompatibleShapes;
    defer allocator.free(result_shape);

    var result = try DenseTensor(f64).init(allocator, result_shape);

    // Iterate over all result indices
    const total = result.data.len;
    var flat_idx: usize = 0;
    while (flat_idx < total) : (flat_idx += 1) {
        // Convert flat index to multi-index
        const result_idx = try allocator.alloc(usize, result_shape.len);
        defer allocator.free(result_idx);
        flatToMultiIndex(flat_idx, result_shape, result_idx);

        // Map to A and B indices (with broadcasting)
        const a_val = getBroadcastValue(a, result_idx, result_shape.len);
        const b_val = getBroadcastValue(b, result_idx, result_shape.len);

        result.data[flat_idx] = a_val + b_val;
    }

    return result;
}

/// Broadcast subtract: C = A - B with broadcasting
pub fn broadcastSub(
    allocator: std.mem.Allocator,
    a: *const DenseTensor(f64),
    b: *const DenseTensor(f64),
) !DenseTensor(f64) {
    const result_shape = try broadcastShape(allocator, a.shape.dims, b.shape.dims) orelse
        return error.IncompatibleShapes;
    defer allocator.free(result_shape);

    var result = try DenseTensor(f64).init(allocator, result_shape);

    const total = result.data.len;
    var flat_idx: usize = 0;
    while (flat_idx < total) : (flat_idx += 1) {
        const result_idx = try allocator.alloc(usize, result_shape.len);
        defer allocator.free(result_idx);
        flatToMultiIndex(flat_idx, result_shape, result_idx);

        const a_val = getBroadcastValue(a, result_idx, result_shape.len);
        const b_val = getBroadcastValue(b, result_idx, result_shape.len);

        result.data[flat_idx] = a_val - b_val;
    }

    return result;
}

/// Broadcast multiply (element-wise): C = A * B with broadcasting
pub fn broadcastMul(
    allocator: std.mem.Allocator,
    a: *const DenseTensor(f64),
    b: *const DenseTensor(f64),
) !DenseTensor(f64) {
    const result_shape = try broadcastShape(allocator, a.shape.dims, b.shape.dims) orelse
        return error.IncompatibleShapes;
    defer allocator.free(result_shape);

    var result = try DenseTensor(f64).init(allocator, result_shape);

    const total = result.data.len;
    var flat_idx: usize = 0;
    while (flat_idx < total) : (flat_idx += 1) {
        const result_idx = try allocator.alloc(usize, result_shape.len);
        defer allocator.free(result_idx);
        flatToMultiIndex(flat_idx, result_shape, result_idx);

        const a_val = getBroadcastValue(a, result_idx, result_shape.len);
        const b_val = getBroadcastValue(b, result_idx, result_shape.len);

        result.data[flat_idx] = a_val * b_val;
    }

    return result;
}

/// Broadcast divide: C = A / B with broadcasting
pub fn broadcastDiv(
    allocator: std.mem.Allocator,
    a: *const DenseTensor(f64),
    b: *const DenseTensor(f64),
) !DenseTensor(f64) {
    const result_shape = try broadcastShape(allocator, a.shape.dims, b.shape.dims) orelse
        return error.IncompatibleShapes;
    defer allocator.free(result_shape);

    var result = try DenseTensor(f64).init(allocator, result_shape);

    const total = result.data.len;
    var flat_idx: usize = 0;
    while (flat_idx < total) : (flat_idx += 1) {
        const result_idx = try allocator.alloc(usize, result_shape.len);
        defer allocator.free(result_idx);
        flatToMultiIndex(flat_idx, result_shape, result_idx);

        const a_val = getBroadcastValue(a, result_idx, result_shape.len);
        const b_val = getBroadcastValue(b, result_idx, result_shape.len);

        result.data[flat_idx] = if (b_val != 0) a_val / b_val else 0;
    }

    return result;
}

/// Convert flat index to multi-dimensional index
fn flatToMultiIndex(flat_idx: usize, shape: []const usize, out_idx: []usize) void {
    var remaining = flat_idx;
    var i = shape.len;
    while (i > 0) {
        i -= 1;
        out_idx[i] = remaining % shape[i];
        remaining /= shape[i];
    }
}

/// Get value from tensor with broadcasting
/// result_idx is in the result coordinate space (potentially larger rank)
fn getBroadcastValue(t: *const DenseTensor(f64), result_idx: []const usize, result_rank: usize) f64 {
    const t_rank = t.shape.dims.len;
    var t_idx: [8]usize = undefined; // Max 8 dimensions

    // Map result indices to tensor indices (align from right)
    var i: usize = 0;
    while (i < t_rank) : (i += 1) {
        const result_pos = result_rank - t_rank + i;
        const t_dim = t.shape.dims[i];
        // If dimension is 1, broadcast (use index 0)
        // Otherwise, use the result index
        t_idx[i] = if (t_dim == 1) 0 else result_idx[result_pos];
    }

    return t.get(t_idx[0..t_rank]);
}

/// Add scalar to tensor (broadcast scalar over all elements)
pub fn addScalar(t: *DenseTensor(f64), scalar: f64) void {
    for (t.data) |*x| {
        x.* += scalar;
    }
}

/// Multiply tensor by scalar
pub fn mulScalar(t: *DenseTensor(f64), scalar: f64) void {
    for (t.data) |*x| {
        x.* *= scalar;
    }
}

// ============================================================================
// Element-wise Conditional (where)
// ============================================================================

/// Element-wise conditional: where(cond, then, else)
/// Returns then[i] if cond[i] > 0, else returns else_val[i]
/// Supports broadcasting between all three tensors
pub fn where(
    allocator: std.mem.Allocator,
    cond: *const DenseTensor(f64),
    then_val: *const DenseTensor(f64),
    else_val: *const DenseTensor(f64),
) !DenseTensor(f64) {
    // Compute broadcast shape from all three tensors
    const shape12 = try broadcastShape(allocator, cond.shape.dims, then_val.shape.dims) orelse
        return error.IncompatibleShapes;
    defer allocator.free(shape12);

    const result_shape = try broadcastShape(allocator, shape12, else_val.shape.dims) orelse
        return error.IncompatibleShapes;
    defer allocator.free(result_shape);

    var result = try DenseTensor(f64).init(allocator, result_shape);

    const total = result.data.len;
    var flat_idx: usize = 0;
    while (flat_idx < total) : (flat_idx += 1) {
        const result_idx = try allocator.alloc(usize, result_shape.len);
        defer allocator.free(result_idx);
        flatToMultiIndex(flat_idx, result_shape, result_idx);

        const cond_val = getBroadcastValue(cond, result_idx, result_shape.len);
        const t_val = getBroadcastValue(then_val, result_idx, result_shape.len);
        const e_val = getBroadcastValue(else_val, result_idx, result_shape.len);

        result.data[flat_idx] = if (cond_val > 0) t_val else e_val;
    }

    return result;
}

/// Element-wise conditional with scalar then/else values
pub fn whereScalar(
    allocator: std.mem.Allocator,
    cond: *const DenseTensor(f64),
    then_scalar: f64,
    else_scalar: f64,
) !DenseTensor(f64) {
    var result = try DenseTensor(f64).init(allocator, cond.shape.dims);

    for (cond.data, 0..) |c, i| {
        result.data[i] = if (c > 0) then_scalar else else_scalar;
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "matmul 2x3 @ 3x2" {
    const allocator = std.testing.allocator;

    var a = try DenseTensor(f64).init(allocator, &[_]usize{ 2, 3 });
    defer a.deinit();
    // A = [[1, 2, 3], [4, 5, 6]]
    a.set(&[_]usize{ 0, 0 }, 1);
    a.set(&[_]usize{ 0, 1 }, 2);
    a.set(&[_]usize{ 0, 2 }, 3);
    a.set(&[_]usize{ 1, 0 }, 4);
    a.set(&[_]usize{ 1, 1 }, 5);
    a.set(&[_]usize{ 1, 2 }, 6);

    var b = try DenseTensor(f64).init(allocator, &[_]usize{ 3, 2 });
    defer b.deinit();
    // B = [[1, 2], [3, 4], [5, 6]]
    b.set(&[_]usize{ 0, 0 }, 1);
    b.set(&[_]usize{ 0, 1 }, 2);
    b.set(&[_]usize{ 1, 0 }, 3);
    b.set(&[_]usize{ 1, 1 }, 4);
    b.set(&[_]usize{ 2, 0 }, 5);
    b.set(&[_]usize{ 2, 1 }, 6);

    var c = try matmul(allocator, &a, &b);
    defer c.deinit();

    // C = A @ B = [[22, 28], [49, 64]]
    try std.testing.expectEqual(@as(f64, 22), c.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 28), c.get(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f64, 49), c.get(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f64, 64), c.get(&[_]usize{ 1, 1 }));
}

test "step function" {
    try std.testing.expectEqual(@as(f64, 0.0), step(-1.0));
    try std.testing.expectEqual(@as(f64, 0.0), step(0.0));
    try std.testing.expectEqual(@as(f64, 1.0), step(0.001));
    try std.testing.expectEqual(@as(f64, 1.0), step(100.0));
}

test "relu function" {
    try std.testing.expectEqual(@as(f64, 0.0), relu(-5.0));
    try std.testing.expectEqual(@as(f64, 0.0), relu(0.0));
    try std.testing.expectEqual(@as(f64, 3.0), relu(3.0));
}

test "sigmoid function" {
    // sigmoid(0) = 0.5
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), sigmoid(0.0), 0.0001);
    // sigmoid approaches 1 for large positive
    try std.testing.expect(sigmoid(10.0) > 0.99);
    // sigmoid approaches 0 for large negative
    try std.testing.expect(sigmoid(-10.0) < 0.01);
}

test "tanh function" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), tanh_fn(0.0), 0.0001);
    try std.testing.expect(tanh_fn(10.0) > 0.99);
    try std.testing.expect(tanh_fn(-10.0) < -0.99);
}

test "exp function" {
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), exp_fn(0.0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.71828), exp_fn(1.0), 0.001);
}

test "log function" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), log_fn(1.0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), log_fn(2.71828), 0.001);
}

test "abs function" {
    try std.testing.expectEqual(@as(f64, 5.0), abs_fn(-5.0));
    try std.testing.expectEqual(@as(f64, 5.0), abs_fn(5.0));
    try std.testing.expectEqual(@as(f64, 0.0), abs_fn(0.0));
}

test "sqrt function" {
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), sqrt_fn(4.0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), sqrt_fn(9.0), 0.0001);
}

test "softmax 1D" {
    const allocator = std.testing.allocator;

    var t = try DenseTensor(f64).init(allocator, &[_]usize{3});
    defer t.deinit();
    t.data[0] = 1.0;
    t.data[1] = 2.0;
    t.data[2] = 3.0;

    softmaxTensor(&t);

    // Sum should be 1.0
    var sum: f64 = 0.0;
    for (t.data) |x| sum += x;
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 0.0001);

    // Larger input should have larger probability
    try std.testing.expect(t.data[2] > t.data[1]);
    try std.testing.expect(t.data[1] > t.data[0]);
}

test "softmax 2D" {
    const allocator = std.testing.allocator;

    var t = try DenseTensor(f64).init(allocator, &[_]usize{ 2, 3 });
    defer t.deinit();
    // Row 0: [1, 2, 3]
    t.data[0] = 1.0;
    t.data[1] = 2.0;
    t.data[2] = 3.0;
    // Row 1: [0, 0, 0]
    t.data[3] = 0.0;
    t.data[4] = 0.0;
    t.data[5] = 0.0;

    softmaxTensor(&t);

    // Each row should sum to 1.0
    const sum0: f64 = t.data[0] + t.data[1] + t.data[2];
    const sum1: f64 = t.data[3] + t.data[4] + t.data[5];
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum0, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum1, 0.0001);

    // Uniform input should give uniform output
    try std.testing.expectApproxEqAbs(t.data[3], t.data[4], 0.0001);
    try std.testing.expectApproxEqAbs(t.data[4], t.data[5], 0.0001);
}

test "sparse einsum matmul" {
    // Test sparse matrix multiplication: C[i,k] = A[i,j] B[j,k]
    // A = [[1, 0], [0, 2]] (sparse diagonal-ish)
    // B = [[3, 0], [0, 4]] (sparse diagonal)
    // C = A @ B = [[3, 0], [0, 8]]
    const allocator = std.testing.allocator;

    var a = try SparseTensor(f64).init(allocator, &[_]usize{ 2, 2 });
    defer a.deinit();
    try a.set(&[_]usize{ 0, 0 }, 1.0);
    try a.set(&[_]usize{ 1, 1 }, 2.0);

    var b = try SparseTensor(f64).init(allocator, &[_]usize{ 2, 2 });
    defer b.deinit();
    try b.set(&[_]usize{ 0, 0 }, 3.0);
    try b.set(&[_]usize{ 1, 1 }, 4.0);

    const a_indices = [_][]const u8{ "i", "j" };
    const b_indices = [_][]const u8{ "j", "k" };
    const out_indices = [_][]const u8{ "i", "k" };

    var c = try sparseEinsum2(allocator, &a, &a_indices, &b, &b_indices, &out_indices);
    defer c.deinit();

    // C should have 2 non-zeros: [0,0]=3, [1,1]=8
    try std.testing.expectEqual(@as(usize, 2), c.nnz());
    try std.testing.expectEqual(@as(f64, 3.0), c.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 8.0), c.get(&[_]usize{ 1, 1 }));
    try std.testing.expectEqual(@as(f64, 0.0), c.get(&[_]usize{ 0, 1 }));
}

test "sparse einsum transitive closure" {
    // Test transitive closure: Ancestor[i,k] = Parent[i,j] Ancestor[j,k]
    // Parent = {(0,1), (1,2)} means 0->1->2
    // After one iteration with initial Ancestor = Parent:
    // Ancestor[0,2] = Parent[0,1] * Ancestor[1,2] = 1 * 1 = 1
    const allocator = std.testing.allocator;

    var parent = try SparseTensor(f64).init(allocator, &[_]usize{ 3, 3 });
    defer parent.deinit();
    try parent.set(&[_]usize{ 0, 1 }, 1.0);
    try parent.set(&[_]usize{ 1, 2 }, 1.0);

    var ancestor = try SparseTensor(f64).init(allocator, &[_]usize{ 3, 3 });
    defer ancestor.deinit();
    try ancestor.set(&[_]usize{ 0, 1 }, 1.0);
    try ancestor.set(&[_]usize{ 1, 2 }, 1.0);

    // Compute: Result[i,k] = Parent[i,j] Ancestor[j,k]
    const p_indices = [_][]const u8{ "i", "j" };
    const a_indices = [_][]const u8{ "j", "k" };
    const out_indices = [_][]const u8{ "i", "k" };

    var result = try sparseEinsum2(allocator, &parent, &p_indices, &ancestor, &a_indices, &out_indices);
    defer result.deinit();

    // Result should have (0,2) = 1.0 from Parent[0,1] * Ancestor[1,2]
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&[_]usize{ 0, 2 }));
    // Direct parent relations should also be there
    try std.testing.expectEqual(@as(f64, 0.0), result.get(&[_]usize{ 0, 0 })); // No self-loop
}

test "sparse step function" {
    const allocator = std.testing.allocator;

    var t = try SparseTensor(f64).init(allocator, &[_]usize{ 3, 3 });
    defer t.deinit();

    try t.set(&[_]usize{ 0, 0 }, 0.5);
    try t.set(&[_]usize{ 1, 1 }, -0.3);
    try t.set(&[_]usize{ 2, 2 }, 2.0);

    stepSparseTensor(&t);

    try std.testing.expectEqual(@as(f64, 1.0), t.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 0.0), t.get(&[_]usize{ 1, 1 }));
    try std.testing.expectEqual(@as(f64, 1.0), t.get(&[_]usize{ 2, 2 }));
}

test "broadcast shape computation" {
    const allocator = std.testing.allocator;

    // [3, 1] + [1, 4] -> [3, 4]
    {
        const a_shape = [_]usize{ 3, 1 };
        const b_shape = [_]usize{ 1, 4 };
        const result = (try broadcastShape(allocator, &a_shape, &b_shape)).?;
        defer allocator.free(result);
        try std.testing.expectEqual(@as(usize, 2), result.len);
        try std.testing.expectEqual(@as(usize, 3), result[0]);
        try std.testing.expectEqual(@as(usize, 4), result[1]);
    }

    // [5, 3, 1] + [3, 4] -> [5, 3, 4]
    {
        const a_shape = [_]usize{ 5, 3, 1 };
        const b_shape = [_]usize{ 3, 4 };
        const result = (try broadcastShape(allocator, &a_shape, &b_shape)).?;
        defer allocator.free(result);
        try std.testing.expectEqual(@as(usize, 3), result.len);
        try std.testing.expectEqual(@as(usize, 5), result[0]);
        try std.testing.expectEqual(@as(usize, 3), result[1]);
        try std.testing.expectEqual(@as(usize, 4), result[2]);
    }

    // [4] + [3, 4] -> [3, 4]
    {
        const a_shape = [_]usize{4};
        const b_shape = [_]usize{ 3, 4 };
        const result = (try broadcastShape(allocator, &a_shape, &b_shape)).?;
        defer allocator.free(result);
        try std.testing.expectEqual(@as(usize, 2), result.len);
        try std.testing.expectEqual(@as(usize, 3), result[0]);
        try std.testing.expectEqual(@as(usize, 4), result[1]);
    }

    // [3, 4] + [5, 4] -> null (incompatible)
    {
        const a_shape = [_]usize{ 3, 4 };
        const b_shape = [_]usize{ 5, 4 };
        const result = try broadcastShape(allocator, &a_shape, &b_shape);
        try std.testing.expect(result == null);
    }
}

test "broadcast add" {
    const allocator = std.testing.allocator;

    // A[3, 1] + B[1, 4] -> C[3, 4]
    var a = try DenseTensor(f64).init(allocator, &[_]usize{ 3, 1 });
    defer a.deinit();
    a.data[0] = 1.0; // [0, 0]
    a.data[1] = 2.0; // [1, 0]
    a.data[2] = 3.0; // [2, 0]

    var b = try DenseTensor(f64).init(allocator, &[_]usize{ 1, 4 });
    defer b.deinit();
    b.data[0] = 10.0;
    b.data[1] = 20.0;
    b.data[2] = 30.0;
    b.data[3] = 40.0;

    var c = try broadcastAdd(allocator, &a, &b);
    defer c.deinit();

    try std.testing.expectEqual(@as(usize, 2), c.shape.dims.len);
    try std.testing.expectEqual(@as(usize, 3), c.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), c.shape.dims[1]);

    // C[0, 0] = A[0, 0] + B[0, 0] = 1 + 10 = 11
    try std.testing.expectEqual(@as(f64, 11.0), c.get(&[_]usize{ 0, 0 }));
    // C[0, 1] = A[0, 0] + B[0, 1] = 1 + 20 = 21
    try std.testing.expectEqual(@as(f64, 21.0), c.get(&[_]usize{ 0, 1 }));
    // C[1, 2] = A[1, 0] + B[0, 2] = 2 + 30 = 32
    try std.testing.expectEqual(@as(f64, 32.0), c.get(&[_]usize{ 1, 2 }));
    // C[2, 3] = A[2, 0] + B[0, 3] = 3 + 40 = 43
    try std.testing.expectEqual(@as(f64, 43.0), c.get(&[_]usize{ 2, 3 }));
}

test "broadcast 1d bias to 2d" {
    // This is the key neural network use case: Z[batch, hidden] + b[hidden]
    const allocator = std.testing.allocator;

    // Z[4, 8] - batch of 4, hidden dim 8
    var z = try DenseTensor(f64).init(allocator, &[_]usize{ 4, 8 });
    defer z.deinit();
    // Fill with 1.0
    for (z.data) |*x| x.* = 1.0;

    // b[8] - bias vector
    var b = try DenseTensor(f64).init(allocator, &[_]usize{8});
    defer b.deinit();
    for (0..8) |i| b.data[i] = @floatFromInt(i);

    // Z + b should broadcast b to [4, 8]
    var result = try broadcastAdd(allocator, &z, &b);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.shape.dims.len);
    try std.testing.expectEqual(@as(usize, 4), result.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 8), result.shape.dims[1]);

    // Each row should be [1, 2, 3, 4, 5, 6, 7, 8] (1.0 + 0..7)
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 2.0), result.get(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f64, 8.0), result.get(&[_]usize{ 0, 7 }));
    // Same for other rows
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&[_]usize{ 3, 0 }));
    try std.testing.expectEqual(@as(f64, 8.0), result.get(&[_]usize{ 3, 7 }));
}
