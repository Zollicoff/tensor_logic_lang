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
