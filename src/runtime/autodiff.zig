// Automatic Differentiation for Tensor Logic
//
// Implements reverse-mode AD (backpropagation) for tensor operations.
// This allows gradient-based learning on tensor logic programs.
//
// Key concepts:
// - Tape: Records operations during forward pass
// - Backward pass: Computes gradients by traversing tape in reverse
// - Gradient accumulation: Gradients are summed when a tensor is used multiple times

const std = @import("std");
const tensor_mod = @import("tensor.zig");
const DenseTensor = tensor_mod.DenseTensor;
const SparseTensor = tensor_mod.SparseTensor;
const TensorUnion = tensor_mod.TensorUnion;

/// Operation types that can be differentiated
pub const OpType = enum {
    /// Element-wise operations
    add,
    sub,
    mul,
    div,
    neg,

    /// Nonlinearities
    relu,
    sigmoid,
    tanh,
    exp,
    log,
    sqrt,
    softmax,

    /// Tensor operations
    einsum, // Generalized contraction
    sum, // Sum over indices
    copy, // Copy tensor (for broadcasting)
};

/// A recorded operation on the tape
pub const TapeEntry = struct {
    op: OpType,
    /// Output tensor name
    output: []const u8,
    /// Input tensor names
    inputs: []const []const u8,
    /// Cached values needed for backward pass
    cached_output: ?*DenseTensor(f64),
    cached_inputs: []?*DenseTensor(f64),
    /// For einsum: contraction indices
    einsum_info: ?EinsumInfo,

    pub const EinsumInfo = struct {
        left_indices: []const []const u8,
        right_indices: []const []const u8,
        output_indices: []const []const u8,
    };
};

/// Gradient tape for recording and differentiating computations
pub const GradientTape = struct {
    allocator: std.mem.Allocator,
    /// Recorded operations
    entries: std.ArrayListUnmanaged(TapeEntry),
    /// Gradients for each tensor (accumulated during backward pass)
    gradients: std.StringHashMap(*DenseTensor(f64)),
    /// Tensor values at forward pass (for backward computation)
    values: std.StringHashMap(*DenseTensor(f64)),

    pub fn init(allocator: std.mem.Allocator) GradientTape {
        return .{
            .allocator = allocator,
            .entries = std.ArrayListUnmanaged(TapeEntry){},
            .gradients = std.StringHashMap(*DenseTensor(f64)).init(allocator),
            .values = std.StringHashMap(*DenseTensor(f64)).init(allocator),
        };
    }

    pub fn deinit(self: *GradientTape) void {
        // Free cached tensors in entries
        for (self.entries.items) |entry| {
            if (entry.cached_output) |t| {
                t.deinit();
                self.allocator.destroy(t);
            }
            for (entry.cached_inputs) |maybe_t| {
                if (maybe_t) |t| {
                    t.deinit();
                    self.allocator.destroy(t);
                }
            }
            self.allocator.free(entry.cached_inputs);
        }
        self.entries.deinit(self.allocator);

        // Free gradient tensors
        var grad_iter = self.gradients.valueIterator();
        while (grad_iter.next()) |t| {
            t.*.deinit();
            self.allocator.destroy(t.*);
        }
        self.gradients.deinit();

        // Free value tensors
        var val_iter = self.values.valueIterator();
        while (val_iter.next()) |t| {
            t.*.deinit();
            self.allocator.destroy(t.*);
        }
        self.values.deinit();
    }

    /// Record an operation on the tape
    pub fn record(
        self: *GradientTape,
        op: OpType,
        output: []const u8,
        inputs: []const []const u8,
        output_tensor: *const DenseTensor(f64),
        input_tensors: []const *const DenseTensor(f64),
    ) !void {
        // Clone output tensor for backward pass
        const cached_output = try self.allocator.create(DenseTensor(f64));
        cached_output.* = try output_tensor.clone(self.allocator);

        // Clone input tensors
        const cached_inputs = try self.allocator.alloc(?*DenseTensor(f64), input_tensors.len);
        for (input_tensors, 0..) |input, i| {
            const cached = try self.allocator.create(DenseTensor(f64));
            cached.* = try input.clone(self.allocator);
            cached_inputs[i] = cached;
        }

        try self.entries.append(self.allocator, .{
            .op = op,
            .output = output,
            .inputs = inputs,
            .cached_output = cached_output,
            .cached_inputs = cached_inputs,
            .einsum_info = null,
        });
    }

    /// Record an einsum operation with index information
    pub fn recordEinsum(
        self: *GradientTape,
        output: []const u8,
        inputs: []const []const u8,
        output_tensor: *const DenseTensor(f64),
        input_tensors: []const *const DenseTensor(f64),
        left_indices: []const []const u8,
        right_indices: []const []const u8,
        output_indices: []const []const u8,
    ) !void {
        const cached_output = try self.allocator.create(DenseTensor(f64));
        cached_output.* = try output_tensor.clone(self.allocator);

        const cached_inputs = try self.allocator.alloc(?*DenseTensor(f64), input_tensors.len);
        for (input_tensors, 0..) |input, i| {
            const cached = try self.allocator.create(DenseTensor(f64));
            cached.* = try input.clone(self.allocator);
            cached_inputs[i] = cached;
        }

        try self.entries.append(self.allocator, .{
            .op = .einsum,
            .output = output,
            .inputs = inputs,
            .cached_output = cached_output,
            .cached_inputs = cached_inputs,
            .einsum_info = .{
                .left_indices = left_indices,
                .right_indices = right_indices,
                .output_indices = output_indices,
            },
        });
    }

    /// Perform backward pass starting from a scalar loss
    pub fn backward(self: *GradientTape, loss_name: []const u8) !void {
        // Initialize gradient of loss to 1.0
        if (self.values.get(loss_name)) |loss_tensor| {
            const grad = try self.allocator.create(DenseTensor(f64));
            grad.* = try DenseTensor(f64).init(self.allocator, loss_tensor.shape.dims);
            // Set all gradients to 1.0 (for scalar loss this is just one element)
            for (grad.data) |*v| {
                v.* = 1.0;
            }
            try self.gradients.put(loss_name, grad);
        }

        // Traverse tape in reverse order
        var i = self.entries.items.len;
        while (i > 0) {
            i -= 1;
            const entry = self.entries.items[i];

            // Get gradient of output
            const output_grad = self.gradients.get(entry.output) orelse continue;

            // Compute gradients for inputs based on operation type
            switch (entry.op) {
                .add => try self.backwardAdd(entry, output_grad),
                .sub => try self.backwardSub(entry, output_grad),
                .mul => try self.backwardMul(entry, output_grad),
                .div => try self.backwardDiv(entry, output_grad),
                .neg => try self.backwardNeg(entry, output_grad),
                .relu => try self.backwardRelu(entry, output_grad),
                .sigmoid => try self.backwardSigmoid(entry, output_grad),
                .tanh => try self.backwardTanh(entry, output_grad),
                .exp => try self.backwardExp(entry, output_grad),
                .log => try self.backwardLog(entry, output_grad),
                .sqrt => try self.backwardSqrt(entry, output_grad),
                .softmax => try self.backwardSoftmax(entry, output_grad),
                .einsum => try self.backwardEinsum(entry, output_grad),
                .sum => try self.backwardSum(entry, output_grad),
                .copy => try self.backwardCopy(entry, output_grad),
            }
        }
    }

    /// Get gradient for a tensor
    pub fn getGradient(self: *GradientTape, name: []const u8) ?*DenseTensor(f64) {
        return self.gradients.get(name);
    }

    // === Backward implementations ===

    fn backwardAdd(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // d_a = d_out, d_b = d_out
        for (entry.inputs) |input_name| {
            try self.accumulateGradient(input_name, output_grad);
        }
    }

    fn backwardSub(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // d_a = d_out, d_b = -d_out
        if (entry.inputs.len >= 1) {
            try self.accumulateGradient(entry.inputs[0], output_grad);
        }
        if (entry.inputs.len >= 2) {
            // Negate gradient for second input
            const neg_grad = try self.allocator.create(DenseTensor(f64));
            neg_grad.* = try DenseTensor(f64).init(self.allocator, output_grad.shape.dims);
            for (output_grad.data, 0..) |v, j| {
                neg_grad.data[j] = -v;
            }
            try self.accumulateGradientOwned(entry.inputs[1], neg_grad);
        }
    }

    fn backwardMul(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // d_a = d_out * b, d_b = d_out * a
        if (entry.inputs.len >= 2) {
            const a = entry.cached_inputs[0].?;
            const b = entry.cached_inputs[1].?;

            // Gradient for first input: d_out * b
            const grad_a = try self.allocator.create(DenseTensor(f64));
            grad_a.* = try DenseTensor(f64).init(self.allocator, a.shape.dims);
            for (output_grad.data, 0..) |d_out, j| {
                grad_a.data[j] = d_out * b.data[j];
            }
            try self.accumulateGradientOwned(entry.inputs[0], grad_a);

            // Gradient for second input: d_out * a
            const grad_b = try self.allocator.create(DenseTensor(f64));
            grad_b.* = try DenseTensor(f64).init(self.allocator, b.shape.dims);
            for (output_grad.data, 0..) |d_out, j| {
                grad_b.data[j] = d_out * a.data[j];
            }
            try self.accumulateGradientOwned(entry.inputs[1], grad_b);
        }
    }

    fn backwardDiv(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // d_a = d_out / b, d_b = -d_out * a / b^2
        if (entry.inputs.len >= 2) {
            const a = entry.cached_inputs[0].?;
            const b = entry.cached_inputs[1].?;

            // Gradient for first input
            const grad_a = try self.allocator.create(DenseTensor(f64));
            grad_a.* = try DenseTensor(f64).init(self.allocator, a.shape.dims);
            for (output_grad.data, 0..) |d_out, j| {
                grad_a.data[j] = d_out / b.data[j];
            }
            try self.accumulateGradientOwned(entry.inputs[0], grad_a);

            // Gradient for second input
            const grad_b = try self.allocator.create(DenseTensor(f64));
            grad_b.* = try DenseTensor(f64).init(self.allocator, b.shape.dims);
            for (output_grad.data, 0..) |d_out, j| {
                grad_b.data[j] = -d_out * a.data[j] / (b.data[j] * b.data[j]);
            }
            try self.accumulateGradientOwned(entry.inputs[1], grad_b);
        }
    }

    fn backwardNeg(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // d_a = -d_out
        if (entry.inputs.len >= 1) {
            const neg_grad = try self.allocator.create(DenseTensor(f64));
            neg_grad.* = try DenseTensor(f64).init(self.allocator, output_grad.shape.dims);
            for (output_grad.data, 0..) |v, j| {
                neg_grad.data[j] = -v;
            }
            try self.accumulateGradientOwned(entry.inputs[0], neg_grad);
        }
    }

    fn backwardRelu(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // d_x = d_out * (x > 0)
        if (entry.inputs.len >= 1 and entry.cached_inputs[0] != null) {
            const x = entry.cached_inputs[0].?;
            const grad = try self.allocator.create(DenseTensor(f64));
            grad.* = try DenseTensor(f64).init(self.allocator, x.shape.dims);
            for (x.data, 0..) |v, j| {
                grad.data[j] = if (v > 0) output_grad.data[j] else 0;
            }
            try self.accumulateGradientOwned(entry.inputs[0], grad);
        }
    }

    fn backwardSigmoid(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // d_x = d_out * sigmoid(x) * (1 - sigmoid(x))
        if (entry.cached_output) |y| {
            const grad = try self.allocator.create(DenseTensor(f64));
            grad.* = try DenseTensor(f64).init(self.allocator, y.shape.dims);
            for (y.data, 0..) |sig_x, j| {
                grad.data[j] = output_grad.data[j] * sig_x * (1.0 - sig_x);
            }
            try self.accumulateGradientOwned(entry.inputs[0], grad);
        }
    }

    fn backwardTanh(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // d_x = d_out * (1 - tanh(x)^2)
        if (entry.cached_output) |y| {
            const grad = try self.allocator.create(DenseTensor(f64));
            grad.* = try DenseTensor(f64).init(self.allocator, y.shape.dims);
            for (y.data, 0..) |tanh_x, j| {
                grad.data[j] = output_grad.data[j] * (1.0 - tanh_x * tanh_x);
            }
            try self.accumulateGradientOwned(entry.inputs[0], grad);
        }
    }

    fn backwardExp(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // d_x = d_out * exp(x)
        if (entry.cached_output) |y| {
            const grad = try self.allocator.create(DenseTensor(f64));
            grad.* = try DenseTensor(f64).init(self.allocator, y.shape.dims);
            for (y.data, 0..) |exp_x, j| {
                grad.data[j] = output_grad.data[j] * exp_x;
            }
            try self.accumulateGradientOwned(entry.inputs[0], grad);
        }
    }

    fn backwardLog(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // d_x = d_out / x
        if (entry.inputs.len >= 1 and entry.cached_inputs[0] != null) {
            const x = entry.cached_inputs[0].?;
            const grad = try self.allocator.create(DenseTensor(f64));
            grad.* = try DenseTensor(f64).init(self.allocator, x.shape.dims);
            for (x.data, 0..) |v, j| {
                grad.data[j] = output_grad.data[j] / v;
            }
            try self.accumulateGradientOwned(entry.inputs[0], grad);
        }
    }

    fn backwardSqrt(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // d_x = d_out / (2 * sqrt(x))
        if (entry.cached_output) |y| {
            const grad = try self.allocator.create(DenseTensor(f64));
            grad.* = try DenseTensor(f64).init(self.allocator, y.shape.dims);
            for (y.data, 0..) |sqrt_x, j| {
                grad.data[j] = output_grad.data[j] / (2.0 * sqrt_x);
            }
            try self.accumulateGradientOwned(entry.inputs[0], grad);
        }
    }

    fn backwardSoftmax(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // Softmax backward is more complex: d_x[i] = softmax[i] * (d_out[i] - sum_j(d_out[j] * softmax[j]))
        if (entry.cached_output) |softmax_out| {
            const grad = try self.allocator.create(DenseTensor(f64));
            grad.* = try DenseTensor(f64).init(self.allocator, softmax_out.shape.dims);

            // Compute sum of d_out * softmax
            var dot_sum: f64 = 0;
            for (output_grad.data, 0..) |d_out, j| {
                dot_sum += d_out * softmax_out.data[j];
            }

            // Compute gradient
            for (softmax_out.data, 0..) |s, j| {
                grad.data[j] = s * (output_grad.data[j] - dot_sum);
            }
            try self.accumulateGradientOwned(entry.inputs[0], grad);
        }
    }

    fn backwardEinsum(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // For einsum C[i,k] = A[i,j] B[j,k]:
        // d_A[i,j] = d_C[i,k] B[k,j] (contract over k)
        // d_B[j,k] = A[j,i] d_C[i,k] (contract over i)
        //
        // This is a simplified implementation for 2D matrix multiply
        // Full einsum backward would need index permutation logic

        if (entry.cached_inputs.len >= 2) {
            const a = entry.cached_inputs[0].?;
            const b = entry.cached_inputs[1].?;

            // For now, implement basic matmul backward
            // d_A = d_C @ B^T
            // d_B = A^T @ d_C

            if (a.shape.dims.len == 2 and b.shape.dims.len == 2 and output_grad.shape.dims.len == 2) {
                const m = a.shape.dims[0];
                const k_a = a.shape.dims[1];
                const k_b = b.shape.dims[0];
                const n = b.shape.dims[1];

                if (k_a == k_b) {
                    // d_A = d_C @ B^T
                    const grad_a = try self.allocator.create(DenseTensor(f64));
                    grad_a.* = try DenseTensor(f64).init(self.allocator, a.shape.dims);
                    for (grad_a.data) |*v| v.* = 0;

                    for (0..m) |i| {
                        for (0..k_a) |j| {
                            var sum: f64 = 0;
                            for (0..n) |ki| {
                                sum += output_grad.get(&[_]usize{ i, ki }) * b.get(&[_]usize{ j, ki });
                            }
                            grad_a.set(&[_]usize{ i, j }, sum);
                        }
                    }
                    try self.accumulateGradientOwned(entry.inputs[0], grad_a);

                    // d_B = A^T @ d_C
                    const grad_b = try self.allocator.create(DenseTensor(f64));
                    grad_b.* = try DenseTensor(f64).init(self.allocator, b.shape.dims);
                    for (grad_b.data) |*v| v.* = 0;

                    for (0..k_b) |j| {
                        for (0..n) |ki| {
                            var sum: f64 = 0;
                            for (0..m) |i| {
                                sum += a.get(&[_]usize{ i, j }) * output_grad.get(&[_]usize{ i, ki });
                            }
                            grad_b.set(&[_]usize{ j, ki }, sum);
                        }
                    }
                    try self.accumulateGradientOwned(entry.inputs[1], grad_b);
                }
            }
        }
    }

    fn backwardSum(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // Sum backward: gradient broadcasts back
        if (entry.inputs.len >= 1 and entry.cached_inputs[0] != null) {
            const x = entry.cached_inputs[0].?;
            const grad = try self.allocator.create(DenseTensor(f64));
            grad.* = try DenseTensor(f64).init(self.allocator, x.shape.dims);

            // If output is scalar, broadcast gradient to all elements
            if (output_grad.data.len == 1) {
                const scalar_grad = output_grad.data[0];
                for (grad.data) |*v| {
                    v.* = scalar_grad;
                }
            } else {
                // Otherwise, just copy
                @memcpy(grad.data, output_grad.data);
            }
            try self.accumulateGradientOwned(entry.inputs[0], grad);
        }
    }

    fn backwardCopy(self: *GradientTape, entry: TapeEntry, output_grad: *DenseTensor(f64)) !void {
        // Copy backward: just pass gradient through
        if (entry.inputs.len >= 1) {
            try self.accumulateGradient(entry.inputs[0], output_grad);
        }
    }

    /// Accumulate gradient by copying
    fn accumulateGradient(self: *GradientTape, name: []const u8, grad: *DenseTensor(f64)) !void {
        if (self.gradients.get(name)) |existing| {
            // Add to existing gradient
            for (existing.data, 0..) |*v, j| {
                v.* += grad.data[j];
            }
        } else {
            // Create new gradient
            const new_grad = try self.allocator.create(DenseTensor(f64));
            new_grad.* = try grad.clone(self.allocator);
            try self.gradients.put(name, new_grad);
        }
    }

    /// Accumulate gradient, taking ownership
    fn accumulateGradientOwned(self: *GradientTape, name: []const u8, grad: *DenseTensor(f64)) !void {
        if (self.gradients.get(name)) |existing| {
            // Add to existing gradient
            for (existing.data, 0..) |*v, j| {
                v.* += grad.data[j];
            }
            // Free the owned gradient since we accumulated it
            grad.deinit();
            self.allocator.destroy(grad);
        } else {
            // Store owned gradient directly
            try self.gradients.put(name, grad);
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "gradient tape basic" {
    const allocator = std.testing.allocator;
    var tape = GradientTape.init(allocator);
    defer tape.deinit();

    // Create a simple computation: y = relu(x)
    var x = try DenseTensor(f64).init(allocator, &[_]usize{4});
    defer x.deinit();
    x.data[0] = -1.0;
    x.data[1] = 0.0;
    x.data[2] = 1.0;
    x.data[3] = 2.0;

    // Compute relu
    var y = try DenseTensor(f64).init(allocator, &[_]usize{4});
    defer y.deinit();
    for (x.data, 0..) |v, i| {
        y.data[i] = @max(0, v);
    }

    // Record operation
    const inputs = [_][]const u8{"x"};
    const input_ptrs = [_]*const DenseTensor(f64){&x};
    try tape.record(.relu, "y", &inputs, &y, &input_ptrs);

    // Store value for backward pass
    const y_copy = try allocator.create(DenseTensor(f64));
    y_copy.* = try y.clone(allocator);
    try tape.values.put("y", y_copy);

    // Run backward
    try tape.backward("y");

    // Check gradients
    const grad_x = tape.getGradient("x").?;
    try std.testing.expectEqual(@as(f64, 0), grad_x.data[0]); // x=-1, relu'=0
    try std.testing.expectEqual(@as(f64, 0), grad_x.data[1]); // x=0, relu'=0
    try std.testing.expectEqual(@as(f64, 1), grad_x.data[2]); // x=1, relu'=1
    try std.testing.expectEqual(@as(f64, 1), grad_x.data[3]); // x=2, relu'=1
}

test "gradient sigmoid" {
    const allocator = std.testing.allocator;
    var tape = GradientTape.init(allocator);
    defer tape.deinit();

    // x = 0, sigmoid(0) = 0.5, sigmoid'(0) = 0.25
    var x = try DenseTensor(f64).init(allocator, &[_]usize{1});
    defer x.deinit();
    x.data[0] = 0.0;

    // Compute sigmoid
    var y = try DenseTensor(f64).init(allocator, &[_]usize{1});
    defer y.deinit();
    y.data[0] = 1.0 / (1.0 + @exp(-x.data[0])); // Should be 0.5

    const inputs = [_][]const u8{"x"};
    const input_ptrs = [_]*const DenseTensor(f64){&x};
    try tape.record(.sigmoid, "y", &inputs, &y, &input_ptrs);

    const y_copy = try allocator.create(DenseTensor(f64));
    y_copy.* = try y.clone(allocator);
    try tape.values.put("y", y_copy);

    try tape.backward("y");

    const grad_x = tape.getGradient("x").?;
    // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), grad_x.data[0], 1e-6);
}

test "gradient matmul" {
    const allocator = std.testing.allocator;
    var tape = GradientTape.init(allocator);
    defer tape.deinit();

    // A = [[1, 2], [3, 4]], B = [[1, 0], [0, 1]]
    // C = A @ B = A (identity)
    var a = try DenseTensor(f64).init(allocator, &[_]usize{ 2, 2 });
    defer a.deinit();
    a.set(&[_]usize{ 0, 0 }, 1);
    a.set(&[_]usize{ 0, 1 }, 2);
    a.set(&[_]usize{ 1, 0 }, 3);
    a.set(&[_]usize{ 1, 1 }, 4);

    var b = try DenseTensor(f64).init(allocator, &[_]usize{ 2, 2 });
    defer b.deinit();
    b.set(&[_]usize{ 0, 0 }, 1);
    b.set(&[_]usize{ 0, 1 }, 0);
    b.set(&[_]usize{ 1, 0 }, 0);
    b.set(&[_]usize{ 1, 1 }, 1);

    var c = try DenseTensor(f64).init(allocator, &[_]usize{ 2, 2 });
    defer c.deinit();
    // C = A @ B (identity, so C = A)
    c.set(&[_]usize{ 0, 0 }, 1);
    c.set(&[_]usize{ 0, 1 }, 2);
    c.set(&[_]usize{ 1, 0 }, 3);
    c.set(&[_]usize{ 1, 1 }, 4);

    const inputs = [_][]const u8{ "a", "b" };
    const input_ptrs = [_]*const DenseTensor(f64){ &a, &b };
    const left_idx = [_][]const u8{ "i", "j" };
    const right_idx = [_][]const u8{ "j", "k" };
    const out_idx = [_][]const u8{ "i", "k" };
    try tape.recordEinsum("c", &inputs, &c, &input_ptrs, &left_idx, &right_idx, &out_idx);

    const c_copy = try allocator.create(DenseTensor(f64));
    c_copy.* = try c.clone(allocator);
    try tape.values.put("c", c_copy);

    try tape.backward("c");

    // Since B is identity, d_A = d_C @ B^T = d_C
    const grad_a = tape.getGradient("a").?;
    try std.testing.expectEqual(@as(f64, 1), grad_a.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 1), grad_a.get(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f64, 1), grad_a.get(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f64, 1), grad_a.get(&[_]usize{ 1, 1 }));
}
