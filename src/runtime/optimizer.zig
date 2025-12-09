// Optimizer Module for Tensor Logic
//
// Implements gradient descent optimizers for training tensor logic programs.
//

const std = @import("std");
const tensor_mod = @import("tensor.zig");
const autodiff = @import("autodiff.zig");
const DenseTensor = tensor_mod.DenseTensor;

/// Stochastic Gradient Descent optimizer
pub const SGD = struct {
    learning_rate: f64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, learning_rate: f64) SGD {
        return .{
            .allocator = allocator,
            .learning_rate = learning_rate,
        };
    }

    /// Update a tensor in-place using its gradient
    /// param = param - learning_rate * gradient
    pub fn step(self: *SGD, param: *DenseTensor(f64), grad: *const DenseTensor(f64)) void {
        for (param.data, 0..) |*p, i| {
            p.* -= self.learning_rate * grad.data[i];
        }
    }

    /// Update multiple parameters using gradients from tape
    pub fn stepAll(
        self: *SGD,
        params: []const []const u8,
        tensors: *std.StringHashMap(tensor_mod.Tensor),
        tape: *autodiff.GradientTape,
    ) void {
        for (params) |name| {
            if (tape.getGradient(name)) |grad| {
                if (tensors.getPtr(name)) |tensor| {
                    switch (tensor.*) {
                        .f64_dense => |*dense| {
                            self.step(dense, grad);
                        },
                        else => {},
                    }
                }
            }
        }
    }
};

/// SGD with momentum
pub const MomentumSGD = struct {
    learning_rate: f64,
    momentum: f64,
    allocator: std.mem.Allocator,
    /// Velocity buffers for each parameter
    velocities: std.StringHashMap(*DenseTensor(f64)),

    pub fn init(allocator: std.mem.Allocator, learning_rate: f64, momentum: f64) MomentumSGD {
        return .{
            .allocator = allocator,
            .learning_rate = learning_rate,
            .momentum = momentum,
            .velocities = std.StringHashMap(*DenseTensor(f64)).init(allocator),
        };
    }

    pub fn deinit(self: *MomentumSGD) void {
        var iter = self.velocities.valueIterator();
        while (iter.next()) |v| {
            v.*.deinit();
            self.allocator.destroy(v.*);
        }
        self.velocities.deinit();
    }

    /// Update a tensor using momentum
    /// v = momentum * v + grad
    /// param = param - learning_rate * v
    pub fn step(self: *MomentumSGD, name: []const u8, param: *DenseTensor(f64), grad: *const DenseTensor(f64)) !void {
        // Get or create velocity buffer
        const v = self.velocities.get(name) orelse blk: {
            const new_v = try self.allocator.create(DenseTensor(f64));
            new_v.* = try DenseTensor(f64).init(self.allocator, param.shape.dims);
            // Initialize to zeros
            for (new_v.data) |*x| x.* = 0;
            try self.velocities.put(name, new_v);
            break :blk new_v;
        };

        // Update velocity and parameter
        for (v.data, 0..) |*vi, i| {
            vi.* = self.momentum * vi.* + grad.data[i];
            param.data[i] -= self.learning_rate * vi.*;
        }
    }
};

/// Adam optimizer
pub const Adam = struct {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    allocator: std.mem.Allocator,
    /// First moment estimates
    m: std.StringHashMap(*DenseTensor(f64)),
    /// Second moment estimates
    v: std.StringHashMap(*DenseTensor(f64)),
    /// Timestep
    t: usize,

    pub fn init(allocator: std.mem.Allocator, learning_rate: f64) Adam {
        return .{
            .allocator = allocator,
            .learning_rate = learning_rate,
            .beta1 = 0.9,
            .beta2 = 0.999,
            .epsilon = 1e-8,
            .m = std.StringHashMap(*DenseTensor(f64)).init(allocator),
            .v = std.StringHashMap(*DenseTensor(f64)).init(allocator),
            .t = 0,
        };
    }

    pub fn deinit(self: *Adam) void {
        var m_iter = self.m.valueIterator();
        while (m_iter.next()) |x| {
            x.*.deinit();
            self.allocator.destroy(x.*);
        }
        self.m.deinit();

        var v_iter = self.v.valueIterator();
        while (v_iter.next()) |x| {
            x.*.deinit();
            self.allocator.destroy(x.*);
        }
        self.v.deinit();
    }

    /// Update a tensor using Adam
    pub fn step(self: *Adam, name: []const u8, param: *DenseTensor(f64), grad: *const DenseTensor(f64)) !void {
        self.t += 1;

        // Get or create moment estimates
        const m = self.m.get(name) orelse blk: {
            const new_m = try self.allocator.create(DenseTensor(f64));
            new_m.* = try DenseTensor(f64).init(self.allocator, param.shape.dims);
            for (new_m.data) |*x| x.* = 0;
            try self.m.put(name, new_m);
            break :blk new_m;
        };

        const v = self.v.get(name) orelse blk: {
            const new_v = try self.allocator.create(DenseTensor(f64));
            new_v.* = try DenseTensor(f64).init(self.allocator, param.shape.dims);
            for (new_v.data) |*x| x.* = 0;
            try self.v.put(name, new_v);
            break :blk new_v;
        };

        // Bias correction terms
        const beta1_t = std.math.pow(f64, self.beta1, @floatFromInt(self.t));
        const beta2_t = std.math.pow(f64, self.beta2, @floatFromInt(self.t));

        // Update moments and parameter
        for (m.data, 0..) |*mi, i| {
            const g = grad.data[i];

            // m = beta1 * m + (1 - beta1) * g
            mi.* = self.beta1 * mi.* + (1 - self.beta1) * g;

            // v = beta2 * v + (1 - beta2) * g^2
            v.data[i] = self.beta2 * v.data[i] + (1 - self.beta2) * g * g;

            // Bias-corrected estimates
            const m_hat = mi.* / (1 - beta1_t);
            const v_hat = v.data[i] / (1 - beta2_t);

            // Update parameter
            param.data[i] -= self.learning_rate * m_hat / (@sqrt(v_hat) + self.epsilon);
        }
    }
};

// ============================================================================
// Loss Functions
// ============================================================================

/// Mean Squared Error loss
pub fn mseLoss(pred: *const DenseTensor(f64), target: *const DenseTensor(f64)) f64 {
    var sum: f64 = 0;
    for (pred.data, 0..) |p, i| {
        const diff = p - target.data[i];
        sum += diff * diff;
    }
    return sum / @as(f64, @floatFromInt(pred.data.len));
}

/// MSE gradient with respect to predictions
pub fn mseLossGrad(allocator: std.mem.Allocator, pred: *const DenseTensor(f64), target: *const DenseTensor(f64)) !DenseTensor(f64) {
    var grad = try DenseTensor(f64).init(allocator, pred.shape.dims);
    const n: f64 = @floatFromInt(pred.data.len);
    for (pred.data, 0..) |p, i| {
        grad.data[i] = 2.0 * (p - target.data[i]) / n;
    }
    return grad;
}

/// Cross-entropy loss (for softmax outputs)
pub fn crossEntropyLoss(pred: *const DenseTensor(f64), target: *const DenseTensor(f64)) f64 {
    var sum: f64 = 0;
    for (pred.data, 0..) |p, i| {
        if (target.data[i] > 0) {
            // -target * log(pred) - only for target=1 entries
            sum -= target.data[i] * @log(@max(p, 1e-10));
        }
    }
    return sum / @as(f64, @floatFromInt(pred.shape.dims[0])); // Average over batch
}

/// Cross-entropy gradient (assuming pred is softmax output)
/// grad = pred - target (when target is one-hot)
pub fn crossEntropyGrad(allocator: std.mem.Allocator, pred: *const DenseTensor(f64), target: *const DenseTensor(f64)) !DenseTensor(f64) {
    var grad = try DenseTensor(f64).init(allocator, pred.shape.dims);
    const batch_size: f64 = @floatFromInt(pred.shape.dims[0]);
    for (pred.data, 0..) |p, i| {
        grad.data[i] = (p - target.data[i]) / batch_size;
    }
    return grad;
}

// ============================================================================
// Tests
// ============================================================================

test "sgd step" {
    const allocator = std.testing.allocator;

    var param = try DenseTensor(f64).init(allocator, &[_]usize{3});
    defer param.deinit();
    param.data[0] = 1.0;
    param.data[1] = 2.0;
    param.data[2] = 3.0;

    var grad = try DenseTensor(f64).init(allocator, &[_]usize{3});
    defer grad.deinit();
    grad.data[0] = 0.1;
    grad.data[1] = 0.2;
    grad.data[2] = 0.3;

    var sgd = SGD.init(allocator, 1.0);
    sgd.step(&param, &grad);

    try std.testing.expectApproxEqAbs(@as(f64, 0.9), param.data[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.8), param.data[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.7), param.data[2], 0.0001);
}

test "mse loss" {
    const allocator = std.testing.allocator;

    var pred = try DenseTensor(f64).init(allocator, &[_]usize{3});
    defer pred.deinit();
    pred.data[0] = 1.0;
    pred.data[1] = 2.0;
    pred.data[2] = 3.0;

    var target = try DenseTensor(f64).init(allocator, &[_]usize{3});
    defer target.deinit();
    target.data[0] = 1.0;
    target.data[1] = 2.0;
    target.data[2] = 3.0;

    // Perfect prediction -> loss = 0
    const loss = mseLoss(&pred, &target);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), loss, 0.0001);

    // With error
    pred.data[0] = 2.0; // Off by 1
    const loss2 = mseLoss(&pred, &target);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), loss2, 0.0001);
}
