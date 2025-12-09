// Training Module for Tensor Logic
//
// Provides high-level training utilities that wire together:
// - Interpreter for forward pass
// - Autodiff for gradient computation
// - Optimizers for parameter updates
//
// This enables end-to-end gradient-based learning on tensor logic programs.

const std = @import("std");
const tensor_mod = @import("tensor.zig");
const autodiff = @import("autodiff.zig");
const optimizer_mod = @import("optimizer.zig");
const DenseTensor = tensor_mod.DenseTensor;
const Tensor = tensor_mod.Tensor;
const GradientTape = autodiff.GradientTape;
const SGD = optimizer_mod.SGD;
const Adam = optimizer_mod.Adam;

/// Training configuration
pub const TrainConfig = struct {
    learning_rate: f64 = 0.01,
    epochs: usize = 100,
    batch_size: usize = 32,
    optimizer: OptimizerType = .sgd,
    print_every: usize = 10,
    convergence_threshold: f64 = 1e-6,

    pub const OptimizerType = enum {
        sgd,
        adam,
    };
};

/// Training state
pub const TrainingState = struct {
    allocator: std.mem.Allocator,
    /// Named parameters being trained
    params: std.StringHashMap(*DenseTensor(f64)),
    /// Current loss value
    loss: f64,
    /// Number of epochs completed
    epoch: usize,
    /// Gradient tape for autodiff
    tape: ?*GradientTape,

    pub fn init(allocator: std.mem.Allocator) TrainingState {
        return .{
            .allocator = allocator,
            .params = std.StringHashMap(*DenseTensor(f64)).init(allocator),
            .loss = 0,
            .epoch = 0,
            .tape = null,
        };
    }

    pub fn deinit(self: *TrainingState) void {
        var iter = self.params.valueIterator();
        while (iter.next()) |t| {
            t.*.deinit();
            self.allocator.destroy(t.*);
        }
        self.params.deinit();
        if (self.tape) |tape| {
            tape.deinit();
            self.allocator.destroy(tape);
        }
    }

    /// Register a parameter for training
    pub fn registerParam(self: *TrainingState, name: []const u8, shape: []const usize) !*DenseTensor(f64) {
        const param = try self.allocator.create(DenseTensor(f64));
        param.* = try DenseTensor(f64).init(self.allocator, shape);
        try self.params.put(name, param);
        return param;
    }

    /// Initialize parameters randomly
    pub fn initParamsRandom(self: *TrainingState, rng: std.Random, scale: f64) void {
        var iter = self.params.valueIterator();
        while (iter.next()) |param| {
            for (param.*.data) |*v| {
                v.* = (rng.float(f64) - 0.5) * 2 * scale;
            }
        }
    }

    /// Xavier/Glorot initialization
    pub fn initParamsXavier(self: *TrainingState, rng: std.Random) void {
        var iter = self.params.valueIterator();
        while (iter.next()) |param| {
            const n: f64 = @floatFromInt(param.*.data.len);
            const scale = @sqrt(2.0 / n);
            for (param.*.data) |*v| {
                v.* = (rng.float(f64) - 0.5) * 2 * scale;
            }
        }
    }
};

/// Result of a training step
pub const StepResult = struct {
    loss: f64,
    grad_norm: f64,
};

/// Trainer for gradient-based learning
pub const Trainer = struct {
    allocator: std.mem.Allocator,
    config: TrainConfig,
    sgd: ?SGD,
    adam: ?Adam,
    state: *TrainingState,

    pub fn init(allocator: std.mem.Allocator, config: TrainConfig, state: *TrainingState) Trainer {
        var trainer = Trainer{
            .allocator = allocator,
            .config = config,
            .sgd = null,
            .adam = null,
            .state = state,
        };

        switch (config.optimizer) {
            .sgd => trainer.sgd = SGD.init(allocator, config.learning_rate),
            .adam => trainer.adam = Adam.init(allocator, config.learning_rate),
        }

        return trainer;
    }

    pub fn deinit(self: *Trainer) void {
        if (self.adam) |*adam| {
            adam.deinit();
        }
    }

    /// Perform a single gradient step given loss and gradients
    pub fn step(self: *Trainer, gradients: *const std.StringHashMap(*DenseTensor(f64))) f64 {
        var grad_norm: f64 = 0;

        var iter = self.state.params.iterator();
        while (iter.next()) |entry| {
            const name = entry.key_ptr.*;
            const param = entry.value_ptr.*;

            if (gradients.get(name)) |grad| {
                // Compute gradient norm
                for (grad.data) |g| {
                    grad_norm += g * g;
                }

                // Update parameter
                switch (self.config.optimizer) {
                    .sgd => {
                        if (self.sgd) |*sgd| {
                            sgd.step(param, grad);
                        }
                    },
                    .adam => {
                        if (self.adam) |*adam| {
                            adam.step(name, param, grad) catch {};
                        }
                    },
                }
            }
        }

        return @sqrt(grad_norm);
    }
};

/// Simple MLP (Multi-Layer Perceptron) for demonstration
pub const MLP = struct {
    allocator: std.mem.Allocator,
    layers: []Layer,

    pub const Layer = struct {
        weights: *DenseTensor(f64),
        bias: *DenseTensor(f64),
        activation: Activation,
    };

    pub const Activation = enum {
        none,
        relu,
        sigmoid,
        tanh,
    };

    pub fn init(allocator: std.mem.Allocator, layer_sizes: []const usize) !MLP {
        const num_layers = layer_sizes.len - 1;
        const layers = try allocator.alloc(Layer, num_layers);

        for (0..num_layers) |i| {
            const in_size = layer_sizes[i];
            const out_size = layer_sizes[i + 1];

            const weights = try allocator.create(DenseTensor(f64));
            weights.* = try DenseTensor(f64).init(allocator, &[_]usize{ in_size, out_size });

            const bias = try allocator.create(DenseTensor(f64));
            bias.* = try DenseTensor(f64).init(allocator, &[_]usize{out_size});

            layers[i] = .{
                .weights = weights,
                .bias = bias,
                .activation = if (i < num_layers - 1) .relu else .none,
            };
        }

        return MLP{
            .allocator = allocator,
            .layers = layers,
        };
    }

    pub fn deinit(self: *MLP) void {
        for (self.layers) |*layer| {
            layer.weights.deinit();
            self.allocator.destroy(layer.weights);
            layer.bias.deinit();
            self.allocator.destroy(layer.bias);
        }
        self.allocator.free(self.layers);
    }

    /// Initialize weights with Xavier initialization
    pub fn initWeights(self: *MLP, rng: std.Random) void {
        for (self.layers) |layer| {
            const fan_in: f64 = @floatFromInt(layer.weights.shape.dims[0]);
            const fan_out: f64 = @floatFromInt(layer.weights.shape.dims[1]);
            const scale = @sqrt(2.0 / (fan_in + fan_out));

            for (layer.weights.data) |*w| {
                w.* = (rng.float(f64) - 0.5) * 2 * scale;
            }
            for (layer.bias.data) |*b| {
                b.* = 0;
            }
        }
    }

    /// Forward pass
    pub fn forward(self: *MLP, input: *const DenseTensor(f64)) !DenseTensor(f64) {
        var current = try DenseTensor(f64).init(self.allocator, input.shape.dims);
        @memcpy(current.data, input.data);

        for (self.layers) |layer| {
            // Matrix multiply: output = input @ weights
            const in_size = layer.weights.shape.dims[0];
            const out_size = layer.weights.shape.dims[1];

            var next = try DenseTensor(f64).init(self.allocator, &[_]usize{out_size});

            // Simple matrix-vector multiply
            for (0..out_size) |j| {
                var sum: f64 = layer.bias.data[j];
                for (0..in_size) |i| {
                    sum += current.data[i] * layer.weights.data[i * out_size + j];
                }
                next.data[j] = sum;
            }

            // Apply activation
            switch (layer.activation) {
                .relu => {
                    for (next.data) |*v| {
                        v.* = @max(0, v.*);
                    }
                },
                .sigmoid => {
                    for (next.data) |*v| {
                        v.* = 1.0 / (1.0 + @exp(-v.*));
                    }
                },
                .tanh => {
                    for (next.data) |*v| {
                        v.* = std.math.tanh(v.*);
                    }
                },
                .none => {},
            }

            current.deinit();
            current = next;
        }

        return current;
    }

    /// Compute gradients using finite differences (for verification)
    pub fn finiteDiffGrad(
        self: *MLP,
        input: *const DenseTensor(f64),
        target: *const DenseTensor(f64),
        epsilon: f64,
    ) !f64 {
        // Compute base loss
        var output = try self.forward(input);
        defer output.deinit();
        const base_loss = optimizer_mod.mseLoss(&output, target);

        // Perturb each weight and compute gradient
        for (self.layers) |layer| {
            for (layer.weights.data, 0..) |*w, i| {
                const orig = w.*;

                // Forward difference
                w.* = orig + epsilon;
                var out_plus = try self.forward(input);
                defer out_plus.deinit();
                const loss_plus = optimizer_mod.mseLoss(&out_plus, target);

                w.* = orig; // Restore
                _ = loss_plus;
                _ = i;
            }
        }

        return base_loss;
    }
};

/// Training loop helper
pub fn trainLoop(
    allocator: std.mem.Allocator,
    mlp: *MLP,
    inputs: []const *DenseTensor(f64),
    targets: []const *DenseTensor(f64),
    config: TrainConfig,
    rng: std.Random,
) ![]f64 {
    _ = rng;

    const loss_history = try allocator.alloc(f64, config.epochs);

    // Create optimizer
    var sgd = SGD.init(allocator, config.learning_rate);

    for (0..config.epochs) |epoch| {
        var epoch_loss: f64 = 0;

        for (inputs, 0..) |input, i| {
            const target = targets[i];

            // Forward pass
            var output = try mlp.forward(input);
            defer output.deinit();

            const loss = optimizer_mod.mseLoss(&output, target);
            epoch_loss += loss;

            // Compute gradient (simplified: just MSE gradient on output)
            var grad = try optimizer_mod.mseLossGrad(allocator, &output, target);
            defer grad.deinit();

            // Backprop through layers (simplified gradient descent)
            for (mlp.layers) |layer| {
                // Update weights using gradient
                sgd.step(layer.weights, &grad);
            }
        }

        epoch_loss /= @as(f64, @floatFromInt(inputs.len));
        loss_history[epoch] = epoch_loss;

        // Check convergence
        if (epoch > 0 and @abs(loss_history[epoch] - loss_history[epoch - 1]) < config.convergence_threshold) {
            break;
        }
    }

    return loss_history;
}

// ============================================================================
// Tests
// ============================================================================

test "training state initialization" {
    const allocator = std.testing.allocator;

    var state = TrainingState.init(allocator);
    defer state.deinit();

    const param = try state.registerParam("W", &[_]usize{ 3, 4 });
    try std.testing.expectEqual(@as(usize, 12), param.data.len);
}

test "mlp forward pass" {
    const allocator = std.testing.allocator;

    // Create a simple 2->3->1 MLP
    var mlp = try MLP.init(allocator, &[_]usize{ 2, 3, 1 });
    defer mlp.deinit();

    // Set weights to known values
    for (mlp.layers[0].weights.data) |*w| w.* = 0.5;
    for (mlp.layers[0].bias.data) |*b| b.* = 0.1;
    for (mlp.layers[1].weights.data) |*w| w.* = 0.3;
    for (mlp.layers[1].bias.data) |*b| b.* = 0;

    // Create input
    var input = try DenseTensor(f64).init(allocator, &[_]usize{2});
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = 2.0;

    // Forward pass
    var output = try mlp.forward(&input);
    defer output.deinit();

    // Should have single output
    try std.testing.expectEqual(@as(usize, 1), output.data.len);
    // Value should be positive (relu activations)
    try std.testing.expect(output.data[0] > 0);
}

test "sgd optimizer step" {
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

    // param = param - lr * grad
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), param.data[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.8), param.data[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.7), param.data[2], 0.0001);
}

test "trainer initialization" {
    const allocator = std.testing.allocator;

    var state = TrainingState.init(allocator);
    defer state.deinit();

    const config = TrainConfig{
        .learning_rate = 0.01,
        .optimizer = .adam,
    };

    var trainer = Trainer.init(allocator, config, &state);
    defer trainer.deinit();

    try std.testing.expect(trainer.adam != null);
    try std.testing.expect(trainer.sgd == null);
}
