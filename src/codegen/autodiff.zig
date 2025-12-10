// Automatic Differentiation for Tensor Logic
//
// Implements reverse-mode autodiff (backpropagation) by deriving gradient
// equations from forward equations. The key insight is that gradients are
// just more tensor equations - this aligns with the tensor logic paradigm.
//
// Given: Y[i] = f(X[j])
// Derive: dL_dX[j] += dL_dY[i] * df/dX[i,j]
//
// Gradient rules:
// - Matmul: C[i,k] = A[i,j] B[j,k]
//   -> dL_dA[i,j] += dL_dC[i,k] * B[j,k]
//   -> dL_dB[j,k] += A[i,j] * dL_dC[i,k]
//
// - Addition: C[i] = A[i] + B[i]
//   -> dL_dA[i] += dL_dC[i]
//   -> dL_dB[i] += dL_dC[i]
//
// - ReLU: Y[i] = relu(X[i])
//   -> dL_dX[i] += dL_dY[i] * step(X[i])
//
// - Sigmoid: Y[i] = sigmoid(X[i])
//   -> dL_dX[i] += dL_dY[i] * Y[i] * (1 - Y[i])
//
// - Softmax: Y[i,j.] = softmax(X[i,j])
//   -> dL_dX[i,j] += Y[i,j] * (dL_dY[i,j] - sum_k(dL_dY[i,k] * Y[i,k]))

const std = @import("std");
const ast = @import("../frontend/ast.zig");

/// Computation graph node for autodiff
pub const ComputeNode = struct {
    /// Output tensor name
    output: []const u8,
    /// Input tensor names
    inputs: []const []const u8,
    /// Operation type
    op: OpType,
    /// Original equation
    equation: *const ast.Equation,
    /// Index mapping for gradient computation
    indices: []const ast.Index,
};

/// Operation types for gradient rules
pub const OpType = enum {
    matmul, // Product of tensors (einsum contraction)
    add, // Binary addition
    mul, // Element-wise multiplication
    scalar, // Scalar assignment
    relu,
    sigmoid,
    tanh,
    softmax,
    lnorm,
    exp,
    log,
    identity, // Direct assignment
};

/// Autodiff context
pub const Autodiff = struct {
    allocator: std.mem.Allocator,
    /// Computation graph: tensor name -> node that computes it
    nodes: std.StringHashMapUnmanaged(ComputeNode),
    /// Reverse topological order for backprop
    topo_order: std.ArrayListUnmanaged([]const u8),
    /// Generated gradient equations
    grad_equations: std.ArrayListUnmanaged(GradEquation),

    pub fn init(allocator: std.mem.Allocator) Autodiff {
        return .{
            .allocator = allocator,
            .nodes = .{},
            .topo_order = .{},
            .grad_equations = .{},
        };
    }

    pub fn deinit(self: *Autodiff) void {
        self.nodes.deinit(self.allocator);
        self.topo_order.deinit(self.allocator);
        self.grad_equations.deinit(self.allocator);
    }

    /// Build computation graph from program equations
    pub fn buildGraph(self: *Autodiff, program: *const ast.Program) !void {
        for (program.statements) |stmt| {
            if (stmt == .equation) {
                const eq = &stmt.equation;
                try self.addEquation(eq);
            }
        }
    }

    /// Add an equation to the computation graph
    fn addEquation(self: *Autodiff, eq: *const ast.Equation) !void {
        const output = eq.lhs.name;

        // Collect input tensors and determine operation type
        var inputs = std.ArrayListUnmanaged([]const u8){};
        defer inputs.deinit(self.allocator);

        const op = try self.analyzeExpr(eq.rhs, &inputs);

        try self.nodes.put(self.allocator, output, .{
            .output = output,
            .inputs = try inputs.toOwnedSlice(self.allocator),
            .op = op,
            .equation = eq,
            .indices = eq.lhs.indices,
        });
    }

    /// Analyze expression to determine operation type and collect inputs
    fn analyzeExpr(self: *Autodiff, expr: *const ast.Expr, inputs: *std.ArrayListUnmanaged([]const u8)) !OpType {
        switch (expr.*) {
            .tensor_ref => |ref| {
                try inputs.append(self.allocator, ref.name);
                return .identity;
            },
            .literal => return .scalar,
            .product => |prod| {
                for (prod.factors) |factor| {
                    _ = try self.analyzeExpr(factor, inputs);
                }
                return .matmul;
            },
            .binary => |bin| {
                _ = try self.analyzeExpr(bin.left, inputs);
                _ = try self.analyzeExpr(bin.right, inputs);
                return switch (bin.op) {
                    .add => .add,
                    .mul => .mul,
                    else => .identity,
                };
            },
            .nonlinearity => |nl| {
                _ = try self.analyzeExpr(nl.arg, inputs);
                return switch (nl.func) {
                    .relu => .relu,
                    .sigmoid => .sigmoid,
                    .tanh => .tanh,
                    .softmax => .softmax,
                    .lnorm => .lnorm,
                    .exp => .exp,
                    .log => .log,
                    else => .identity,
                };
            },
            .unary => |un| {
                _ = try self.analyzeExpr(un.operand, inputs);
                return .identity;
            },
            .group => |g| return self.analyzeExpr(g, inputs),
            else => return .identity,
        }
    }

    /// Compute gradients for loss with respect to parameters
    pub fn computeGradients(
        self: *Autodiff,
        loss: []const u8,
        params: []const []const u8,
    ) !void {
        // Build topological order via DFS from loss
        var visited = std.StringHashMapUnmanaged(void){};
        defer visited.deinit(self.allocator);

        try self.topoSort(loss, &visited);

        // Reverse order for backprop
        std.mem.reverse([]const u8, self.topo_order.items);

        // Initialize gradient of loss w.r.t. itself as 1
        // dL/dL = 1 (this is implicit)

        // Generate gradient equations for each node in reverse topo order
        for (self.topo_order.items) |tensor_name| {
            if (self.nodes.get(tensor_name)) |node| {
                try self.genGradients(&node, params);
            }
        }
    }

    /// Topological sort via DFS
    fn topoSort(self: *Autodiff, name: []const u8, visited: *std.StringHashMapUnmanaged(void)) !void {
        if (visited.contains(name)) return;
        try visited.put(self.allocator, name, {});

        if (self.nodes.get(name)) |node| {
            for (node.inputs) |input| {
                try self.topoSort(input, visited);
            }
        }

        try self.topo_order.append(self.allocator, name);
    }

    /// Generate gradient equations for a compute node
    /// We generate gradients for ALL inputs, not just parameters, to enable chain rule
    fn genGradients(self: *Autodiff, node: *const ComputeNode, params: []const []const u8) !void {
        const output = node.output;
        const grad_output = try self.gradName(output);

        switch (node.op) {
            .matmul => {
                // For C = A * B (matrix multiplication):
                // dL/dA = dL/dC * B^T (matmul_left)
                // dL/dB = A^T * dL/dC (matmul_right)
                // Special case: L = Y * Y -> dL/dY = 2 * Y (matmul_self)

                // Track which inputs we've already handled to avoid duplicates
                var handled = std.StringHashMapUnmanaged(void){};
                defer handled.deinit(self.allocator);

                // Check if this is Y*Y (same tensor twice) or A*B (different tensors)
                const is_self_product = node.inputs.len == 2 and
                    std.mem.eql(u8, node.inputs[0], node.inputs[1]);

                if (is_self_product) {
                    // Y*Y case: dL/dY = 2*Y
                    const input = node.inputs[0];
                    if (self.needsGradient(input, params)) {
                        try self.grad_equations.append(self.allocator, .{
                            .output = try self.gradName(input),
                            .rule = .matmul_self,
                            .grad_upstream = grad_output,
                            .operands = try self.allocator.dupe([]const u8, &[_][]const u8{input}),
                            .original_eq = node.equation,
                        });
                    }
                } else if (node.inputs.len >= 2) {
                    // A*B case: proper matrix multiplication gradients
                    const a_name = node.inputs[0];
                    const b_name = node.inputs[1];

                    if (self.needsGradient(a_name, params)) {
                        try self.grad_equations.append(self.allocator, .{
                            .output = try self.gradName(a_name),
                            .rule = .matmul_left, // dL/dA = dL/dC * B^T
                            .grad_upstream = grad_output,
                            .operands = try self.allocator.dupe([]const u8, &[_][]const u8{ a_name, b_name, output }),
                            .original_eq = node.equation,
                        });
                    }

                    if (self.needsGradient(b_name, params)) {
                        try self.grad_equations.append(self.allocator, .{
                            .output = try self.gradName(b_name),
                            .rule = .matmul_right, // dL/dB = A^T * dL/dC
                            .grad_upstream = grad_output,
                            .operands = try self.allocator.dupe([]const u8, &[_][]const u8{ a_name, b_name, output }),
                            .original_eq = node.equation,
                        });
                    }
                } else {
                    // Single input product - pass through
                    for (node.inputs) |input| {
                        if (handled.contains(input)) continue;
                        try handled.put(self.allocator, input, {});

                        if (self.needsGradient(input, params)) {
                            try self.grad_equations.append(self.allocator, .{
                                .output = try self.gradName(input),
                                .rule = .pass_through,
                                .grad_upstream = grad_output,
                                .operands = try self.allocator.dupe([]const u8, &[_][]const u8{input}),
                                .original_eq = node.equation,
                            });
                        }
                    }
                }
            },
            .add => {
                for (node.inputs) |input| {
                    if (self.needsGradient(input, params)) {
                        try self.grad_equations.append(self.allocator, .{
                            .output = try self.gradName(input),
                            .rule = .pass_through,
                            .grad_upstream = grad_output,
                            .operands = try self.allocator.dupe([]const u8, &[_][]const u8{input}),
                            .original_eq = node.equation,
                        });
                    }
                }
            },
            .relu => {
                if (node.inputs.len >= 1) {
                    const input = node.inputs[0];
                    if (self.needsGradient(input, params)) {
                        try self.grad_equations.append(self.allocator, .{
                            .output = try self.gradName(input),
                            .rule = .relu_grad,
                            .grad_upstream = grad_output,
                            .operands = try self.allocator.dupe([]const u8, &[_][]const u8{input}),
                            .original_eq = node.equation,
                        });
                    }
                }
            },
            .sigmoid => {
                if (node.inputs.len >= 1) {
                    const input = node.inputs[0];
                    if (self.needsGradient(input, params)) {
                        try self.grad_equations.append(self.allocator, .{
                            .output = try self.gradName(input),
                            .rule = .sigmoid_grad,
                            .grad_upstream = grad_output,
                            .operands = try self.allocator.dupe([]const u8, &[_][]const u8{ input, output }),
                            .original_eq = node.equation,
                        });
                    }
                }
            },
            .softmax => {
                if (node.inputs.len >= 1) {
                    const input = node.inputs[0];
                    if (self.needsGradient(input, params)) {
                        try self.grad_equations.append(self.allocator, .{
                            .output = try self.gradName(input),
                            .rule = .softmax_grad,
                            .grad_upstream = grad_output,
                            .operands = try self.allocator.dupe([]const u8, &[_][]const u8{ input, output }),
                            .original_eq = node.equation,
                        });
                    }
                }
            },
            .tanh => {
                // tanh gradient: dL/dX = dL/dY * (1 - Y^2)
                if (node.inputs.len >= 1) {
                    const input = node.inputs[0];
                    if (self.needsGradient(input, params)) {
                        try self.grad_equations.append(self.allocator, .{
                            .output = try self.gradName(input),
                            .rule = .tanh_grad,
                            .grad_upstream = grad_output,
                            .operands = try self.allocator.dupe([]const u8, &[_][]const u8{ input, output }),
                            .original_eq = node.equation,
                        });
                    }
                }
            },
            .exp => {
                // exp gradient: dL/dX = dL/dY * exp(X) = dL/dY * Y
                if (node.inputs.len >= 1) {
                    const input = node.inputs[0];
                    if (self.needsGradient(input, params)) {
                        try self.grad_equations.append(self.allocator, .{
                            .output = try self.gradName(input),
                            .rule = .exp_grad,
                            .grad_upstream = grad_output,
                            .operands = try self.allocator.dupe([]const u8, &[_][]const u8{ input, output }),
                            .original_eq = node.equation,
                        });
                    }
                }
            },
            .log => {
                // log gradient: dL/dX = dL/dY / X
                if (node.inputs.len >= 1) {
                    const input = node.inputs[0];
                    if (self.needsGradient(input, params)) {
                        try self.grad_equations.append(self.allocator, .{
                            .output = try self.gradName(input),
                            .rule = .log_grad,
                            .grad_upstream = grad_output,
                            .operands = try self.allocator.dupe([]const u8, &[_][]const u8{input}),
                            .original_eq = node.equation,
                        });
                    }
                }
            },
            .identity => {
                for (node.inputs) |input| {
                    if (self.needsGradient(input, params)) {
                        try self.grad_equations.append(self.allocator, .{
                            .output = try self.gradName(input),
                            .rule = .pass_through,
                            .grad_upstream = grad_output,
                            .operands = try self.allocator.dupe([]const u8, &[_][]const u8{input}),
                            .original_eq = node.equation,
                        });
                    }
                }
            },
            else => {},
        }
    }

    /// Check if a tensor needs a gradient computed (is param or has path to param)
    fn needsGradient(self: *Autodiff, name: []const u8, params: []const []const u8) bool {
        return self.isParam(name, params) or self.hasPathToParam(name, params);
    }

    /// Generate gradient tensor name: X -> dL_dX
    fn gradName(self: *Autodiff, name: []const u8) ![]const u8 {
        return std.fmt.allocPrint(self.allocator, "dL_d{s}", .{name});
    }

    /// Check if tensor is a parameter we're differentiating w.r.t.
    fn isParam(self: *Autodiff, name: []const u8, params: []const []const u8) bool {
        _ = self;
        for (params) |p| {
            if (std.mem.eql(u8, name, p)) return true;
        }
        return false;
    }

    /// Check if tensor has a path to a parameter (for chain rule)
    fn hasPathToParam(self: *Autodiff, name: []const u8, params: []const []const u8) bool {
        if (self.isParam(name, params)) return true;
        if (self.nodes.get(name)) |node| {
            for (node.inputs) |input| {
                if (self.hasPathToParam(input, params)) return true;
            }
        }
        return false;
    }
};

/// Gradient equation representation
pub const GradEquation = struct {
    output: []const u8, // e.g., "dL_dW"
    rule: GradRule,
    grad_upstream: []const u8, // e.g., "dL_dY"
    operands: []const []const u8,
    original_eq: *const ast.Equation,
};

/// Gradient computation rules
pub const GradRule = enum {
    pass_through, // dL/dX = dL/dY (for identity/addition)
    matmul_left, // dL/dA = dL/dC * B^T (2D matmul)
    matmul_right, // dL/dB = A^T * dL/dC (2D matmul)
    matmul_self, // dL/dY from L = Y*Y -> 2*Y
    dot_product_left, // dL/dA[i] = dL/dY * B[i] (1D dot product)
    dot_product_right, // dL/dB[i] = dL/dY * A[i] (1D dot product)
    relu_grad, // dL/dX = dL/dY * step(X)
    sigmoid_grad, // dL/dX = dL/dY * Y * (1-Y)
    softmax_grad, // Complex Jacobian
    tanh_grad, // dL/dX = dL/dY * (1 - Y^2)
    exp_grad, // dL/dX = dL/dY * exp(X)
    log_grad, // dL/dX = dL/dY / X
};
