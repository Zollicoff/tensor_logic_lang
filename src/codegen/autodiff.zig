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

// =============================================================================
// Gradient Codegen - LLVM IR generation for gradient equations
// =============================================================================

const types = @import("types.zig");
const TensorInfo = types.TensorInfo;

/// Codegen context type (forward declaration to avoid circular import)
pub const CodegenContext = @import("llvm.zig").LLVMCodegen;

/// Generate LLVM IR for a gradient equation
pub fn genGradEquation(ctx: *CodegenContext, grad_eq: *const GradEquation) !void {
    try ctx.emitFmt("    ; Gradient: {s} ({s})\n", .{ grad_eq.output, @tagName(grad_eq.rule) });

    // Ensure output gradient tensor exists
    if (!ctx.tensors.contains(grad_eq.output)) {
        // Get the tensor we're computing gradient for (remove dL_d prefix)
        const base_name = if (std.mem.startsWith(u8, grad_eq.output, "dL_d"))
            grad_eq.output[4..]
        else
            grad_eq.output;

        if (ctx.tensors.get(base_name)) |info| {
            const ptr = try ctx.newTemp();
            try ctx.emitFmt("    {s} = call ptr @calloc(i64 {d}, i64 8)\n", .{ ptr, info.totalSize() });
            try ctx.tensors.put(ctx.allocator, grad_eq.output, .{
                .name = grad_eq.output,
                .llvm_ptr = ptr,
                .rank = info.rank,
                .dims = try ctx.allocator.dupe(usize, info.dims),
                .strides = try ctx.allocator.dupe(usize, info.strides),
            });
        }
    }

    switch (grad_eq.rule) {
        .pass_through => try genGradPassThrough(ctx, grad_eq),
        .matmul_self => try genGradMatmulSelf(ctx, grad_eq),
        .relu_grad => try genGradRelu(ctx, grad_eq),
        .sigmoid_grad => try genGradSigmoid(ctx, grad_eq),
        .matmul_left => try genGradMatmulLeft(ctx, grad_eq),
        .matmul_right => try genGradMatmulRight(ctx, grad_eq),
        .tanh_grad => try genGradTanh(ctx, grad_eq),
        .exp_grad => try genGradExp(ctx, grad_eq),
        .log_grad => try genGradLog(ctx, grad_eq),
        .softmax_grad => try genGradSoftmax(ctx, grad_eq),
        else => {
            try ctx.emitFmt("    ; TODO: gradient rule {s}\n", .{@tagName(grad_eq.rule)});
        },
    }
}

fn genGradPassThrough(ctx: *CodegenContext, grad_eq: *const GradEquation) !void {
    const upstream_info = ctx.tensors.get(grad_eq.grad_upstream) orelse return;
    const output_info = ctx.tensors.get(grad_eq.output) orelse return;

    const total = upstream_info.totalSize();
    for (0..total) |i| {
        const src_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ src_ptr, upstream_info.llvm_ptr, i });
        const val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ val, src_ptr });

        const dst_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ dst_ptr, output_info.llvm_ptr, i });
        const old_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, dst_ptr });
        const new_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, val });
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, dst_ptr });
    }
}

fn genGradRelu(ctx: *CodegenContext, grad_eq: *const GradEquation) !void {
    const upstream_info = ctx.tensors.get(grad_eq.grad_upstream) orelse return;
    const output_info = ctx.tensors.get(grad_eq.output) orelse return;
    const input_info = ctx.tensors.get(grad_eq.operands[0]) orelse return;

    const total = upstream_info.totalSize();
    for (0..total) |i| {
        const up_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ up_ptr, upstream_info.llvm_ptr, i });
        const up_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ up_val, up_ptr });

        const in_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ in_ptr, input_info.llvm_ptr, i });
        const in_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ in_val, in_ptr });

        const cmp = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fcmp ogt double {s}, 0.0\n", .{ cmp, in_val });
        const step_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = select i1 {s}, double 1.0, double 0.0\n", .{ step_val, cmp });

        const grad = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ grad, up_val, step_val });

        const out_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ out_ptr, output_info.llvm_ptr, i });
        const old_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, out_ptr });
        const new_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, grad });
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, out_ptr });
    }
}

fn genGradSigmoid(ctx: *CodegenContext, grad_eq: *const GradEquation) !void {
    const upstream_info = ctx.tensors.get(grad_eq.grad_upstream) orelse return;
    const output_info = ctx.tensors.get(grad_eq.output) orelse return;
    const sigmoid_output_info = ctx.tensors.get(grad_eq.operands[1]) orelse return;

    const total = upstream_info.totalSize();
    for (0..total) |i| {
        const up_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ up_ptr, upstream_info.llvm_ptr, i });
        const up_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ up_val, up_ptr });

        const y_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ y_ptr, sigmoid_output_info.llvm_ptr, i });
        const y_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ y_val, y_ptr });

        const one_minus_y = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fsub double 1.0, {s}\n", .{ one_minus_y, y_val });
        const deriv = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ deriv, y_val, one_minus_y });

        const grad = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ grad, up_val, deriv });

        const out_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ out_ptr, output_info.llvm_ptr, i });
        const old_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, out_ptr });
        const new_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, grad });
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, out_ptr });
    }
}

fn genGradTanh(ctx: *CodegenContext, grad_eq: *const GradEquation) !void {
    const upstream_info = ctx.tensors.get(grad_eq.grad_upstream) orelse return;
    const output_info = ctx.tensors.get(grad_eq.output) orelse return;
    const tanh_output_info = ctx.tensors.get(grad_eq.operands[1]) orelse return;

    const total = upstream_info.totalSize();
    for (0..total) |i| {
        const up_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ up_ptr, upstream_info.llvm_ptr, i });
        const up_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ up_val, up_ptr });

        const y_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ y_ptr, tanh_output_info.llvm_ptr, i });
        const y_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ y_val, y_ptr });

        const y_sq = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ y_sq, y_val, y_val });
        const deriv = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fsub double 1.0, {s}\n", .{ deriv, y_sq });

        const grad = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ grad, up_val, deriv });

        const out_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ out_ptr, output_info.llvm_ptr, i });
        const old_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, out_ptr });
        const new_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, grad });
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, out_ptr });
    }
}

fn genGradExp(ctx: *CodegenContext, grad_eq: *const GradEquation) !void {
    const upstream_info = ctx.tensors.get(grad_eq.grad_upstream) orelse return;
    const output_info = ctx.tensors.get(grad_eq.output) orelse return;
    const exp_output_info = ctx.tensors.get(grad_eq.operands[1]) orelse return;

    const total = upstream_info.totalSize();
    for (0..total) |i| {
        const up_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ up_ptr, upstream_info.llvm_ptr, i });
        const up_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ up_val, up_ptr });

        const y_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ y_ptr, exp_output_info.llvm_ptr, i });
        const y_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ y_val, y_ptr });

        const grad = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ grad, up_val, y_val });

        const out_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ out_ptr, output_info.llvm_ptr, i });
        const old_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, out_ptr });
        const new_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, grad });
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, out_ptr });
    }
}

fn genGradLog(ctx: *CodegenContext, grad_eq: *const GradEquation) !void {
    const upstream_info = ctx.tensors.get(grad_eq.grad_upstream) orelse return;
    const output_info = ctx.tensors.get(grad_eq.output) orelse return;
    const input_info = ctx.tensors.get(grad_eq.operands[0]) orelse return;

    const total = upstream_info.totalSize();
    for (0..total) |i| {
        const up_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ up_ptr, upstream_info.llvm_ptr, i });
        const up_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ up_val, up_ptr });

        const x_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ x_ptr, input_info.llvm_ptr, i });
        const x_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ x_val, x_ptr });

        const grad = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fdiv double {s}, {s}\n", .{ grad, up_val, x_val });

        const out_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ out_ptr, output_info.llvm_ptr, i });
        const old_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, out_ptr });
        const new_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, grad });
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, out_ptr });
    }
}

fn genGradSoftmax(ctx: *CodegenContext, grad_eq: *const GradEquation) !void {
    const upstream_info = ctx.tensors.get(grad_eq.grad_upstream) orelse return;
    const output_info = ctx.tensors.get(grad_eq.output) orelse return;
    const softmax_output_info = ctx.tensors.get(grad_eq.operands[1]) orelse return;

    const total = upstream_info.totalSize();

    // First compute sum_j(dL/dY[j] * Y[j])
    const dot_sum = try ctx.newTemp();
    try ctx.emitFmt("    {s} = alloca double\n", .{dot_sum});
    try ctx.emitFmt("    store double 0.0, ptr {s}\n", .{dot_sum});

    for (0..total) |i| {
        const up_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ up_ptr, upstream_info.llvm_ptr, i });
        const up_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ up_val, up_ptr });

        const y_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ y_ptr, softmax_output_info.llvm_ptr, i });
        const y_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ y_val, y_ptr });

        const prod = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ prod, up_val, y_val });

        const old_sum = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_sum, dot_sum });
        const new_sum = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_sum, old_sum, prod });
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_sum, dot_sum });
    }

    const sum_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ sum_val, dot_sum });

    for (0..total) |i| {
        const up_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ up_ptr, upstream_info.llvm_ptr, i });
        const up_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ up_val, up_ptr });

        const y_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ y_ptr, softmax_output_info.llvm_ptr, i });
        const y_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ y_val, y_ptr });

        const diff = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fsub double {s}, {s}\n", .{ diff, up_val, sum_val });

        const grad = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ grad, y_val, diff });

        const out_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ out_ptr, output_info.llvm_ptr, i });
        const old_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, out_ptr });
        const new_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, grad });
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, out_ptr });
    }
}

fn genGradMatmulSelf(ctx: *CodegenContext, grad_eq: *const GradEquation) !void {
    const output_info = ctx.tensors.get(grad_eq.output) orelse return;
    const input_name = grad_eq.operands[0];
    const input_info = ctx.tensors.get(input_name) orelse return;

    const total = input_info.totalSize();
    for (0..total) |i| {
        const y_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ y_ptr, input_info.llvm_ptr, i });
        const y_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ y_val, y_ptr });

        const grad = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fmul double {s}, 2.0\n", .{ grad, y_val });

        const out_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ out_ptr, output_info.llvm_ptr, i });
        const old_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, out_ptr });
        const new_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, grad });
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, out_ptr });
    }
}

fn genGradMatmulLeft(ctx: *CodegenContext, grad_eq: *const GradEquation) !void {
    const output_info = ctx.tensors.get(grad_eq.output) orelse return;
    const upstream_info = ctx.tensors.get(grad_eq.grad_upstream) orelse return;

    if (grad_eq.operands.len < 3) return;
    const b_name = grad_eq.operands[1];
    const b_info = ctx.tensors.get(b_name) orelse return;

    const upstream_is_scalar = upstream_info.totalSize() == 1;
    const output_is_1d = output_info.dims.len == 1;

    if (upstream_is_scalar and output_is_1d) {
        try genGradDotProductLeft(ctx, grad_eq, output_info, upstream_info, b_info);
        return;
    }

    const dim_i = if (output_info.dims.len > 0) output_info.dims[0] else 1;
    const dim_j = if (output_info.dims.len > 1) output_info.dims[1] else 1;
    const dim_k = if (upstream_info.dims.len > 1) upstream_info.dims[1] else 1;

    const i_var = try ctx.newTemp();
    const j_var = try ctx.newTemp();
    const k_var = try ctx.newTemp();
    try ctx.emitFmt("    {s} = alloca i64\n", .{i_var});
    try ctx.emitFmt("    {s} = alloca i64\n", .{j_var});
    try ctx.emitFmt("    {s} = alloca i64\n", .{k_var});

    const i_start = try ctx.newLabel();
    const i_body = try ctx.newLabel();
    const i_end = try ctx.newLabel();

    try ctx.emitFmt("    store i64 0, ptr {s}\n", .{i_var});
    try ctx.emitFmt("    br label %{s}\n", .{i_start});
    try ctx.emitFmt("{s}:\n", .{i_start});
    const i_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ i_val, i_var });
    const i_cmp = try ctx.newTemp();
    try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ i_cmp, i_val, dim_i });
    try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ i_cmp, i_body, i_end });
    try ctx.emitFmt("{s}:\n", .{i_body});

    const j_start = try ctx.newLabel();
    const j_body = try ctx.newLabel();
    const j_end = try ctx.newLabel();

    try ctx.emitFmt("    store i64 0, ptr {s}\n", .{j_var});
    try ctx.emitFmt("    br label %{s}\n", .{j_start});
    try ctx.emitFmt("{s}:\n", .{j_start});
    const j_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ j_val, j_var });
    const j_cmp = try ctx.newTemp();
    try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ j_cmp, j_val, dim_j });
    try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ j_cmp, j_body, j_end });
    try ctx.emitFmt("{s}:\n", .{j_body});

    const k_start = try ctx.newLabel();
    const k_body = try ctx.newLabel();
    const k_end = try ctx.newLabel();

    try ctx.emitFmt("    store i64 0, ptr {s}\n", .{k_var});
    try ctx.emitFmt("    br label %{s}\n", .{k_start});
    try ctx.emitFmt("{s}:\n", .{k_start});
    const k_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ k_val, k_var });
    const k_cmp = try ctx.newTemp();
    try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ k_cmp, k_val, dim_k });
    try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ k_cmp, k_body, k_end });
    try ctx.emitFmt("{s}:\n", .{k_body});

    const c_offset = try ctx.newTemp();
    try ctx.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ c_offset, i_val, dim_k });
    const c_offset2 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, {s}\n", .{ c_offset2, c_offset, k_val });
    const c_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ c_ptr, upstream_info.llvm_ptr, c_offset2 });
    const dc_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ dc_val, c_ptr });

    const b_offset = try ctx.newTemp();
    try ctx.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ b_offset, j_val, dim_k });
    const b_offset2 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, {s}\n", .{ b_offset2, b_offset, k_val });
    const b_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ b_ptr, b_info.llvm_ptr, b_offset2 });
    const b_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ b_val, b_ptr });

    const prod = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ prod, dc_val, b_val });

    const a_offset = try ctx.newTemp();
    try ctx.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ a_offset, i_val, dim_j });
    const a_offset2 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, {s}\n", .{ a_offset2, a_offset, j_val });
    const out_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ out_ptr, output_info.llvm_ptr, a_offset2 });
    const old_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, out_ptr });
    const new_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, prod });
    try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, out_ptr });

    const k_next = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ k_next, k_val });
    try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ k_next, k_var });
    try ctx.emitFmt("    br label %{s}\n", .{k_start});
    try ctx.emitFmt("{s}:\n", .{k_end});

    const j_next = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ j_next, j_val });
    try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ j_next, j_var });
    try ctx.emitFmt("    br label %{s}\n", .{j_start});
    try ctx.emitFmt("{s}:\n", .{j_end});

    const i_next = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ i_next, i_val });
    try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ i_next, i_var });
    try ctx.emitFmt("    br label %{s}\n", .{i_start});
    try ctx.emitFmt("{s}:\n", .{i_end});
}

fn genGradMatmulRight(ctx: *CodegenContext, grad_eq: *const GradEquation) !void {
    const output_info = ctx.tensors.get(grad_eq.output) orelse return;
    const upstream_info = ctx.tensors.get(grad_eq.grad_upstream) orelse return;

    if (grad_eq.operands.len < 3) return;
    const a_name = grad_eq.operands[0];
    const a_info = ctx.tensors.get(a_name) orelse return;

    const upstream_is_scalar = upstream_info.totalSize() == 1;
    const output_is_1d = output_info.dims.len == 1;

    if (upstream_is_scalar and output_is_1d) {
        try genGradDotProductRight(ctx, grad_eq, output_info, upstream_info, a_info);
        return;
    }

    const dim_i = if (a_info.dims.len > 0) a_info.dims[0] else 1;
    const dim_j = if (output_info.dims.len > 0) output_info.dims[0] else 1;
    const dim_k = if (output_info.dims.len > 1) output_info.dims[1] else 1;

    const i_var = try ctx.newTemp();
    const j_var = try ctx.newTemp();
    const k_var = try ctx.newTemp();
    try ctx.emitFmt("    {s} = alloca i64\n", .{i_var});
    try ctx.emitFmt("    {s} = alloca i64\n", .{j_var});
    try ctx.emitFmt("    {s} = alloca i64\n", .{k_var});

    const j_start = try ctx.newLabel();
    const j_body = try ctx.newLabel();
    const j_end = try ctx.newLabel();

    try ctx.emitFmt("    store i64 0, ptr {s}\n", .{j_var});
    try ctx.emitFmt("    br label %{s}\n", .{j_start});
    try ctx.emitFmt("{s}:\n", .{j_start});
    const j_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ j_val, j_var });
    const j_cmp = try ctx.newTemp();
    try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ j_cmp, j_val, dim_j });
    try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ j_cmp, j_body, j_end });
    try ctx.emitFmt("{s}:\n", .{j_body});

    const k_start = try ctx.newLabel();
    const k_body = try ctx.newLabel();
    const k_end = try ctx.newLabel();

    try ctx.emitFmt("    store i64 0, ptr {s}\n", .{k_var});
    try ctx.emitFmt("    br label %{s}\n", .{k_start});
    try ctx.emitFmt("{s}:\n", .{k_start});
    const k_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ k_val, k_var });
    const k_cmp = try ctx.newTemp();
    try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ k_cmp, k_val, dim_k });
    try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ k_cmp, k_body, k_end });
    try ctx.emitFmt("{s}:\n", .{k_body});

    const i_start = try ctx.newLabel();
    const i_body = try ctx.newLabel();
    const i_end = try ctx.newLabel();

    try ctx.emitFmt("    store i64 0, ptr {s}\n", .{i_var});
    try ctx.emitFmt("    br label %{s}\n", .{i_start});
    try ctx.emitFmt("{s}:\n", .{i_start});
    const i_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ i_val, i_var });
    const i_cmp = try ctx.newTemp();
    try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ i_cmp, i_val, dim_i });
    try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ i_cmp, i_body, i_end });
    try ctx.emitFmt("{s}:\n", .{i_body});

    const a_offset = try ctx.newTemp();
    try ctx.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ a_offset, i_val, dim_j });
    const a_offset2 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, {s}\n", .{ a_offset2, a_offset, j_val });
    const a_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ a_ptr, a_info.llvm_ptr, a_offset2 });
    const a_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ a_val, a_ptr });

    const c_offset = try ctx.newTemp();
    try ctx.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ c_offset, i_val, dim_k });
    const c_offset2 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, {s}\n", .{ c_offset2, c_offset, k_val });
    const c_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ c_ptr, upstream_info.llvm_ptr, c_offset2 });
    const dc_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ dc_val, c_ptr });

    const prod = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ prod, a_val, dc_val });

    const b_offset = try ctx.newTemp();
    try ctx.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ b_offset, j_val, dim_k });
    const b_offset2 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, {s}\n", .{ b_offset2, b_offset, k_val });
    const out_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ out_ptr, output_info.llvm_ptr, b_offset2 });
    const old_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, out_ptr });
    const new_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, prod });
    try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, out_ptr });

    const i_next = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ i_next, i_val });
    try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ i_next, i_var });
    try ctx.emitFmt("    br label %{s}\n", .{i_start});
    try ctx.emitFmt("{s}:\n", .{i_end});

    const k_next = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ k_next, k_val });
    try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ k_next, k_var });
    try ctx.emitFmt("    br label %{s}\n", .{k_start});
    try ctx.emitFmt("{s}:\n", .{k_end});

    const j_next = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ j_next, j_val });
    try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ j_next, j_var });
    try ctx.emitFmt("    br label %{s}\n", .{j_start});
    try ctx.emitFmt("{s}:\n", .{j_end});
}

fn genGradDotProductLeft(
    ctx: *CodegenContext,
    grad_eq: *const GradEquation,
    output_info: TensorInfo,
    upstream_info: TensorInfo,
    b_info: TensorInfo,
) !void {
    _ = grad_eq;

    const dy_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 0\n", .{ dy_ptr, upstream_info.llvm_ptr });
    const dy_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ dy_val, dy_ptr });

    const dim = output_info.dims[0];
    for (0..dim) |i| {
        const b_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ b_ptr, b_info.llvm_ptr, i });
        const b_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ b_val, b_ptr });

        const grad = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ grad, dy_val, b_val });

        const out_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ out_ptr, output_info.llvm_ptr, i });
        const old_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, out_ptr });
        const new_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, grad });
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, out_ptr });
    }
}

fn genGradDotProductRight(
    ctx: *CodegenContext,
    grad_eq: *const GradEquation,
    output_info: TensorInfo,
    upstream_info: TensorInfo,
    a_info: TensorInfo,
) !void {
    _ = grad_eq;

    const dy_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 0\n", .{ dy_ptr, upstream_info.llvm_ptr });
    const dy_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ dy_val, dy_ptr });

    const dim = output_info.dims[0];
    for (0..dim) |i| {
        const a_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ a_ptr, a_info.llvm_ptr, i });
        const a_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ a_val, a_ptr });

        const grad = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ grad, dy_val, a_val });

        const out_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ out_ptr, output_info.llvm_ptr, i });
        const old_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, out_ptr });
        const new_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, grad });
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, out_ptr });
    }
}
