// Expression evaluation for LLVM code generation
//
// Handles generation of LLVM IR for expressions: literals, tensor access,
// binary operations, products, unary operations, and nonlinearities.

const std = @import("std");
const ast = @import("../frontend/ast.zig");
const tensor = @import("tensor.zig");

/// Codegen context (forward declaration)
pub const CodegenContext = @import("llvm.zig").LLVMCodegen;

/// Generate expression evaluation
pub fn genExpr(ctx: *CodegenContext, expr: *const ast.Expr, loop_vars: *std.StringHashMapUnmanaged([]const u8)) error{OutOfMemory}![]const u8 {
    switch (expr.*) {
        .literal => |lit| return genLiteral(ctx, lit),
        .tensor_ref => |ref| return genTensorAccess(ctx, ref, loop_vars),
        .binary => |bin| return genBinary(ctx, bin, loop_vars),
        .product => |prod| return genProduct(ctx, prod, loop_vars),
        .nonlinearity => |nl| return genNonlinearity(ctx, nl, loop_vars),
        .unary => |un| return genUnary(ctx, un, loop_vars),
        .group => |g| return genExpr(ctx, g, loop_vars),
        else => return "0.0",
    }
}

/// Generate literal value
pub fn genLiteral(ctx: *CodegenContext, lit: ast.Literal) ![]const u8 {
    const arena = ctx.string_arena.allocator();
    return switch (lit) {
        .float => |f| blk: {
            // Ensure float literals always have decimal point for LLVM
            const str = try std.fmt.allocPrint(arena, "{d}", .{f});
            // Check if it has a decimal point
            for (str) |c| {
                if (c == '.' or c == 'e' or c == 'E') break :blk str;
            }
            // No decimal point, add .0
            break :blk try std.fmt.allocPrint(arena, "{d}.0", .{f});
        },
        .integer => |i| try std.fmt.allocPrint(arena, "{d}.0", .{i}),
        .boolean => |b| if (b) "1.0" else "0.0",
        .string => "0.0",
    };
}

/// Generate tensor element access
pub fn genTensorAccess(ctx: *CodegenContext, ref: ast.TensorRef, loop_vars: *std.StringHashMapUnmanaged([]const u8)) error{OutOfMemory}![]const u8 {
    const info = ctx.tensors.get(ref.name) orelse {
        // Tensor not yet allocated - treat as zero
        return "0.0";
    };

    // Build index values
    var idx_vals = std.ArrayListUnmanaged([]const u8){};
    defer idx_vals.deinit(ctx.allocator);

    for (ref.indices) |idx| {
        switch (idx) {
            .name => |name| {
                if (loop_vars.get(name)) |var_name| {
                    const val = try ctx.newTemp();
                    try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ val, var_name });
                    try idx_vals.append(ctx.allocator, val);
                } else {
                    try idx_vals.append(ctx.allocator, "0");
                }
            },
            .constant => |c| {
                try idx_vals.append(ctx.allocator, try std.fmt.allocPrint(ctx.string_arena.allocator(), "{d}", .{c}));
            },
            .arithmetic => |arith| {
                // Handle i+1, i-1, etc.
                if (loop_vars.get(arith.base)) |var_name| {
                    const base_val = try ctx.newTemp();
                    try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ base_val, var_name });
                    const result_val = try ctx.newTemp();
                    switch (arith.op) {
                        .add => try ctx.emitFmt("    {s} = add i64 {s}, {d}\n", .{ result_val, base_val, arith.offset }),
                        .sub => try ctx.emitFmt("    {s} = sub i64 {s}, {d}\n", .{ result_val, base_val, arith.offset }),
                    }
                    try idx_vals.append(ctx.allocator, result_val);
                } else {
                    try idx_vals.append(ctx.allocator, try std.fmt.allocPrint(ctx.string_arena.allocator(), "{d}", .{arith.offset}));
                }
            },
            else => try idx_vals.append(ctx.allocator, "0"),
        }
    }

    const offset = try tensor.computeLinearOffset(ctx, info.strides, idx_vals.items);
    const ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ ptr, info.llvm_ptr, offset });
    const val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ val, ptr });
    return val;
}

/// Generate binary operation
pub fn genBinary(ctx: *CodegenContext, bin: anytype, loop_vars: *std.StringHashMapUnmanaged([]const u8)) error{OutOfMemory}![]const u8 {
    const left = try genExpr(ctx, bin.left, loop_vars);
    const right = try genExpr(ctx, bin.right, loop_vars);
    const result = try ctx.newTemp();

    switch (bin.op) {
        .add => try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ result, left, right }),
        .sub => try ctx.emitFmt("    {s} = fsub double {s}, {s}\n", .{ result, left, right }),
        .mul => try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ result, left, right }),
        .div => try ctx.emitFmt("    {s} = fdiv double {s}, {s}\n", .{ result, left, right }),
        .pow => try ctx.emitFmt("    {s} = call double @llvm.pow.f64(double {s}, double {s})\n", .{ result, left, right }),
        .gt => {
            const cmp = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fcmp ogt double {s}, {s}\n", .{ cmp, left, right });
            try ctx.emitFmt("    {s} = uitofp i1 {s} to double\n", .{ result, cmp });
        },
        .lt => {
            const cmp = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fcmp olt double {s}, {s}\n", .{ cmp, left, right });
            try ctx.emitFmt("    {s} = uitofp i1 {s} to double\n", .{ result, cmp });
        },
        .ge => {
            const cmp = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fcmp oge double {s}, {s}\n", .{ cmp, left, right });
            try ctx.emitFmt("    {s} = uitofp i1 {s} to double\n", .{ result, cmp });
        },
        .le => {
            const cmp = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fcmp ole double {s}, {s}\n", .{ cmp, left, right });
            try ctx.emitFmt("    {s} = uitofp i1 {s} to double\n", .{ result, cmp });
        },
        .eq => {
            const cmp = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fcmp oeq double {s}, {s}\n", .{ cmp, left, right });
            try ctx.emitFmt("    {s} = uitofp i1 {s} to double\n", .{ result, cmp });
        },
        .ne => {
            const cmp = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fcmp one double {s}, {s}\n", .{ cmp, left, right });
            try ctx.emitFmt("    {s} = uitofp i1 {s} to double\n", .{ result, cmp });
        },
        else => try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ result, left, right }),
    }
    return result;
}

/// Generate product of factors
pub fn genProduct(ctx: *CodegenContext, prod: anytype, loop_vars: *std.StringHashMapUnmanaged([]const u8)) error{OutOfMemory}![]const u8 {
    if (prod.factors.len == 0) return "1.0";

    var result = try genExpr(ctx, prod.factors[0], loop_vars);

    for (prod.factors[1..]) |factor| {
        const right = try genExpr(ctx, factor, loop_vars);
        const new_result = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ new_result, result, right });
        result = new_result;
    }

    return result;
}

/// Generate unary operation
pub fn genUnary(ctx: *CodegenContext, un: anytype, loop_vars: *std.StringHashMapUnmanaged([]const u8)) error{OutOfMemory}![]const u8 {
    const operand = try genExpr(ctx, un.operand, loop_vars);
    const result = try ctx.newTemp();

    switch (un.op) {
        .negate => try ctx.emitFmt("    {s} = fneg double {s}\n", .{ result, operand }),
        .not => {
            // Logical not: 0 -> 1, nonzero -> 0
            const cmp = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fcmp oeq double {s}, 0.0\n", .{ cmp, operand });
            try ctx.emitFmt("    {s} = uitofp i1 {s} to double\n", .{ result, cmp });
        },
    }
    return result;
}

/// Generate nonlinearity (activation function)
pub fn genNonlinearity(ctx: *CodegenContext, nl: anytype, loop_vars: *std.StringHashMapUnmanaged([]const u8)) error{OutOfMemory}![]const u8 {
    const arg = try genExpr(ctx, nl.arg, loop_vars);
    const result = try ctx.newTemp();

    switch (nl.func) {
        .relu => {
            const cmp = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fcmp ogt double {s}, 0.0\n", .{ cmp, arg });
            try ctx.emitFmt("    {s} = select i1 {s}, double {s}, double 0.0\n", .{ result, cmp, arg });
        },
        .sigmoid => {
            const neg = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fneg double {s}\n", .{ neg, arg });
            const exp_val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = call double @llvm.exp.f64(double {s})\n", .{ exp_val, neg });
            const denom = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fadd double 1.0, {s}\n", .{ denom, exp_val });
            try ctx.emitFmt("    {s} = fdiv double 1.0, {s}\n", .{ result, denom });
        },
        .tanh => {
            // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
            const two_x = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fmul double {s}, 2.0\n", .{ two_x, arg });
            const exp_2x = try ctx.newTemp();
            try ctx.emitFmt("    {s} = call double @llvm.exp.f64(double {s})\n", .{ exp_2x, two_x });
            const num = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fsub double {s}, 1.0\n", .{ num, exp_2x });
            const denom = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fadd double {s}, 1.0\n", .{ denom, exp_2x });
            try ctx.emitFmt("    {s} = fdiv double {s}, {s}\n", .{ result, num, denom });
        },
        .step => {
            const cmp = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fcmp ogt double {s}, 0.0\n", .{ cmp, arg });
            try ctx.emitFmt("    {s} = select i1 {s}, double 1.0, double 0.0\n", .{ result, cmp });
        },
        .exp => try ctx.emitFmt("    {s} = call double @llvm.exp.f64(double {s})\n", .{ result, arg }),
        .log => try ctx.emitFmt("    {s} = call double @llvm.log.f64(double {s})\n", .{ result, arg }),
        .sqrt => try ctx.emitFmt("    {s} = call double @llvm.sqrt.f64(double {s})\n", .{ result, arg }),
        .abs => try ctx.emitFmt("    {s} = call double @llvm.fabs.f64(double {s})\n", .{ result, arg }),
        .sin => try ctx.emitFmt("    {s} = call double @llvm.sin.f64(double {s})\n", .{ result, arg }),
        .cos => try ctx.emitFmt("    {s} = call double @llvm.cos.f64(double {s})\n", .{ result, arg }),
        else => try ctx.emitFmt("    {s} = fadd double {s}, 0.0\n", .{ result, arg }),
    }
    return result;
}

/// Collect all named indices from an expression
pub fn collectExprIndices(ctx: *CodegenContext, expr_ptr: *const ast.Expr, indices: *std.StringHashMapUnmanaged(usize)) !void {
    switch (expr_ptr.*) {
        .tensor_ref => |ref| {
            for (ref.indices) |idx| {
                switch (idx) {
                    .name => |name| {
                        const size = ctx.domains.get(name) orelse 10;
                        try indices.put(ctx.allocator, name, size);
                    },
                    .arithmetic => |arith| {
                        // Arithmetic index like i+1 - collect the base variable
                        const size = ctx.domains.get(arith.base) orelse 10;
                        try indices.put(ctx.allocator, arith.base, size);
                    },
                    else => {},
                }
            }
        },
        .product => |prod| {
            for (prod.factors) |factor| {
                try collectExprIndices(ctx, factor, indices);
            }
        },
        .binary => |bin| {
            try collectExprIndices(ctx, bin.left, indices);
            try collectExprIndices(ctx, bin.right, indices);
        },
        .nonlinearity => |nl| {
            try collectExprIndices(ctx, nl.arg, indices);
        },
        .unary => |un| {
            try collectExprIndices(ctx, un.operand, indices);
        },
        .group => |g| {
            try collectExprIndices(ctx, g, indices);
        },
        else => {},
    }
}
