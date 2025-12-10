// Backward chaining for LLVM code generation
//
// Compiles tensor equations to RECURSIVE FUNCTIONS instead of loops.
// This is demand-driven inference: only computes what's queried.
//
// Forward chaining (einsum.zig): Y? → loops over all i,j → computes everything
// Backward chaining (this file): Y[0,5]? → recursive call → computes only Y[0,5]
//
// Key insight from paper: Both modes compile to native LLVM code.
// - Forward = nested loops (optimal for dense/neural)
// - Backward = recursive functions + memoization (optimal for sparse/symbolic)

const std = @import("std");
const ast = @import("../frontend/ast.zig");
const types = @import("types.zig");
const tensor = @import("tensor.zig");
const expr = @import("expr.zig");

const TensorInfo = types.TensorInfo;

/// Codegen context (forward declaration)
pub const CodegenContext = @import("llvm.zig").LLVMCodegen;

/// Information about a backward-chaining function
pub const BackwardFn = struct {
    tensor_name: []const u8,
    fn_name: []const u8,
    rank: usize,
    dims: []const usize,
    /// Indices into program.statements for equations that define this tensor
    equation_indices: std.ArrayListUnmanaged(usize),
};

/// Check if a query should use backward chaining
/// Backward chaining is used when query has CONSTANT indices: Tensor[0,5]?
/// Forward chaining is used when query has NO indices or VARIABLE indices: Tensor?
pub fn isBackwardQuery(query: *const ast.Query) bool {
    // If query has specific constant indices, use backward chaining
    for (query.tensor.indices) |idx| {
        switch (idx) {
            .constant => return true, // Has at least one constant index
            else => {},
        }
    }
    return false;
}

/// Generate backward chaining infrastructure for a program
/// This creates:
/// 1. Memoization tables for each tensor
/// 2. "Computed" flag arrays
/// 3. Recursive compute functions
pub fn genBackwardInfra(ctx: *CodegenContext, program: *const ast.Program) !void {
    // Collect all tensors that need backward chaining functions
    var backward_tensors = std.StringHashMap(BackwardFn).init(ctx.allocator);
    defer {
        var iter = backward_tensors.valueIterator();
        while (iter.next()) |fn_info| {
            fn_info.equation_indices.deinit(ctx.allocator);
        }
        backward_tensors.deinit();
    }

    // Find all backward queries
    var needs_backward = std.StringHashMap(void).init(ctx.allocator);
    defer needs_backward.deinit();

    for (program.statements) |stmt| {
        if (stmt == .query) {
            if (isBackwardQuery(&stmt.query)) {
                try needs_backward.put(stmt.query.tensor.name, {});
            }
        }
    }

    if (needs_backward.count() == 0) return;

    // Collect equations for each tensor that needs backward chaining
    for (program.statements, 0..) |stmt, stmt_idx| {
        if (stmt == .equation) {
            const eq = stmt.equation;
            if (needs_backward.contains(eq.lhs.name)) {
                var entry = try backward_tensors.getOrPut(eq.lhs.name);
                if (!entry.found_existing) {
                    const info = ctx.tensors.get(eq.lhs.name) orelse continue;
                    entry.value_ptr.* = .{
                        .tensor_name = eq.lhs.name,
                        .fn_name = try std.fmt.allocPrint(ctx.string_arena.allocator(), "compute_{s}", .{eq.lhs.name}),
                        .rank = info.rank,
                        .dims = info.dims,
                        .equation_indices = std.ArrayListUnmanaged(usize){},
                    };
                }
                try entry.value_ptr.equation_indices.append(ctx.allocator, stmt_idx);
            }
        }
    }

    // Generate memoization tables and compute functions
    var fn_iter = backward_tensors.valueIterator();
    while (fn_iter.next()) |fn_info| {
        try genMemoTable(ctx, fn_info);
        try genComputeFunction(ctx, fn_info, program);
    }
}

/// Generate memoization table for a tensor
fn genMemoTable(ctx: *CodegenContext, fn_info: *const BackwardFn) !void {
    const info = ctx.tensors.get(fn_info.tensor_name) orelse return;
    const total_size = info.totalSize();

    // Emit global memo table declaration
    try ctx.emitFmt("\n; Memoization for {s}\n", .{fn_info.tensor_name});
    try ctx.emitFmt("@memo_{s} = internal global [{d} x double] zeroinitializer\n", .{ fn_info.tensor_name, total_size });
    try ctx.emitFmt("@memo_{s}_computed = internal global [{d} x i1] zeroinitializer\n", .{ fn_info.tensor_name, total_size });
}

/// Generate recursive compute function for a tensor
fn genComputeFunction(ctx: *CodegenContext, fn_info: *const BackwardFn, program: *const ast.Program) !void {
    if (fn_info.equation_indices.items.len == 0) return;

    const info = ctx.tensors.get(fn_info.tensor_name) orelse return;

    // Function signature: double @compute_Tensor(i64 %i0, i64 %i1, ...)
    var sig = std.ArrayListUnmanaged(u8){};
    defer sig.deinit(ctx.allocator);

    try sig.appendSlice(ctx.allocator, "define double @");
    try sig.appendSlice(ctx.allocator, fn_info.fn_name);
    try sig.append(ctx.allocator, '(');

    for (0..fn_info.rank) |i| {
        if (i > 0) try sig.appendSlice(ctx.allocator, ", ");
        try sig.writer(ctx.allocator).print("i64 %idx{d}", .{i});
    }
    try sig.appendSlice(ctx.allocator, ") {\n");
    try ctx.emit(sig.items);

    // Entry block
    try ctx.emit("entry:\n");

    // Compute linear offset for memo lookup
    const offset = try genLinearOffset(ctx, info.strides, fn_info.rank);

    // Check if already computed
    const computed_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr [{d} x i1], ptr @memo_{s}_computed, i64 0, i64 {s}\n", .{ computed_ptr, info.totalSize(), fn_info.tensor_name, offset });
    const is_computed = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load i1, ptr {s}\n", .{ is_computed, computed_ptr });
    try ctx.emitFmt("    br i1 {s}, label %return_memo, label %compute\n", .{is_computed});

    // Return memoized value
    try ctx.emit("\nreturn_memo:\n");
    const memo_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr [{d} x double], ptr @memo_{s}, i64 0, i64 {s}\n", .{ memo_ptr, info.totalSize(), fn_info.tensor_name, offset });
    const memo_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ memo_val, memo_ptr });
    try ctx.emitFmt("    ret double {s}\n", .{memo_val});

    // Compute block - evaluate all equations
    try ctx.emit("\ncompute:\n");

    // CRITICAL: Mark as computed FIRST to prevent infinite recursion
    // For transitive closure: we mark with initial value 0, evaluate base case,
    // then add recursive contributions. Recursive calls will see memo_computed=true
    // and return the current value (which gets updated as we go).
    const init_computed_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr [{d} x i1], ptr @memo_{s}_computed, i64 0, i64 {s}\n", .{ init_computed_ptr, info.totalSize(), fn_info.tensor_name, offset });
    try ctx.emitFmt("    store i1 true, ptr {s}\n", .{init_computed_ptr});

    // Initialize result accumulator
    const result_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = alloca double\n", .{result_ptr});
    try ctx.emitFmt("    store double 0.0, ptr {s}\n", .{result_ptr});

    // Store initial value in memo (so recursive calls see 0 initially)
    const init_memo_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr [{d} x double], ptr @memo_{s}, i64 0, i64 {s}\n", .{ init_memo_ptr, info.totalSize(), fn_info.tensor_name, offset });
    try ctx.emitFmt("    store double 0.0, ptr {s}\n", .{init_memo_ptr});

    // Evaluate all equations and accumulate
    for (fn_info.equation_indices.items) |stmt_idx| {
        const eq = &program.statements[stmt_idx].equation;
        const eq_result = try genBackwardEquation(ctx, eq, fn_info.rank);

        // Accumulate based on operator
        const old_result = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_result, result_ptr });

        const new_result = try ctx.newTemp();
        switch (eq.op) {
            .add => try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_result, old_result, eq_result }),
            .max => {
                const cmp = try ctx.newTemp();
                try ctx.emitFmt("    {s} = fcmp ogt double {s}, {s}\n", .{ cmp, eq_result, old_result });
                try ctx.emitFmt("    {s} = select i1 {s}, double {s}, double {s}\n", .{ new_result, cmp, eq_result, old_result });
            },
            .min => {
                const cmp = try ctx.newTemp();
                try ctx.emitFmt("    {s} = fcmp olt double {s}, {s}\n", .{ cmp, eq_result, old_result });
                try ctx.emitFmt("    {s} = select i1 {s}, double {s}, double {s}\n", .{ new_result, cmp, eq_result, old_result });
            },
            .mul => try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ new_result, old_result, eq_result }),
            else => try ctx.emitFmt("    {s} = fadd double 0.0, {s}\n", .{ new_result, eq_result }), // assign
        }
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_result, result_ptr });

        // Update memo with intermediate result (important for transitive closure)
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_result, init_memo_ptr });
    }

    // Return final result
    const final_result = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ final_result, result_ptr });
    try ctx.emitFmt("    ret double {s}\n", .{final_result});
    try ctx.emit("}\n\n");
}

/// Generate code for a single equation in backward mode
/// Returns the LLVM value containing the result
fn genBackwardEquation(ctx: *CodegenContext, eq: *const ast.Equation, _: usize) ![]const u8 {
    // Build mapping from LHS index names to function parameters
    var idx_to_param = std.StringHashMap([]const u8).init(ctx.allocator);
    defer idx_to_param.deinit();

    for (eq.lhs.indices, 0..) |idx, i| {
        const name = switch (idx) {
            .name => |n| n,
            else => continue,
        };
        const param = try std.fmt.allocPrint(ctx.string_arena.allocator(), "%idx{d}", .{i});
        try idx_to_param.put(name, param);
    }

    // Collect contracted indices (in RHS but not LHS)
    var contracted = std.StringHashMap(usize).init(ctx.allocator);
    defer contracted.deinit();

    try collectContracted(eq.rhs, &idx_to_param, &contracted, ctx);

    // If no contracted indices, simple evaluation
    if (contracted.count() == 0) {
        return try genBackwardExpr(ctx, eq.rhs, &idx_to_param);
    }

    // Generate sum loop over contracted indices
    return try genContractedSum(ctx, eq, &idx_to_param, &contracted);
}

/// Collect contracted indices from expression
fn collectContracted(
    expression: *const ast.Expr,
    bound: *std.StringHashMap([]const u8),
    contracted: *std.StringHashMap(usize),
    ctx: *CodegenContext,
) !void {
    switch (expression.*) {
        .tensor_ref => |ref| {
            for (ref.indices) |idx| {
                const name = switch (idx) {
                    .name => |n| n,
                    else => continue,
                };
                if (!bound.contains(name)) {
                    const size = ctx.domains.get(name) orelse 10;
                    try contracted.put(name, size);
                }
            }
        },
        .product => |prod| {
            for (prod.factors) |factor| {
                try collectContracted(factor, bound, contracted, ctx);
            }
        },
        .binary => |bin| {
            try collectContracted(bin.left, bound, contracted, ctx);
            try collectContracted(bin.right, bound, contracted, ctx);
        },
        .nonlinearity => |nl| try collectContracted(nl.arg, bound, contracted, ctx),
        .unary => |un| try collectContracted(un.operand, bound, contracted, ctx),
        .group => |g| try collectContracted(g, bound, contracted, ctx),
        else => {},
    }
}

/// Generate sum over contracted indices
fn genContractedSum(
    ctx: *CodegenContext,
    eq: *const ast.Equation,
    idx_to_param: *std.StringHashMap([]const u8),
    contracted: *std.StringHashMap(usize),
) ![]const u8 {
    // Allocate accumulator
    const accum = try ctx.newTemp();
    try ctx.emitFmt("    {s} = alloca double\n", .{accum});
    try ctx.emitFmt("    store double 0.0, ptr {s}\n", .{accum});

    // Allocate loop variables for contracted indices
    var loop_vars = std.StringHashMap([]const u8).init(ctx.allocator);
    defer loop_vars.deinit();

    // Copy bound indices to loop_vars
    var bound_iter = idx_to_param.iterator();
    while (bound_iter.next()) |entry| {
        try loop_vars.put(entry.key_ptr.*, entry.value_ptr.*);
    }

    // Create loop variables for contracted indices
    var contracted_list = std.ArrayListUnmanaged(struct { name: []const u8, size: usize, var_name: []const u8 }){};
    defer contracted_list.deinit(ctx.allocator);

    var iter = contracted.iterator();
    while (iter.next()) |entry| {
        const var_name = try std.fmt.allocPrint(ctx.string_arena.allocator(), "%bw_{s}", .{entry.key_ptr.*});
        try ctx.emitFmt("    {s} = alloca i64\n", .{var_name});
        try loop_vars.put(entry.key_ptr.*, var_name);
        try contracted_list.append(ctx.allocator, .{
            .name = entry.key_ptr.*,
            .size = entry.value_ptr.*,
            .var_name = var_name,
        });
    }

    // Generate nested loops
    var loop_labels = std.ArrayListUnmanaged(struct { start: []const u8, body: []const u8, end: []const u8 }){};
    defer loop_labels.deinit(ctx.allocator);

    for (contracted_list.items) |c| {
        const start = try ctx.newLabel();
        const body = try ctx.newLabel();
        const end = try ctx.newLabel();
        try loop_labels.append(ctx.allocator, .{ .start = start, .body = body, .end = end });

        try ctx.emitFmt("    store i64 0, ptr {s}\n", .{c.var_name});
        try ctx.emitFmt("    br label %{s}\n", .{start});
        try ctx.emitFmt("{s}:\n", .{start});

        const idx_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ idx_val, c.var_name });
        const cmp = try ctx.newTemp();
        try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ cmp, idx_val, c.size });
        try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ cmp, body, end });
        try ctx.emitFmt("{s}:\n", .{body});
    }

    // Evaluate RHS expression
    const rhs_val = try genBackwardExpr(ctx, eq.rhs, &loop_vars);

    // Accumulate
    const old_accum = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_accum, accum });
    const new_accum = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_accum, old_accum, rhs_val });
    try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_accum, accum });

    // Close loops
    var i: usize = contracted_list.items.len;
    while (i > 0) {
        i -= 1;
        const c = contracted_list.items[i];
        const labels = loop_labels.items[i];

        const idx_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ idx_val, c.var_name });
        const next = try ctx.newTemp();
        try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ next, idx_val });
        try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ next, c.var_name });
        try ctx.emitFmt("    br label %{s}\n", .{labels.start});
        try ctx.emitFmt("{s}:\n", .{labels.end});
    }

    // Return accumulated value
    const result = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ result, accum });
    return result;
}

/// Generate backward expression evaluation
/// This handles recursive calls to compute functions
fn genBackwardExpr(
    ctx: *CodegenContext,
    expression: *const ast.Expr,
    loop_vars: *std.StringHashMap([]const u8),
) ![]const u8 {
    switch (expression.*) {
        .tensor_ref => |ref| {
            return try genBackwardTensorAccess(ctx, &ref, loop_vars);
        },
        .literal => |lit| {
            return switch (lit) {
                .integer => |i| try std.fmt.allocPrint(ctx.string_arena.allocator(), "{d}.0", .{i}),
                .float => |f| try std.fmt.allocPrint(ctx.string_arena.allocator(), "{e}", .{f}),
                .boolean => |b| if (b) "1.0" else "0.0",
                else => "0.0",
            };
        },
        .product => |prod| {
            var result = try genBackwardExpr(ctx, prod.factors[0], loop_vars);
            for (prod.factors[1..]) |factor| {
                const factor_val = try genBackwardExpr(ctx, factor, loop_vars);
                const new_result = try ctx.newTemp();
                try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ new_result, result, factor_val });
                result = new_result;
            }
            return result;
        },
        .binary => |bin| {
            const left = try genBackwardExpr(ctx, bin.left, loop_vars);
            const right = try genBackwardExpr(ctx, bin.right, loop_vars);
            const result = try ctx.newTemp();
            switch (bin.op) {
                .add => try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ result, left, right }),
                .sub => try ctx.emitFmt("    {s} = fsub double {s}, {s}\n", .{ result, left, right }),
                .mul => try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ result, left, right }),
                .div => try ctx.emitFmt("    {s} = fdiv double {s}, {s}\n", .{ result, left, right }),
                else => try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ result, left, right }),
            }
            return result;
        },
        .nonlinearity => |nl| {
            const arg = try genBackwardExpr(ctx, nl.arg, loop_vars);
            const result = try ctx.newTemp();
            switch (nl.func) {
                .step => {
                    const cmp = try ctx.newTemp();
                    try ctx.emitFmt("    {s} = fcmp ogt double {s}, 0.0\n", .{ cmp, arg });
                    try ctx.emitFmt("    {s} = select i1 {s}, double 1.0, double 0.0\n", .{ result, cmp });
                },
                .relu => {
                    const cmp = try ctx.newTemp();
                    try ctx.emitFmt("    {s} = fcmp ogt double {s}, 0.0\n", .{ cmp, arg });
                    try ctx.emitFmt("    {s} = select i1 {s}, double {s}, double 0.0\n", .{ result, cmp, arg });
                },
                .sigmoid => {
                    const neg = try ctx.newTemp();
                    try ctx.emitFmt("    {s} = fneg double {s}\n", .{ neg, arg });
                    const exp_neg = try ctx.newTemp();
                    try ctx.emitFmt("    {s} = call double @llvm.exp.f64(double {s})\n", .{ exp_neg, neg });
                    const denom = try ctx.newTemp();
                    try ctx.emitFmt("    {s} = fadd double 1.0, {s}\n", .{ denom, exp_neg });
                    try ctx.emitFmt("    {s} = fdiv double 1.0, {s}\n", .{ result, denom });
                },
                .exp => try ctx.emitFmt("    {s} = call double @llvm.exp.f64(double {s})\n", .{ result, arg }),
                .log => try ctx.emitFmt("    {s} = call double @llvm.log.f64(double {s})\n", .{ result, arg }),
                .sqrt => try ctx.emitFmt("    {s} = call double @llvm.sqrt.f64(double {s})\n", .{ result, arg }),
                .abs => try ctx.emitFmt("    {s} = call double @llvm.fabs.f64(double {s})\n", .{ result, arg }),
                .sin => try ctx.emitFmt("    {s} = call double @llvm.sin.f64(double {s})\n", .{ result, arg }),
                .cos => try ctx.emitFmt("    {s} = call double @llvm.cos.f64(double {s})\n", .{ result, arg }),
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
                else => try ctx.emitFmt("    {s} = fadd double 0.0, {s}\n", .{ result, arg }),
            }
            return result;
        },
        .unary => |un| {
            const operand = try genBackwardExpr(ctx, un.operand, loop_vars);
            const result = try ctx.newTemp();
            switch (un.op) {
                .negate => try ctx.emitFmt("    {s} = fneg double {s}\n", .{ result, operand }),
                .not => {
                    const cmp = try ctx.newTemp();
                    try ctx.emitFmt("    {s} = fcmp oeq double {s}, 0.0\n", .{ cmp, operand });
                    try ctx.emitFmt("    {s} = select i1 {s}, double 1.0, double 0.0\n", .{ result, cmp });
                },
            }
            return result;
        },
        .group => |g| return try genBackwardExpr(ctx, g, loop_vars),
        else => return "0.0",
    }
}

/// Generate tensor access for backward mode
/// For tensors with backward functions, generates a recursive call
/// For other tensors, loads from memory
fn genBackwardTensorAccess(
    ctx: *CodegenContext,
    ref: *const ast.TensorRef,
    loop_vars: *std.StringHashMap([]const u8),
) ![]const u8 {
    const info = ctx.tensors.get(ref.name) orelse return "0.0";

    // Check if this tensor has a backward compute function
    // For now, check if there's a @compute_Name function declared
    // In practice, we'd track this in a set

    // Collect index values
    var idx_vals = std.ArrayListUnmanaged([]const u8){};
    defer idx_vals.deinit(ctx.allocator);

    for (ref.indices) |idx| {
        const val = switch (idx) {
            .name => |n| blk: {
                if (loop_vars.get(n)) |var_name| {
                    // Load if it's a pointer (loop variable)
                    if (std.mem.startsWith(u8, var_name, "%bw_") or std.mem.startsWith(u8, var_name, "%r")) {
                        const loaded = try ctx.newTemp();
                        try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ loaded, var_name });
                        break :blk loaded;
                    }
                    break :blk var_name;
                }
                break :blk "0";
            },
            .constant => |c| try std.fmt.allocPrint(ctx.string_arena.allocator(), "{d}", .{c}),
            else => "0",
        };
        try idx_vals.append(ctx.allocator, val);
    }

    // Check if this is a recursive call (tensor has compute function)
    // For simplicity, check if tensor name matches a backward-computed tensor
    // This is a heuristic - in full impl we'd track this properly
    if (ctx.backward_tensors.contains(ref.name)) {
        // Generate recursive call
        var call = std.ArrayListUnmanaged(u8){};
        defer call.deinit(ctx.allocator);

        const result = try ctx.newTemp();
        try call.writer(ctx.allocator).print("    {s} = call double @compute_{s}(", .{ result, ref.name });

        for (idx_vals.items, 0..) |val, i| {
            if (i > 0) try call.appendSlice(ctx.allocator, ", ");
            try call.writer(ctx.allocator).print("i64 {s}", .{val});
        }
        try call.appendSlice(ctx.allocator, ")\n");
        try ctx.emit(call.items);
        return result;
    }

    // Standard tensor load
    const offset = try tensor.computeLinearOffset(ctx, info.strides, idx_vals.items);
    const ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ ptr, info.llvm_ptr, offset });
    const result = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ result, ptr });
    return result;
}

/// Generate linear offset from index values (for function params)
fn genLinearOffset(ctx: *CodegenContext, strides: []const usize, rank: usize) ![]const u8 {
    if (rank == 0) return "0";

    var result: []const u8 = "0";

    for (0..rank) |i| {
        const idx_param = try std.fmt.allocPrint(ctx.string_arena.allocator(), "%idx{d}", .{i});
        const term = try ctx.newTemp();
        try ctx.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ term, idx_param, strides[i] });
        const new_result = try ctx.newTemp();
        try ctx.emitFmt("    {s} = add i64 {s}, {s}\n", .{ new_result, result, term });
        result = new_result;
    }

    return result;
}

/// Generate a backward query (calls the compute function)
pub fn genBackwardQuery(ctx: *CodegenContext, query: *const ast.Query) !void {
    try ctx.emitFmt("\n    ; Backward query: {s}\n", .{query.tensor.name});

    // Collect constant indices
    var idx_vals = std.ArrayListUnmanaged([]const u8){};
    defer idx_vals.deinit(ctx.allocator);

    for (query.tensor.indices) |idx| {
        const val = switch (idx) {
            .constant => |c| try std.fmt.allocPrint(ctx.string_arena.allocator(), "{d}", .{c}),
            else => "0",
        };
        try idx_vals.append(ctx.allocator, val);
    }

    // Generate call to compute function
    var call = std.ArrayListUnmanaged(u8){};
    defer call.deinit(ctx.allocator);

    const result = try ctx.newTemp();
    try call.writer(ctx.allocator).print("    {s} = call double @compute_{s}(", .{ result, query.tensor.name });

    for (idx_vals.items, 0..) |val, i| {
        if (i > 0) try call.appendSlice(ctx.allocator, ", ");
        try call.writer(ctx.allocator).print("i64 {s}", .{val});
    }
    try call.appendSlice(ctx.allocator, ")\n");
    try ctx.emit(call.items);

    // Print result
    try ctx.emitFmt("    call i32 (ptr, ...) @printf(ptr @.str.backward_result, ptr @.name.{s}, double {s})\n", .{ query.tensor.name, result });
}
