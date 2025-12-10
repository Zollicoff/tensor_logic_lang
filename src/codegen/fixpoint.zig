// Fixpoint iteration for LLVM code generation
//
// Handles recursive equations that require iteration until convergence.
// Detects recursive equations (LHS tensor appears in RHS) and generates
// a loop that runs until no values change.

const std = @import("std");
const ast = @import("../frontend/ast.zig");
const types = @import("types.zig");
const tensor = @import("tensor.zig");
const expr = @import("expr.zig");

const TensorInfo = types.TensorInfo;
const IndexVar = types.IndexVar;
const LoopLabels = types.LoopLabels;

/// Codegen context (forward declaration)
pub const CodegenContext = @import("llvm.zig").LLVMCodegen;

/// Check if an equation is recursive (LHS tensor appears in RHS)
pub fn isRecursive(eq: *const ast.Equation) bool {
    return exprContainsTensor(eq.rhs, eq.lhs.name);
}

/// Check if expression contains a reference to the given tensor
pub fn exprContainsTensor(expression: *const ast.Expr, name: []const u8) bool {
    switch (expression.*) {
        .tensor_ref => |ref| return std.mem.eql(u8, ref.name, name),
        .product => |prod| {
            for (prod.factors) |factor| {
                if (exprContainsTensor(factor, name)) return true;
            }
            return false;
        },
        .binary => |bin| {
            return exprContainsTensor(bin.left, name) or exprContainsTensor(bin.right, name);
        },
        .nonlinearity => |nl| return exprContainsTensor(nl.arg, name),
        .unary => |un| return exprContainsTensor(un.operand, name),
        .group => |g| return exprContainsTensor(g, name),
        else => return false,
    }
}

/// Generate fixpoint loop for recursive equations
pub fn genFixpointLoop(ctx: *CodegenContext, program: *const ast.Program, recursive_indices: []const usize) !void {
    try ctx.emit("\n    ; Fixpoint iteration for recursive equations\n");

    // Allocate a "changed" flag
    const changed_var = try ctx.newTemp();
    try ctx.emitFmt("    {s} = alloca i1\n", .{changed_var});

    // Allocate iteration counter for safety limit
    const iter_var = try ctx.newTemp();
    try ctx.emitFmt("    {s} = alloca i64\n", .{iter_var});
    try ctx.emitFmt("    store i64 0, ptr {s}\n", .{iter_var});

    // Loop labels
    const loop_start = try ctx.newLabel();
    const loop_body = try ctx.newLabel();
    const loop_end = try ctx.newLabel();

    // Start loop
    try ctx.emitFmt("    br label %{s}\n", .{loop_start});
    try ctx.emitFmt("{s}:\n", .{loop_start});

    // Check iteration limit (max 1000 iterations)
    const iter_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ iter_val, iter_var });
    const iter_cmp = try ctx.newTemp();
    try ctx.emitFmt("    {s} = icmp slt i64 {s}, 1000\n", .{ iter_cmp, iter_val });
    try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ iter_cmp, loop_body, loop_end });

    try ctx.emitFmt("{s}:\n", .{loop_body});

    // Reset changed flag
    try ctx.emitFmt("    store i1 false, ptr {s}\n", .{changed_var});

    // Generate code for each recursive equation, tracking changes
    for (recursive_indices) |idx| {
        const stmt = program.statements[idx];
        if (stmt == .equation) {
            try genRecursiveEquation(ctx, &stmt.equation, changed_var);
        }
    }

    // Increment iteration counter
    const next_iter = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ next_iter, iter_val });
    try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ next_iter, iter_var });

    // Check if changed - if so, continue; otherwise exit
    const did_change = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load i1, ptr {s}\n", .{ did_change, changed_var });
    try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ did_change, loop_start, loop_end });

    try ctx.emitFmt("{s}:\n", .{loop_end});
}

/// Generate a recursive equation with change tracking
fn genRecursiveEquation(ctx: *CodegenContext, eq: *const ast.Equation, changed_var: []const u8) !void {
    try ctx.emitFmt("\n    ; Recursive equation: {s}[...] += ...\n", .{eq.lhs.name});

    // Ensure LHS tensor exists
    if (!ctx.tensors.contains(eq.lhs.name)) {
        try tensor.allocateTensor(ctx, eq.lhs.name, eq.lhs.indices);
    }

    // Analyze indices
    var all_indices = std.StringHashMapUnmanaged(usize){};
    defer all_indices.deinit(ctx.allocator);

    var lhs_indices = std.StringHashMapUnmanaged(void){};
    defer lhs_indices.deinit(ctx.allocator);

    for (eq.lhs.indices) |idx| {
        const name = switch (idx) {
            .name => |n| n,
            .normalize => |n| n,
            else => continue,
        };
        try lhs_indices.put(ctx.allocator, name, {});
        const size = ctx.domains.get(name) orelse 10;
        try all_indices.put(ctx.allocator, name, size);
    }

    try expr.collectExprIndices(ctx, eq.rhs, &all_indices);

    // Build index variable list
    var index_vars = std.ArrayListUnmanaged(IndexVar){};
    defer index_vars.deinit(ctx.allocator);

    // Free indices (on LHS)
    for (eq.lhs.indices) |idx| {
        const name = switch (idx) {
            .name => |n| n,
            .normalize => |n| n,
            else => continue,
        };
        const size = all_indices.get(name) orelse 10;
        try index_vars.append(ctx.allocator, .{
            .name = name,
            .size = size,
            .llvm_var = "",
            .is_contracted = false,
        });
    }

    // Contracted indices
    var iter = all_indices.iterator();
    while (iter.next()) |entry| {
        if (!lhs_indices.contains(entry.key_ptr.*)) {
            try index_vars.append(ctx.allocator, .{
                .name = entry.key_ptr.*,
                .size = entry.value_ptr.*,
                .llvm_var = "",
                .is_contracted = true,
            });
        }
    }

    // Generate the loops with change tracking
    try genRecursiveEinsumLoops(ctx, eq, index_vars.items, changed_var);
}

/// Generate einsum loops for recursive equation with change tracking
fn genRecursiveEinsumLoops(ctx: *CodegenContext, eq: *const ast.Equation, index_vars: []IndexVar, changed_var: []const u8) !void {
    const lhs_info = ctx.tensors.get(eq.lhs.name).?;
    const eq_id = ctx.equation_counter;
    ctx.equation_counter += 1;

    // Allocate loop variables (with unique prefix per equation)
    var loop_vars = std.StringHashMapUnmanaged([]const u8){};
    defer loop_vars.deinit(ctx.allocator);

    for (index_vars) |*iv| {
        const var_name = try std.fmt.allocPrint(ctx.string_arena.allocator(), "%r{d}_{s}", .{ eq_id, iv.name });
        try ctx.emitFmt("    {s} = alloca i64\n", .{var_name});
        try loop_vars.put(ctx.allocator, iv.name, var_name);
    }

    // Generate loop headers
    var loop_labels = std.ArrayListUnmanaged(LoopLabels){};
    defer loop_labels.deinit(ctx.allocator);

    for (index_vars) |iv| {
        const var_name = loop_vars.get(iv.name).?;
        const start_label = try ctx.newLabel();
        const body_label = try ctx.newLabel();
        const end_label = try ctx.newLabel();

        try loop_labels.append(ctx.allocator, .{ .start = start_label, .body = body_label, .end = end_label });

        try ctx.emitFmt("    store i64 0, ptr {s}\n", .{var_name});
        try ctx.emitFmt("    br label %{s}\n", .{start_label});
        try ctx.emitFmt("{s}:\n", .{start_label});

        const idx_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ idx_val, var_name });
        const cmp = try ctx.newTemp();
        try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ cmp, idx_val, iv.size });
        try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ cmp, body_label, end_label });
        try ctx.emitFmt("{s}:\n", .{body_label});
    }

    // Compute LHS address
    var lhs_idx_vals = std.ArrayListUnmanaged([]const u8){};
    defer lhs_idx_vals.deinit(ctx.allocator);

    for (eq.lhs.indices) |idx| {
        const name = switch (idx) {
            .name => |n| n,
            .normalize => |n| n,
            else => continue,
        };
        if (loop_vars.get(name)) |var_name| {
            const val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ val, var_name });
            try lhs_idx_vals.append(ctx.allocator, val);
        }
    }

    const lhs_offset = try tensor.computeLinearOffset(ctx, lhs_info.strides, lhs_idx_vals.items);
    const lhs_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ lhs_ptr, lhs_info.llvm_ptr, lhs_offset });

    // Load old value
    const old_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, lhs_ptr });

    // Evaluate RHS
    const rhs_val = try expr.genExpr(ctx, eq.rhs, &loop_vars);

    // Apply accumulation operator with change detection
    switch (eq.op) {
        .add => {
            // For +=: new = old + rhs, changed if rhs > 0
            const new_val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, rhs_val });

            // Check if value increased
            const did_increase = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fcmp ogt double {s}, {s}\n", .{ did_increase, new_val, old_val });

            // Update changed flag if increased
            const update_label = try ctx.newLabel();
            const skip_label = try ctx.newLabel();
            try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ did_increase, update_label, skip_label });
            try ctx.emitFmt("{s}:\n", .{update_label});
            try ctx.emitFmt("    store i1 true, ptr {s}\n", .{changed_var});
            try ctx.emitFmt("    br label %{s}\n", .{skip_label});
            try ctx.emitFmt("{s}:\n", .{skip_label});

            try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, lhs_ptr });
        },
        .max => {
            // For max=: new = max(old, rhs)
            const cmp = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fcmp ogt double {s}, {s}\n", .{ cmp, rhs_val, old_val });
            const new_val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = select i1 {s}, double {s}, double {s}\n", .{ new_val, cmp, rhs_val, old_val });

            // Update changed if new > old
            const update_label = try ctx.newLabel();
            const skip_label = try ctx.newLabel();
            try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ cmp, update_label, skip_label });
            try ctx.emitFmt("{s}:\n", .{update_label});
            try ctx.emitFmt("    store i1 true, ptr {s}\n", .{changed_var});
            try ctx.emitFmt("    br label %{s}\n", .{skip_label});
            try ctx.emitFmt("{s}:\n", .{skip_label});

            try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, lhs_ptr });
        },
        else => {
            // For other ops, just apply and assume change
            try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ rhs_val, lhs_ptr });
            try ctx.emitFmt("    store i1 true, ptr {s}\n", .{changed_var});
        },
    }

    // Generate loop footers
    var i: usize = index_vars.len;
    while (i > 0) {
        i -= 1;
        const iv = index_vars[i];
        const var_name = loop_vars.get(iv.name).?;
        const labels = loop_labels.items[i];

        const idx_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ idx_val, var_name });
        const next_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ next_val, idx_val });
        try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ next_val, var_name });
        try ctx.emitFmt("    br label %{s}\n", .{labels.start});
        try ctx.emitFmt("{s}:\n", .{labels.end});
    }
}
