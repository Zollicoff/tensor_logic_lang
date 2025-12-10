// Einstein summation loop generation for LLVM code generation
//
// Generates nested loops for tensor contractions following Einstein notation.
// Handles index analysis, loop structure, and accumulation operators.

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

/// Generate nested loops for einsum contraction
pub fn genEinsumLoops(ctx: *CodegenContext, eq: *const ast.Equation, index_vars: []IndexVar) !void {
    const lhs_info = ctx.tensors.get(eq.lhs.name).?;
    const eq_id = ctx.equation_counter;
    ctx.equation_counter += 1;

    // Allocate loop variables (with unique prefix per equation)
    var loop_vars = std.StringHashMapUnmanaged([]const u8){};
    defer loop_vars.deinit(ctx.allocator);

    for (index_vars) |*iv| {
        // Sanitize name for LLVM (replace ' with _prime)
        const sanitized_name = try sanitizeLLVMName(ctx.string_arena.allocator(), iv.name);
        const var_name = try std.fmt.allocPrint(ctx.string_arena.allocator(), "%e{d}_{s}", .{ eq_id, sanitized_name });
        try ctx.emitFmt("    {s} = alloca i64\n", .{var_name});
        try loop_vars.put(ctx.allocator, iv.name, var_name);
    }

    // Generate loop headers (outermost to innermost)
    var loop_labels = std.ArrayListUnmanaged(LoopLabels){};
    defer loop_labels.deinit(ctx.allocator);

    for (index_vars) |iv| {
        const var_name = loop_vars.get(iv.name).?;
        const start_label = try ctx.newLabel();
        const body_label = try ctx.newLabel();
        const end_label = try ctx.newLabel();

        try loop_labels.append(ctx.allocator, .{ .start = start_label, .body = body_label, .end = end_label });

        // Initialize loop variable
        try ctx.emitFmt("    store i64 0, ptr {s}\n", .{var_name});
        try ctx.emitFmt("    br label %{s}\n", .{start_label});

        // Loop header
        try ctx.emitFmt("{s}:\n", .{start_label});
        const idx_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ idx_val, var_name });
        const cmp = try ctx.newTemp();
        try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ cmp, idx_val, iv.size });
        try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ cmp, body_label, end_label });

        // Loop body start
        try ctx.emitFmt("{s}:\n", .{body_label});
    }

    // Check if we have contracted indices (need accumulation)
    var has_contraction = false;
    for (index_vars) |iv| {
        if (iv.is_contracted) {
            has_contraction = true;
            break;
        }
    }

    // Compute LHS address
    var lhs_idx_vals = std.ArrayListUnmanaged([]const u8){};
    defer lhs_idx_vals.deinit(ctx.allocator);

    for (eq.lhs.indices) |idx| {
        const idx_name: ?[]const u8 = switch (idx) {
            .name => |n| n,
            .primed => |n| blk: {
                const primed_name = std.fmt.allocPrint(ctx.string_arena.allocator(), "{s}'", .{n}) catch null;
                break :blk primed_name;
            },
            else => null,
        };
        if (idx_name) |name| {
            const var_name = loop_vars.get(name) orelse continue;
            const val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ val, var_name });
            try lhs_idx_vals.append(ctx.allocator, val);
        }
    }

    const lhs_offset = try tensor.computeLinearOffset(ctx, lhs_info.strides, lhs_idx_vals.items);
    const lhs_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ lhs_ptr, lhs_info.llvm_ptr, lhs_offset });

    // Evaluate RHS
    const rhs_val = try expr.genExpr(ctx, eq.rhs, &loop_vars);

    // Store result (with accumulation for contracted indices)
    try genAccumulation(ctx, eq.op, lhs_ptr, rhs_val, has_contraction);

    // Generate loop footers (innermost to outermost)
    var i: usize = index_vars.len;
    while (i > 0) {
        i -= 1;
        const iv = index_vars[i];
        const var_name = loop_vars.get(iv.name).?;
        const labels = loop_labels.items[i];

        // Increment and branch back
        const idx_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ idx_val, var_name });
        const next_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ next_val, idx_val });
        try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ next_val, var_name });
        try ctx.emitFmt("    br label %{s}\n", .{labels.start});

        // Loop end
        try ctx.emitFmt("{s}:\n", .{labels.end});
    }
}

/// Generate accumulation operation based on operator
fn genAccumulation(ctx: *CodegenContext, op: ast.AccumulationOp, lhs_ptr: []const u8, rhs_val: []const u8, has_contraction: bool) !void {
    switch (op) {
        .assign => {
            if (has_contraction) {
                // For contraction, we need to accumulate: load, add, store
                const old_val = try ctx.newTemp();
                try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, lhs_ptr });
                const new_val = try ctx.newTemp();
                try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, rhs_val });
                try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, lhs_ptr });
            } else {
                try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ rhs_val, lhs_ptr });
            }
        },
        .add => {
            const old_val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, lhs_ptr });
            const new_val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_val, old_val, rhs_val });
            try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, lhs_ptr });
        },
        .max => {
            const old_val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, lhs_ptr });
            const cmp = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fcmp ogt double {s}, {s}\n", .{ cmp, rhs_val, old_val });
            const new_val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = select i1 {s}, double {s}, double {s}\n", .{ new_val, cmp, rhs_val, old_val });
            try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, lhs_ptr });
        },
        .min => {
            const old_val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, lhs_ptr });
            const cmp = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fcmp olt double {s}, {s}\n", .{ cmp, rhs_val, old_val });
            const new_val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = select i1 {s}, double {s}, double {s}\n", .{ new_val, cmp, rhs_val, old_val });
            try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, lhs_ptr });
        },
        .mul => {
            const old_val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_val, lhs_ptr });
            const new_val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ new_val, old_val, rhs_val });
            try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_val, lhs_ptr });
        },
        else => {
            try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ rhs_val, lhs_ptr });
        },
    }
}

/// Generate scalar assignment (constant indices only, no loop variables)
pub fn genScalarAssign(ctx: *CodegenContext, eq: *const ast.Equation) !void {
    var empty_loop_vars = std.StringHashMapUnmanaged([]const u8){};
    defer empty_loop_vars.deinit(ctx.allocator);

    const val = try expr.genExpr(ctx, eq.rhs, &empty_loop_vars);
    const info = ctx.tensors.get(eq.lhs.name).?;

    // Compute linear offset from constant indices
    var offset: usize = 0;
    for (eq.lhs.indices, 0..) |idx, i| {
        if (i >= info.strides.len) break;
        const idx_val: usize = switch (idx) {
            .constant => |c| @intCast(c),
            else => 0,
        };
        offset += idx_val * info.strides[i];
    }

    const ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ ptr, info.llvm_ptr, offset });
    try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ val, ptr });
}

/// Sanitize index name for LLVM IR (replace ' with _prime, etc.)
fn sanitizeLLVMName(allocator: std.mem.Allocator, name: []const u8) ![]const u8 {
    // Check if sanitization needed
    var needs_sanitize = false;
    for (name) |c| {
        if (c == '\'') {
            needs_sanitize = true;
            break;
        }
    }
    if (!needs_sanitize) return name;

    // Replace ' with _prime
    var result = std.ArrayListUnmanaged(u8){};
    for (name) |c| {
        if (c == '\'') {
            try result.appendSlice(allocator, "_prime");
        } else {
            try result.append(allocator, c);
        }
    }
    return result.items;
}
