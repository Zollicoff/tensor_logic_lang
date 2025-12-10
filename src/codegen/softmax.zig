// Softmax operation for LLVM code generation
//
// Implements softmax with reduction: Y[i, j.] = softmax(X[i, j])
// The normalize index (j.) indicates the axis over which to compute softmax.
// Uses a two-pass algorithm: first sum exp values, then divide.

const std = @import("std");
const ast = @import("../frontend/ast.zig");
const types = @import("types.zig");
const tensor = @import("tensor.zig");

const TensorInfo = types.TensorInfo;
const LoopLabels = types.LoopLabels;

/// Codegen context (forward declaration)
pub const CodegenContext = @import("llvm.zig").LLVMCodegen;

/// Generate softmax operation
pub fn genSoftmax(ctx: *CodegenContext, eq: *const ast.Equation) !void {
    try ctx.emit("    ; Softmax operation\n");

    // Get the input tensor from the softmax argument
    const nl = eq.rhs.nonlinearity;
    const input_ref = if (nl.arg.* == .tensor_ref) nl.arg.tensor_ref else {
        try ctx.emit("    ; Error: softmax argument must be tensor reference\n");
        return;
    };

    // Find the normalize axis in LHS indices
    var norm_axis: ?usize = null;
    var norm_name: ?[]const u8 = null;
    for (eq.lhs.indices, 0..) |idx, i| {
        if (idx == .normalize) {
            norm_axis = i;
            norm_name = idx.normalize;
            break;
        }
    }

    if (norm_axis == null) {
        // Default to last axis if no normalize marker
        if (eq.lhs.indices.len > 0) {
            norm_axis = eq.lhs.indices.len - 1;
            const last_idx = eq.lhs.indices[norm_axis.?];
            norm_name = switch (last_idx) {
                .name => |n| n,
                .normalize => |n| n,
                else => null,
            };
        }
    }

    // Ensure output tensor exists
    if (!ctx.tensors.contains(eq.lhs.name)) {
        try tensor.allocateTensor(ctx, eq.lhs.name, eq.lhs.indices);
    }

    // Ensure input tensor exists (should already be allocated)
    const input_info = ctx.tensors.get(input_ref.name) orelse {
        try ctx.emit("    ; Error: input tensor not found\n");
        return;
    };
    const output_info = ctx.tensors.get(eq.lhs.name).?;

    // Collect dimension sizes for all indices
    var outer_dims = std.ArrayListUnmanaged(struct { name: []const u8, size: usize }){};
    defer outer_dims.deinit(ctx.allocator);
    var norm_size: usize = 1;

    for (eq.lhs.indices, 0..) |idx, i| {
        const idx_name = switch (idx) {
            .name => |n| n,
            .normalize => |n| n,
            else => continue,
        };
        const size = ctx.domains.get(idx_name) orelse 10;

        if (i == norm_axis) {
            norm_size = size;
        } else {
            try outer_dims.append(ctx.allocator, .{ .name = idx_name, .size = size });
        }
    }

    // Allocate loop variables for outer dimensions
    var outer_vars = std.StringHashMapUnmanaged([]const u8){};
    defer outer_vars.deinit(ctx.allocator);

    for (outer_dims.items) |dim| {
        const var_name = try std.fmt.allocPrint(ctx.string_arena.allocator(), "%sm_{s}", .{dim.name});
        try ctx.emitFmt("    {s} = alloca i64\n", .{var_name});
        try outer_vars.put(ctx.allocator, dim.name, var_name);
    }

    // Allocate sum variable and normalize index variable
    const sum_var = try ctx.newTemp();
    try ctx.emitFmt("    {s} = alloca double\n", .{sum_var});
    const norm_var = try std.fmt.allocPrint(ctx.string_arena.allocator(), "%sm_{s}", .{norm_name orelse "n"});
    try ctx.emitFmt("    {s} = alloca i64\n", .{norm_var});

    // Generate nested loops for outer dimensions
    var outer_labels = std.ArrayListUnmanaged(LoopLabels){};
    defer outer_labels.deinit(ctx.allocator);

    for (outer_dims.items) |dim| {
        const var_name = outer_vars.get(dim.name).?;
        const start_label = try ctx.newLabel();
        const body_label = try ctx.newLabel();
        const end_label = try ctx.newLabel();

        try outer_labels.append(ctx.allocator, .{ .start = start_label, .body = body_label, .end = end_label });

        try ctx.emitFmt("    store i64 0, ptr {s}\n", .{var_name});
        try ctx.emitFmt("    br label %{s}\n", .{start_label});
        try ctx.emitFmt("{s}:\n", .{start_label});

        const idx_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ idx_val, var_name });
        const cmp = try ctx.newTemp();
        try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ cmp, idx_val, dim.size });
        try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ cmp, body_label, end_label });
        try ctx.emitFmt("{s}:\n", .{body_label});
    }

    // Initialize sum to 0
    try ctx.emitFmt("    store double 0.0, ptr {s}\n", .{sum_var});

    // Pass 1: Sum exp(x) over normalize axis
    const sum_start = try ctx.newLabel();
    const sum_body = try ctx.newLabel();
    const sum_end = try ctx.newLabel();

    try ctx.emitFmt("    store i64 0, ptr {s}\n", .{norm_var});
    try ctx.emitFmt("    br label %{s}\n", .{sum_start});
    try ctx.emitFmt("{s}:\n", .{sum_start});

    const norm_idx = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ norm_idx, norm_var });
    const norm_cmp = try ctx.newTemp();
    try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ norm_cmp, norm_idx, norm_size });
    try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ norm_cmp, sum_body, sum_end });
    try ctx.emitFmt("{s}:\n", .{sum_body});

    // Compute input offset and load value
    const input_offset = try computeSoftmaxOffset(ctx, input_info, eq.lhs.indices, norm_axis.?, &outer_vars, norm_var);
    const input_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ input_ptr, input_info.llvm_ptr, input_offset });
    const input_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ input_val, input_ptr });

    // exp(x)
    const exp_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = call double @llvm.exp.f64(double {s})\n", .{ exp_val, input_val });

    // Accumulate sum
    const old_sum = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_sum, sum_var });
    const new_sum = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_sum, old_sum, exp_val });
    try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_sum, sum_var });

    // Increment and loop
    const next_norm = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ next_norm, norm_idx });
    try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ next_norm, norm_var });
    try ctx.emitFmt("    br label %{s}\n", .{sum_start});
    try ctx.emitFmt("{s}:\n", .{sum_end});

    // Pass 2: Compute softmax = exp(x) / sum
    const div_start = try ctx.newLabel();
    const div_body = try ctx.newLabel();
    const div_end = try ctx.newLabel();

    try ctx.emitFmt("    store i64 0, ptr {s}\n", .{norm_var});
    try ctx.emitFmt("    br label %{s}\n", .{div_start});
    try ctx.emitFmt("{s}:\n", .{div_start});

    const norm_idx2 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ norm_idx2, norm_var });
    const norm_cmp2 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ norm_cmp2, norm_idx2, norm_size });
    try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ norm_cmp2, div_body, div_end });
    try ctx.emitFmt("{s}:\n", .{div_body});

    // Load input, compute exp
    const input_offset2 = try computeSoftmaxOffset(ctx, input_info, eq.lhs.indices, norm_axis.?, &outer_vars, norm_var);
    const input_ptr2 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ input_ptr2, input_info.llvm_ptr, input_offset2 });
    const input_val2 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ input_val2, input_ptr2 });
    const exp_val2 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = call double @llvm.exp.f64(double {s})\n", .{ exp_val2, input_val2 });

    // Divide by sum
    const sum_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ sum_val, sum_var });
    const softmax_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fdiv double {s}, {s}\n", .{ softmax_val, exp_val2, sum_val });

    // Store to output
    const output_offset = try computeSoftmaxOffset(ctx, output_info, eq.lhs.indices, norm_axis.?, &outer_vars, norm_var);
    const output_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ output_ptr, output_info.llvm_ptr, output_offset });
    try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ softmax_val, output_ptr });

    // Increment and loop
    const next_norm2 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ next_norm2, norm_idx2 });
    try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ next_norm2, norm_var });
    try ctx.emitFmt("    br label %{s}\n", .{div_start});
    try ctx.emitFmt("{s}:\n", .{div_end});

    // Close outer loops (innermost to outermost)
    var i: usize = outer_dims.items.len;
    while (i > 0) {
        i -= 1;
        const dim = outer_dims.items[i];
        const var_name = outer_vars.get(dim.name).?;
        const labels = outer_labels.items[i];

        const idx_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ idx_val, var_name });
        const next_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ next_val, idx_val });
        try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ next_val, var_name });
        try ctx.emitFmt("    br label %{s}\n", .{labels.start});
        try ctx.emitFmt("{s}:\n", .{labels.end});
    }
}

/// Compute linear offset for softmax tensor access
fn computeSoftmaxOffset(
    ctx: *CodegenContext,
    info: TensorInfo,
    indices: []const ast.Index,
    norm_axis: usize,
    outer_vars: *std.StringHashMapUnmanaged([]const u8),
    norm_var: []const u8,
) ![]const u8 {
    if (info.rank == 0) return "0";

    var result = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 0, 0\n", .{result});

    for (indices, 0..) |idx, i| {
        if (i >= info.strides.len) break;

        const idx_name = switch (idx) {
            .name => |n| n,
            .normalize => |n| n,
            else => continue,
        };

        // Get the loop variable for this index
        const var_ptr = if (i == norm_axis) norm_var else (outer_vars.get(idx_name) orelse continue);
        const idx_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ idx_val, var_ptr });

        const mul = try ctx.newTemp();
        try ctx.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ mul, idx_val, info.strides[i] });
        const new_result = try ctx.newTemp();
        try ctx.emitFmt("    {s} = add i64 {s}, {s}\n", .{ new_result, result, mul });
        result = new_result;
    }

    return result;
}
