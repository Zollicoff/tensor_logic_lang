// Layer normalization for LLVM code generation
//
// Implements layer normalization: Y[i, j.] = lnorm(X[i, j])
// The normalize index (j.) indicates the axis over which to normalize.
// Uses a three-pass algorithm: compute mean, compute variance, normalize.
//
// Formula: (x - mean) / sqrt(variance + epsilon)
// Where:
//   mean = sum(x) / n
//   variance = sum((x - mean)^2) / n
//   epsilon = 1e-5 (for numerical stability)

const std = @import("std");
const ast = @import("../frontend/ast.zig");
const types = @import("types.zig");
const tensor = @import("tensor.zig");

const TensorInfo = types.TensorInfo;
const LoopLabels = types.LoopLabels;

/// Codegen context (forward declaration)
pub const CodegenContext = @import("llvm.zig").LLVMCodegen;

/// Generate layer normalization operation
pub fn genLayerNorm(ctx: *CodegenContext, eq: *const ast.Equation) !void {
    try ctx.emit("    ; Layer normalization operation\n");

    // Get the input tensor from the lnorm argument
    const nl = eq.rhs.nonlinearity;
    const input_ref = if (nl.arg.* == .tensor_ref) nl.arg.tensor_ref else {
        try ctx.emit("    ; Error: lnorm argument must be tensor reference\n");
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
        const var_name = try std.fmt.allocPrint(ctx.string_arena.allocator(), "%ln_{s}", .{dim.name});
        try ctx.emitFmt("    {s} = alloca i64\n", .{var_name});
        try outer_vars.put(ctx.allocator, dim.name, var_name);
    }

    // Allocate mean, variance, and normalize index variables
    const mean_var = try ctx.newTemp();
    try ctx.emitFmt("    {s} = alloca double\n", .{mean_var});
    const var_var = try ctx.newTemp(); // variance
    try ctx.emitFmt("    {s} = alloca double\n", .{var_var});
    const norm_var = try std.fmt.allocPrint(ctx.string_arena.allocator(), "%ln_{s}", .{norm_name orelse "n"});
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

    // =========================================================================
    // Pass 1: Compute mean = sum(x) / n
    // =========================================================================
    try ctx.emit("    ; Pass 1: compute mean\n");
    try ctx.emitFmt("    store double 0.0, ptr {s}\n", .{mean_var});

    const mean_start = try ctx.newLabel();
    const mean_body = try ctx.newLabel();
    const mean_end = try ctx.newLabel();

    try ctx.emitFmt("    store i64 0, ptr {s}\n", .{norm_var});
    try ctx.emitFmt("    br label %{s}\n", .{mean_start});
    try ctx.emitFmt("{s}:\n", .{mean_start});

    const mean_idx = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ mean_idx, norm_var });
    const mean_cmp = try ctx.newTemp();
    try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ mean_cmp, mean_idx, norm_size });
    try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ mean_cmp, mean_body, mean_end });
    try ctx.emitFmt("{s}:\n", .{mean_body});

    // Load input value
    const input_offset1 = try computeLayerNormOffset(ctx, input_info, eq.lhs.indices, norm_axis.?, &outer_vars, norm_var);
    const input_ptr1 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ input_ptr1, input_info.llvm_ptr, input_offset1 });
    const input_val1 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ input_val1, input_ptr1 });

    // Accumulate sum
    const old_sum = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_sum, mean_var });
    const new_sum = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_sum, old_sum, input_val1 });
    try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_sum, mean_var });

    // Increment and loop
    const next_mean = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ next_mean, mean_idx });
    try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ next_mean, norm_var });
    try ctx.emitFmt("    br label %{s}\n", .{mean_start});
    try ctx.emitFmt("{s}:\n", .{mean_end});

    // Divide sum by n to get mean
    const sum_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ sum_val, mean_var });
    const mean_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fdiv double {s}, {d}.0\n", .{ mean_val, sum_val, norm_size });
    try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ mean_val, mean_var });

    // =========================================================================
    // Pass 2: Compute variance = sum((x - mean)^2) / n
    // =========================================================================
    try ctx.emit("    ; Pass 2: compute variance\n");
    try ctx.emitFmt("    store double 0.0, ptr {s}\n", .{var_var});

    const var_start = try ctx.newLabel();
    const var_body = try ctx.newLabel();
    const var_end = try ctx.newLabel();

    try ctx.emitFmt("    store i64 0, ptr {s}\n", .{norm_var});
    try ctx.emitFmt("    br label %{s}\n", .{var_start});
    try ctx.emitFmt("{s}:\n", .{var_start});

    const var_idx = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ var_idx, norm_var });
    const var_cmp = try ctx.newTemp();
    try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ var_cmp, var_idx, norm_size });
    try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ var_cmp, var_body, var_end });
    try ctx.emitFmt("{s}:\n", .{var_body});

    // Load input value
    const input_offset2 = try computeLayerNormOffset(ctx, input_info, eq.lhs.indices, norm_axis.?, &outer_vars, norm_var);
    const input_ptr2 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ input_ptr2, input_info.llvm_ptr, input_offset2 });
    const input_val2 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ input_val2, input_ptr2 });

    // Compute (x - mean)^2
    const mean_loaded = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ mean_loaded, mean_var });
    const diff = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fsub double {s}, {s}\n", .{ diff, input_val2, mean_loaded });
    const diff_sq = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ diff_sq, diff, diff });

    // Accumulate sum of squared differences
    const old_var = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_var, var_var });
    const new_var = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_var, old_var, diff_sq });
    try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_var, var_var });

    // Increment and loop
    const next_var = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ next_var, var_idx });
    try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ next_var, norm_var });
    try ctx.emitFmt("    br label %{s}\n", .{var_start});
    try ctx.emitFmt("{s}:\n", .{var_end});

    // Divide by n to get variance
    const var_sum = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ var_sum, var_var });
    const variance = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fdiv double {s}, {d}.0\n", .{ variance, var_sum, norm_size });
    try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ variance, var_var });

    // =========================================================================
    // Pass 3: Normalize = (x - mean) / sqrt(variance + epsilon)
    // =========================================================================
    try ctx.emit("    ; Pass 3: normalize\n");

    // Compute sqrt(variance + epsilon)
    const var_loaded = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ var_loaded, var_var });
    const var_eps = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fadd double {s}, 1.0e-5\n", .{ var_eps, var_loaded });
    const std_dev = try ctx.newTemp();
    try ctx.emitFmt("    {s} = call double @llvm.sqrt.f64(double {s})\n", .{ std_dev, var_eps });

    const norm_start = try ctx.newLabel();
    const norm_body = try ctx.newLabel();
    const norm_end = try ctx.newLabel();

    try ctx.emitFmt("    store i64 0, ptr {s}\n", .{norm_var});
    try ctx.emitFmt("    br label %{s}\n", .{norm_start});
    try ctx.emitFmt("{s}:\n", .{norm_start});

    const norm_idx = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ norm_idx, norm_var });
    const norm_cmp = try ctx.newTemp();
    try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ norm_cmp, norm_idx, norm_size });
    try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ norm_cmp, norm_body, norm_end });
    try ctx.emitFmt("{s}:\n", .{norm_body});

    // Load input, compute (x - mean) / std
    const input_offset3 = try computeLayerNormOffset(ctx, input_info, eq.lhs.indices, norm_axis.?, &outer_vars, norm_var);
    const input_ptr3 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ input_ptr3, input_info.llvm_ptr, input_offset3 });
    const input_val3 = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ input_val3, input_ptr3 });

    const mean_final = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ mean_final, mean_var });
    const centered = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fsub double {s}, {s}\n", .{ centered, input_val3, mean_final });
    const normalized = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fdiv double {s}, {s}\n", .{ normalized, centered, std_dev });

    // Store to output
    const output_offset = try computeLayerNormOffset(ctx, output_info, eq.lhs.indices, norm_axis.?, &outer_vars, norm_var);
    const output_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ output_ptr, output_info.llvm_ptr, output_offset });
    try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ normalized, output_ptr });

    // Increment and loop
    const next_norm = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ next_norm, norm_idx });
    try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ next_norm, norm_var });
    try ctx.emitFmt("    br label %{s}\n", .{norm_start});
    try ctx.emitFmt("{s}:\n", .{norm_end});

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

/// Compute linear offset for layer norm tensor access
fn computeLayerNormOffset(
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
