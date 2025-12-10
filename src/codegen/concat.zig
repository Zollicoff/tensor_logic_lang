// Concatenation operation for LLVM code generation
//
// Implements concat: Y[dm] = concat(X[h, d])
// The contracted dimension (h) is flattened into the output.
// Output dm = h * d, where h is the first dimension and d is the rest.

const std = @import("std");
const ast = @import("../frontend/ast.zig");
const types = @import("types.zig");
const tensor = @import("tensor.zig");

const TensorInfo = types.TensorInfo;
const LoopLabels = types.LoopLabels;

/// Codegen context (forward declaration)
pub const CodegenContext = @import("llvm.zig").LLVMCodegen;

/// Generate concat operation
pub fn genConcat(ctx: *CodegenContext, eq: *const ast.Equation) !void {
    try ctx.emit("    ; Concat operation\n");

    // Get the input tensor from the concat argument
    const nl = eq.rhs.nonlinearity;
    const input_ref = if (nl.arg.* == .tensor_ref) nl.arg.tensor_ref else {
        try ctx.emit("    ; Error: concat argument must be tensor reference\n");
        return;
    };

    // Ensure output tensor exists
    if (!ctx.tensors.contains(eq.lhs.name)) {
        try tensor.allocateTensor(ctx, eq.lhs.name, eq.lhs.indices);
    }

    // Ensure input tensor exists
    const input_info = ctx.tensors.get(input_ref.name) orelse {
        try ctx.emit("    ; Error: input tensor not found\n");
        return;
    };
    const output_info = ctx.tensors.get(eq.lhs.name).?;

    // Find which input indices are contracted (not on LHS)
    // For concat(A[H,D]) -> B[DM], H is contracted
    var lhs_indices = std.StringHashMapUnmanaged(void){};
    defer lhs_indices.deinit(ctx.allocator);

    for (eq.lhs.indices) |idx| {
        const name = switch (idx) {
            .name => |n| n,
            else => continue,
        };
        try lhs_indices.put(ctx.allocator, name, {});
    }

    // Collect input indices and identify which are contracted
    var input_indices = std.ArrayListUnmanaged(struct { name: []const u8, size: usize, is_contracted: bool }){};
    defer input_indices.deinit(ctx.allocator);

    for (input_ref.indices) |idx| {
        const name = switch (idx) {
            .name => |n| n,
            else => continue,
        };
        const size = ctx.domains.get(name) orelse 10;
        const is_contracted = !lhs_indices.contains(name);
        try input_indices.append(ctx.allocator, .{
            .name = name,
            .size = size,
            .is_contracted = is_contracted,
        });
    }

    // For simplicity, we handle the common case:
    // concat(A[H, D]) -> B[DM] where DM = H * D
    // The first contracted index (H) is unrolled into the output

    if (input_indices.items.len < 2) {
        try ctx.emit("    ; Error: concat needs at least 2 input dimensions\n");
        return;
    }

    // Find the contracted dimension (to flatten) and kept dimensions
    var concat_dim_idx: ?usize = null;
    var concat_size: usize = 1;
    for (input_indices.items, 0..) |idx_info, i| {
        if (idx_info.is_contracted) {
            concat_dim_idx = i;
            concat_size = idx_info.size;
            break;
        }
    }

    if (concat_dim_idx == null) {
        try ctx.emit("    ; Error: no contracted dimension for concat\n");
        return;
    }

    // Allocate loop variables
    var loop_vars = std.StringHashMapUnmanaged([]const u8){};
    defer loop_vars.deinit(ctx.allocator);

    for (input_indices.items) |idx_info| {
        const var_name = try std.fmt.allocPrint(ctx.string_arena.allocator(), "%concat_{s}", .{idx_info.name});
        try ctx.emitFmt("    {s} = alloca i64\n", .{var_name});
        try loop_vars.put(ctx.allocator, idx_info.name, var_name);
    }

    // Generate nested loops for all input dimensions
    var loop_labels = std.ArrayListUnmanaged(LoopLabels){};
    defer loop_labels.deinit(ctx.allocator);

    for (input_indices.items) |idx_info| {
        const var_name = loop_vars.get(idx_info.name).?;
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
        try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ cmp, idx_val, idx_info.size });
        try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ cmp, body_label, end_label });
        try ctx.emitFmt("{s}:\n", .{body_label});
    }

    // Compute input offset
    var input_idx_vals = std.ArrayListUnmanaged([]const u8){};
    defer input_idx_vals.deinit(ctx.allocator);

    for (input_indices.items) |idx_info| {
        const var_name = loop_vars.get(idx_info.name).?;
        const val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ val, var_name });
        try input_idx_vals.append(ctx.allocator, val);
    }

    const input_offset = try tensor.computeLinearOffset(ctx, input_info.strides, input_idx_vals.items);
    const input_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ input_ptr, input_info.llvm_ptr, input_offset });
    const input_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ input_val, input_ptr });

    // Compute output offset
    // For concat, we need to compute: concat_idx * inner_size + inner_idx
    // Where concat_idx is the contracted dimension and inner_size is product of remaining dims
    const concat_idx = input_idx_vals.items[concat_dim_idx.?];

    // Compute inner size (product of dimensions after the concat dimension)
    var inner_size: usize = 1;
    for (input_indices.items[concat_dim_idx.? + 1 ..]) |idx_info| {
        inner_size *= idx_info.size;
    }

    // Compute inner index (from dimensions after concat)
    var inner_idx: []const u8 = "0";
    if (concat_dim_idx.? + 1 < input_indices.items.len) {
        var stride: usize = 1;
        var accumulated = try ctx.newTemp();
        try ctx.emitFmt("    {s} = add i64 0, 0\n", .{accumulated});

        var j: usize = input_indices.items.len;
        while (j > concat_dim_idx.? + 1) {
            j -= 1;
            const idx_val = input_idx_vals.items[j];
            const term = try ctx.newTemp();
            try ctx.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ term, idx_val, stride });
            const new_acc = try ctx.newTemp();
            try ctx.emitFmt("    {s} = add i64 {s}, {s}\n", .{ new_acc, accumulated, term });
            accumulated = new_acc;
            stride *= input_indices.items[j].size;
        }
        inner_idx = accumulated;
    }

    // Output index = concat_idx * inner_size + inner_idx
    const scaled_concat = try ctx.newTemp();
    try ctx.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ scaled_concat, concat_idx, inner_size });
    const output_idx = try ctx.newTemp();
    try ctx.emitFmt("    {s} = add i64 {s}, {s}\n", .{ output_idx, scaled_concat, inner_idx });

    // Store to output
    const output_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ output_ptr, output_info.llvm_ptr, output_idx });
    try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ input_val, output_ptr });

    // Close loops
    var i: usize = input_indices.items.len;
    while (i > 0) {
        i -= 1;
        const idx_info = input_indices.items[i];
        const var_name = loop_vars.get(idx_info.name).?;
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
