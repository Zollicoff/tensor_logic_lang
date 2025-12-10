// Tensor allocation and indexing for LLVM code generation
//
// Handles tensor memory allocation, dimension tracking, and linear offset computation.

const std = @import("std");
const ast = @import("../frontend/ast.zig");
const types = @import("types.zig");
const TensorInfo = types.TensorInfo;

/// Codegen context (forward declaration - actual struct in llvm.zig)
pub const CodegenContext = @import("llvm.zig").LLVMCodegen;

/// Allocate a tensor (zero-initialized via calloc)
pub fn allocateTensor(ctx: *CodegenContext, name: []const u8, indices: []const ast.Index) !void {
    var dims = std.ArrayListUnmanaged(usize){};

    // Use pre-computed max dimensions if available (handles constant indices across multiple equations)
    if (ctx.tensor_max_dims.get(name)) |max_dims| {
        for (max_dims) |d| {
            try dims.append(ctx.allocator, d);
        }
    } else {
        // Fallback to computing from current indices
        for (indices) |idx| {
            const size: usize = switch (idx) {
                .name => |n| ctx.domains.get(n) orelse 10,
                .constant => |c| @as(usize, @intCast(c)) + 1,
                else => 10,
            };
            try dims.append(ctx.allocator, size);
        }
    }

    if (dims.items.len == 0) {
        try dims.append(ctx.allocator, 1); // Scalar
    }

    // Compute strides (row-major)
    var strides = try ctx.allocator.alloc(usize, dims.items.len);
    var stride: usize = 1;
    var i: usize = dims.items.len;
    while (i > 0) {
        i -= 1;
        strides[i] = stride;
        stride *= dims.items[i];
    }

    const total = stride;
    const llvm_ptr = try std.fmt.allocPrint(ctx.string_arena.allocator(), "%{s}.data", .{name});

    try ctx.emitFmt("    ; Allocate tensor '{s}' with dims [", .{name});
    for (dims.items, 0..) |d, idx| {
        if (idx > 0) try ctx.emit(", ");
        try ctx.emitFmt("{d}", .{d});
    }
    try ctx.emitFmt("] = {d} elements\n", .{total});
    try ctx.emitFmt("    {s} = call ptr @calloc(i64 {d}, i64 8)\n", .{ llvm_ptr, total });

    const dims_owned = try ctx.allocator.dupe(usize, dims.items);
    dims.deinit(ctx.allocator);

    try ctx.tensors.put(ctx.allocator, name, .{
        .name = name,
        .llvm_ptr = llvm_ptr,
        .rank = dims_owned.len,
        .dims = dims_owned,
        .strides = strides,
    });
}

/// Update the max dimensions for a tensor based on indices seen in an equation
pub fn updateTensorMaxDims(ctx: *CodegenContext, name: []const u8, indices: []const ast.Index) !void {
    const num_dims = indices.len;
    if (num_dims == 0) return;

    if (ctx.tensor_max_dims.get(name)) |existing| {
        // Update existing max dims
        for (indices, 0..) |idx, i| {
            if (i >= existing.len) break;
            const size: usize = switch (idx) {
                .name => |n| ctx.domains.get(n) orelse 10,
                .normalize => |n| ctx.domains.get(n) orelse 10,
                .constant => |c| @as(usize, @intCast(c)) + 1,
                else => 10,
            };
            existing[i] = @max(existing[i], size);
        }
    } else {
        // Create new entry
        var dims = try ctx.allocator.alloc(usize, num_dims);
        for (indices, 0..) |idx, i| {
            dims[i] = switch (idx) {
                .name => |n| ctx.domains.get(n) orelse 10,
                .normalize => |n| ctx.domains.get(n) orelse 10,
                .constant => |c| @as(usize, @intCast(c)) + 1,
                else => 10,
            };
        }
        try ctx.tensor_max_dims.put(ctx.allocator, name, dims);
    }
}

/// Compute linear offset from strides and index values
pub fn computeLinearOffset(ctx: *CodegenContext, strides: []const usize, idx_vals: []const []const u8) ![]const u8 {
    if (strides.len == 0 or idx_vals.len == 0) {
        return "0";
    }

    var result = try ctx.newTemp();
    // First term
    const first_mul = try ctx.newTemp();
    try ctx.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ first_mul, idx_vals[0], strides[0] });
    try ctx.emitFmt("    {s} = add i64 0, {s}\n", .{ result, first_mul });

    // Remaining terms
    for (1..@min(strides.len, idx_vals.len)) |i| {
        const mul = try ctx.newTemp();
        try ctx.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ mul, idx_vals[i], strides[i] });
        const new_result = try ctx.newTemp();
        try ctx.emitFmt("    {s} = add i64 {s}, {s}\n", .{ new_result, result, mul });
        result = new_result;
    }

    return result;
}
