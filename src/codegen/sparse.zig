// Sparse Tensor Code Generation for Tensor Logic
//
// Implements COO (Coordinate) format sparse tensors:
// - indices: array of [row, col] pairs (or higher-dim coordinates)
// - values: array of values at those coordinates
// - count: number of non-zeros
// - capacity: allocated size
//
// For logic programming, relations are naturally sparse (few true entries in large domains).

const std = @import("std");
const types = @import("types.zig");

pub const SparseInfo = struct {
    name: []const u8,
    rank: usize, // number of dimensions
    dims: []usize, // size of each dimension
    indices_ptr: []const u8, // LLVM ptr to indices array
    values_ptr: []const u8, // LLVM ptr to values array
    count_ptr: []const u8, // LLVM ptr to count (i64)
    capacity: usize, // initial capacity
};

/// Generate sparse tensor allocation
/// Returns struct with pointers to indices, values, count
pub fn genSparseAlloc(
    codegen: anytype,
    name: []const u8,
    rank: usize,
    dims: []const usize,
    initial_capacity: usize,
) !SparseInfo {
    const arena = codegen.string_arena.allocator();

    // Allocate indices array: capacity * rank * sizeof(i64)
    const indices_size = initial_capacity * rank * 8;
    const indices_ptr = try codegen.newTemp();
    try codegen.emitFmt("    {s} = call ptr @malloc(i64 {d})\n", .{ indices_ptr, indices_size });

    // Allocate values array: capacity * sizeof(f64)
    const values_size = initial_capacity * 8;
    const values_ptr = try codegen.newTemp();
    try codegen.emitFmt("    {s} = call ptr @malloc(i64 {d})\n", .{ values_ptr, values_size });

    // Allocate count (single i64)
    const count_ptr = try codegen.newTemp();
    try codegen.emitFmt("    {s} = call ptr @calloc(i64 1, i64 8)\n", .{count_ptr});

    // Store initial count of 0
    try codegen.emitFmt("    store i64 0, ptr {s}\n", .{count_ptr});

    // Copy dims for storage
    const dims_copy = try arena.alloc(usize, dims.len);
    @memcpy(dims_copy, dims);

    return SparseInfo{
        .name = name,
        .rank = rank,
        .dims = dims_copy,
        .indices_ptr = indices_ptr,
        .values_ptr = values_ptr,
        .count_ptr = count_ptr,
        .capacity = initial_capacity,
    };
}

/// Generate code to set a value in sparse tensor
/// Appends to the COO arrays
pub fn genSparseSet(
    codegen: anytype,
    sparse: *const SparseInfo,
    indices: []const []const u8, // LLVM values for each index
    value: []const u8, // LLVM value to store
) !void {
    // Load current count
    const count = try codegen.newTemp();
    try codegen.emitFmt("    {s} = load i64, ptr {s}\n", .{ count, sparse.count_ptr });

    // Compute offset into indices array: count * rank
    const idx_offset = try codegen.newTemp();
    try codegen.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ idx_offset, count, sparse.rank });

    // Store each index coordinate
    for (indices, 0..) |idx, i| {
        const coord_offset = try codegen.newTemp();
        try codegen.emitFmt("    {s} = add i64 {s}, {d}\n", .{ coord_offset, idx_offset, i });

        const coord_ptr = try codegen.newTemp();
        try codegen.emitFmt("    {s} = getelementptr i64, ptr {s}, i64 {s}\n", .{ coord_ptr, sparse.indices_ptr, coord_offset });

        try codegen.emitFmt("    store i64 {s}, ptr {s}\n", .{ idx, coord_ptr });
    }

    // Store value
    const val_ptr = try codegen.newTemp();
    try codegen.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ val_ptr, sparse.values_ptr, count });
    try codegen.emitFmt("    store double {s}, ptr {s}\n", .{ value, val_ptr });

    // Increment count
    const new_count = try codegen.newTemp();
    try codegen.emitFmt("    {s} = add i64 {s}, 1\n", .{ new_count, count });
    try codegen.emitFmt("    store i64 {s}, ptr {s}\n", .{ new_count, sparse.count_ptr });
}

/// Generate loop over sparse tensor non-zeros
/// Returns: loop labels, index variables for each dimension, value variable
pub fn genSparseLoop(
    codegen: anytype,
    sparse: *const SparseInfo,
) !struct {
    loop_start: []const u8,
    loop_body: []const u8,
    loop_end: []const u8,
    iter_var: []const u8,
    index_vars: [][]const u8,
    value_var: []const u8,
} {
    const arena = codegen.string_arena.allocator();

    const loop_start = try codegen.newLabel();
    const loop_body = try codegen.newLabel();
    const loop_end = try codegen.newLabel();

    // Load count
    const count = try codegen.newTemp();
    try codegen.emitFmt("    {s} = load i64, ptr {s}\n", .{ count, sparse.count_ptr });

    // Initialize iterator
    const iter_alloc = try codegen.newTemp();
    try codegen.emitFmt("    {s} = alloca i64\n", .{iter_alloc});
    try codegen.emitFmt("    store i64 0, ptr {s}\n", .{iter_alloc});

    // Loop start
    try codegen.emitFmt("    br label %{s}\n", .{loop_start});
    try codegen.emitFmt("{s}:\n", .{loop_start});

    // Load iterator and check condition
    const iter_var = try codegen.newTemp();
    try codegen.emitFmt("    {s} = load i64, ptr {s}\n", .{ iter_var, iter_alloc });

    const cond = try codegen.newTemp();
    try codegen.emitFmt("    {s} = icmp slt i64 {s}, {s}\n", .{ cond, iter_var, count });
    try codegen.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ cond, loop_body, loop_end });

    // Loop body
    try codegen.emitFmt("{s}:\n", .{loop_body});

    // Load index coordinates
    const idx_base = try codegen.newTemp();
    try codegen.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ idx_base, iter_var, sparse.rank });

    var index_vars = try arena.alloc([]const u8, sparse.rank);
    for (0..sparse.rank) |i| {
        const coord_offset = try codegen.newTemp();
        try codegen.emitFmt("    {s} = add i64 {s}, {d}\n", .{ coord_offset, idx_base, i });

        const coord_ptr = try codegen.newTemp();
        try codegen.emitFmt("    {s} = getelementptr i64, ptr {s}, i64 {s}\n", .{ coord_ptr, sparse.indices_ptr, coord_offset });

        const coord = try codegen.newTemp();
        try codegen.emitFmt("    {s} = load i64, ptr {s}\n", .{ coord, coord_ptr });

        index_vars[i] = coord;
    }

    // Load value
    const val_ptr = try codegen.newTemp();
    try codegen.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ val_ptr, sparse.values_ptr, iter_var });

    const value_var = try codegen.newTemp();
    try codegen.emitFmt("    {s} = load double, ptr {s}\n", .{ value_var, val_ptr });

    return .{
        .loop_start = loop_start,
        .loop_body = loop_body,
        .loop_end = loop_end,
        .iter_var = iter_var,
        .iter_alloc = iter_alloc,
        .index_vars = index_vars,
        .value_var = value_var,
    };
}

/// Generate loop end (increment and branch back)
pub fn genSparseLoopEnd(
    codegen: anytype,
    iter_alloc: []const u8,
    iter_var: []const u8,
    loop_start: []const u8,
    loop_end: []const u8,
) !void {
    // Increment iterator
    const next_iter = try codegen.newTemp();
    try codegen.emitFmt("    {s} = add i64 {s}, 1\n", .{ next_iter, iter_var });
    try codegen.emitFmt("    store i64 {s}, ptr {s}\n", .{ next_iter, iter_alloc });

    // Branch back to loop start
    try codegen.emitFmt("    br label %{s}\n", .{loop_start});

    // Loop end
    try codegen.emitFmt("{s}:\n", .{loop_end});
}
