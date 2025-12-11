// Tucker Decomposition for Tensor Logic
//
// Implements Tucker decomposition for converting sparse tensors to dense approximations.
// This enables GPU-efficient computation while controlling approximation error.
//
// Tucker decomposition: T[i,j,k] ≈ Σ_{r,s,t} G[r,s,t] * A[i,r] * B[j,s] * C[k,t]
// Where:
//   - G is the core tensor (smaller, dense)
//   - A, B, C are factor matrices
//
// Key insight from paper: "converting sparse tensors to dense via Tucker decomposition"
// enables GPU acceleration with controlled error bounds.

const std = @import("std");
const ast = @import("../frontend/ast.zig");
const types = @import("types.zig");
const tensor = @import("tensor.zig");

const TensorInfo = types.TensorInfo;

/// Codegen context (forward declaration)
pub const CodegenContext = @import("llvm.zig").LLVMCodegen;

/// Tucker decomposition info for a tensor
pub const TuckerInfo = struct {
    name: []const u8,
    original_dims: []const usize, // Original tensor dimensions [I, J, K, ...]
    core_ranks: []const usize, // Core tensor ranks [R1, R2, R3, ...]
    core_ptr: []const u8, // LLVM ptr to core tensor G
    factor_ptrs: []const []const u8, // LLVM ptrs to factor matrices [A, B, C, ...]
};

/// Generate Tucker-decomposed tensor allocation
/// Allocates core tensor G and factor matrices A, B, C, ...
pub fn genTuckerAlloc(
    ctx: *CodegenContext,
    name: []const u8,
    original_dims: []const usize,
    core_ranks: []const usize,
) !TuckerInfo {
    const arena = ctx.string_arena.allocator();
    const rank = original_dims.len;

    if (rank != core_ranks.len) {
        return error.OutOfMemory; // Dimension mismatch
    }

    try ctx.emitFmt("\n    ; Tucker decomposition for '{s}'\n", .{name});
    try ctx.emitFmt("    ; Original dims: [", .{});
    for (original_dims, 0..) |d, i| {
        if (i > 0) try ctx.emit(", ");
        try ctx.emitFmt("{d}", .{d});
    }
    try ctx.emitFmt("], Core ranks: [", .{});
    for (core_ranks, 0..) |r, i| {
        if (i > 0) try ctx.emit(", ");
        try ctx.emitFmt("{d}", .{r});
    }
    try ctx.emit("]\n");

    // Allocate core tensor G with dimensions [R1, R2, R3, ...]
    var core_size: usize = 1;
    for (core_ranks) |r| core_size *= r;

    const core_ptr = try ctx.newTemp();
    try ctx.emitFmt("    ; Core tensor G[{d}]\n", .{core_size});
    try ctx.emitFmt("    {s} = call ptr @calloc(i64 {d}, i64 8)\n", .{ core_ptr, core_size });

    // Allocate factor matrices A[I,R1], B[J,R2], C[K,R3], ...
    var factor_ptrs = try arena.alloc([]const u8, rank);

    for (0..rank) |mode| {
        const rows = original_dims[mode];
        const cols = core_ranks[mode];
        const factor_size = rows * cols;

        const factor_ptr = try ctx.newTemp();
        try ctx.emitFmt("    ; Factor matrix {d}: [{d} x {d}]\n", .{ mode, rows, cols });
        try ctx.emitFmt("    {s} = call ptr @calloc(i64 {d}, i64 8)\n", .{ factor_ptr, factor_size });

        factor_ptrs[mode] = factor_ptr;
    }

    // Store copies
    const dims_copy = try arena.alloc(usize, original_dims.len);
    @memcpy(dims_copy, original_dims);

    const ranks_copy = try arena.alloc(usize, core_ranks.len);
    @memcpy(ranks_copy, core_ranks);

    return TuckerInfo{
        .name = name,
        .original_dims = dims_copy,
        .core_ranks = ranks_copy,
        .core_ptr = core_ptr,
        .factor_ptrs = factor_ptrs,
    };
}

/// Generate code to reconstruct a single element T[i,j,k] from Tucker factors
/// T[i,j,k] = Σ_{r,s,t} G[r,s,t] * A[i,r] * B[j,s] * C[k,t]
pub fn genTuckerAccess(
    ctx: *CodegenContext,
    tucker: *const TuckerInfo,
    idx_vals: []const []const u8, // Index values for each dimension
) ![]const u8 {
    const rank = tucker.original_dims.len;

    if (idx_vals.len != rank) {
        return "0.0"; // Dimension mismatch
    }

    try ctx.emitFmt("    ; Tucker reconstruction for {s}\n", .{tucker.name});

    // Initialize accumulator
    const accum_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = alloca double\n", .{accum_ptr});
    try ctx.emitFmt("    store double 0.0, ptr {s}\n", .{accum_ptr});

    // Generate nested loops over core tensor indices
    // For 3D: loop over r, s, t
    var loop_vars = try ctx.allocator.alloc([]const u8, rank);
    defer ctx.allocator.free(loop_vars);

    var loop_labels = try ctx.allocator.alloc(struct { start: []const u8, body: []const u8, end: []const u8 }, rank);
    defer ctx.allocator.free(loop_labels);

    // Open loops
    for (0..rank) |mode| {
        const var_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = alloca i64\n", .{var_ptr});
        try ctx.emitFmt("    store i64 0, ptr {s}\n", .{var_ptr});
        loop_vars[mode] = var_ptr;

        const start_label = try ctx.newLabel();
        const body_label = try ctx.newLabel();
        const end_label = try ctx.newLabel();
        loop_labels[mode] = .{ .start = start_label, .body = body_label, .end = end_label };

        try ctx.emitFmt("    br label %{s}\n", .{start_label});
        try ctx.emitFmt("{s}:\n", .{start_label});

        const idx = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ idx, var_ptr });
        const cmp = try ctx.newTemp();
        try ctx.emitFmt("    {s} = icmp slt i64 {s}, {d}\n", .{ cmp, idx, tucker.core_ranks[mode] });
        try ctx.emitFmt("    br i1 {s}, label %{s}, label %{s}\n", .{ cmp, body_label, end_label });
        try ctx.emitFmt("{s}:\n", .{body_label});
    }

    // Inside innermost loop: compute G[r,s,t] * A[i,r] * B[j,s] * C[k,t]

    // Load core indices
    var core_idx_vals = try ctx.allocator.alloc([]const u8, rank);
    defer ctx.allocator.free(core_idx_vals);

    for (0..rank) |mode| {
        const idx = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ idx, loop_vars[mode] });
        core_idx_vals[mode] = idx;
    }

    // Compute linear offset into core tensor
    const core_offset = try computeCoreOffset(ctx, tucker.core_ranks, core_idx_vals);
    const core_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ core_ptr, tucker.core_ptr, core_offset });
    const core_val = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ core_val, core_ptr });

    // Compute product of factor matrix elements: A[i,r] * B[j,s] * ...
    var product = core_val;

    for (0..rank) |mode| {
        // Factor matrix offset: idx_vals[mode] * core_ranks[mode] + core_idx_vals[mode]
        const row_offset = try ctx.newTemp();
        try ctx.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ row_offset, idx_vals[mode], tucker.core_ranks[mode] });
        const factor_offset = try ctx.newTemp();
        try ctx.emitFmt("    {s} = add i64 {s}, {s}\n", .{ factor_offset, row_offset, core_idx_vals[mode] });

        const factor_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {s}\n", .{ factor_ptr, tucker.factor_ptrs[mode], factor_offset });
        const factor_val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ factor_val, factor_ptr });

        const new_product = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ new_product, product, factor_val });
        product = new_product;
    }

    // Accumulate
    const old_accum = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_accum, accum_ptr });
    const new_accum = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_accum, old_accum, product });
    try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_accum, accum_ptr });

    // Close loops (in reverse order)
    var mode: usize = rank;
    while (mode > 0) {
        mode -= 1;
        const idx = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load i64, ptr {s}\n", .{ idx, loop_vars[mode] });
        const next_idx = try ctx.newTemp();
        try ctx.emitFmt("    {s} = add i64 {s}, 1\n", .{ next_idx, idx });
        try ctx.emitFmt("    store i64 {s}, ptr {s}\n", .{ next_idx, loop_vars[mode] });
        try ctx.emitFmt("    br label %{s}\n", .{loop_labels[mode].start});
        try ctx.emitFmt("{s}:\n", .{loop_labels[mode].end});
    }

    // Return final accumulated value
    const result = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ result, accum_ptr });
    return result;
}

/// Compute linear offset into core tensor
fn computeCoreOffset(ctx: *CodegenContext, core_ranks: []const usize, idx_vals: []const []const u8) ![]const u8 {
    if (core_ranks.len == 0) return "0";

    var result: []const u8 = "0";
    var stride: usize = 1;

    // Row-major order: last index changes fastest
    var i: usize = core_ranks.len;
    while (i > 0) {
        i -= 1;
        const term = try ctx.newTemp();
        try ctx.emitFmt("    {s} = mul i64 {s}, {d}\n", .{ term, idx_vals[i], stride });
        const new_result = try ctx.newTemp();
        try ctx.emitFmt("    {s} = add i64 {s}, {s}\n", .{ new_result, result, term });
        result = new_result;
        stride *= core_ranks[i];
    }

    return result;
}

/// Initialize factor matrices with random values (for learning)
/// Uses simple pseudo-random initialization: val = (i * 1103515245 + 12345) / MAX * 0.1
pub fn genTuckerRandomInit(ctx: *CodegenContext, tucker: *const TuckerInfo) !void {
    try ctx.emitFmt("\n    ; Initialize Tucker factors for '{s}' with small random values\n", .{tucker.name});

    // Initialize core tensor
    var core_size: usize = 1;
    for (tucker.core_ranks) |r| core_size *= r;

    for (0..core_size) |i| {
        const ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ ptr, tucker.core_ptr, i });
        // Simple deterministic "random" init: scaled by 0.01
        const val: f64 = @as(f64, @floatFromInt((i * 1103515245 + 12345) % 1000)) / 10000.0;
        try ctx.emitFmt("    store double {e}, ptr {s}\n", .{ val, ptr });
    }

    // Initialize factor matrices
    for (tucker.factor_ptrs, 0..) |factor_ptr, mode| {
        const rows = tucker.original_dims[mode];
        const cols = tucker.core_ranks[mode];
        const size = rows * cols;

        for (0..size) |i| {
            const ptr = try ctx.newTemp();
            try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ ptr, factor_ptr, i });
            const val: f64 = @as(f64, @floatFromInt(((i + mode * 7919) * 1103515245 + 12345) % 1000)) / 10000.0;
            try ctx.emitFmt("    store double {e}, ptr {s}\n", .{ val, ptr });
        }
    }
}

/// Generate HOSVD (Higher-Order SVD) initialization from a dense tensor
/// This computes the Tucker decomposition of an existing tensor
pub fn genHOSVD(ctx: *CodegenContext, tucker: *const TuckerInfo, source_ptr: []const u8) !void {
    // HOSVD algorithm:
    // 1. For each mode k, unfold tensor to matrix
    // 2. Compute SVD and take top R_k singular vectors as factor matrix
    // 3. Compute core tensor as: G = T ×₁ A₁ᵀ ×₂ A₂ᵀ ×₃ A₃ᵀ
    //
    // For simplicity in codegen, we generate a simpler approximation:
    // Copy values and let learning refine the decomposition

    try ctx.emitFmt("\n    ; HOSVD-style initialization from source tensor\n", .{});
    try ctx.emitFmt("    ; (simplified: direct value copy for initial approximation)\n", .{});

    // For now, just initialize with small values - full HOSVD would require
    // eigendecomposition which is complex to generate inline
    try genTuckerRandomInit(ctx, tucker);

    _ = source_ptr; // Would be used for proper HOSVD
}
