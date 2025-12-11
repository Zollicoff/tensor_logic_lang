// Belief Propagation for Tensor Logic
//
// Implements loopy belief propagation as forward chaining on tensor logic programs.
// Per the paper: "Loopy belief propagation is forward chaining on the tensor logic program
// representing the model."
//
// Key insight: BP message passing IS einsum contraction:
// - Messages μ_{f→x}(x) = Σ_{neighbors} f(x, neighbors) * Π μ_{y→f}(y)
// - This is naturally expressed as tensor equations with fixpoint iteration
//
// This module provides:
// 1. Message tensor initialization
// 2. BP-style message updates as einsum
// 3. Belief computation (product of incoming messages)

const std = @import("std");
const ast = @import("../frontend/ast.zig");
const types = @import("types.zig");
const tensor = @import("tensor.zig");

const TensorInfo = types.TensorInfo;

/// Codegen context (forward declaration)
pub const CodegenContext = @import("llvm.zig").LLVMCodegen;

/// Initialize message tensors to uniform (1.0)
/// Messages start as uniform distributions before BP iterations
pub fn initMessages(ctx: *CodegenContext, msg_name: []const u8, size: usize) !void {
    try ctx.emitFmt("\n    ; Initialize BP message '{s}' to uniform\n", .{msg_name});

    const info = ctx.tensors.get(msg_name) orelse return;
    const total = info.totalSize();

    // Set all elements to 1.0 (uniform)
    for (0..total) |i| {
        const ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ ptr, info.llvm_ptr, i });
        try ctx.emitFmt("    store double 1.0, ptr {s}\n", .{ptr});
    }
    _ = size;
}

/// Normalize a message tensor (divide by sum)
/// Ensures messages remain valid probability distributions
pub fn normalizeMessage(ctx: *CodegenContext, msg_name: []const u8) !void {
    const info = ctx.tensors.get(msg_name) orelse return;
    const total = info.totalSize();

    try ctx.emitFmt("\n    ; Normalize message '{s}'\n", .{msg_name});

    // First pass: compute sum
    const sum_ptr = try ctx.newTemp();
    try ctx.emitFmt("    {s} = alloca double\n", .{sum_ptr});
    try ctx.emitFmt("    store double 0.0, ptr {s}\n", .{sum_ptr});

    for (0..total) |i| {
        const val_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ val_ptr, info.llvm_ptr, i });
        const val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ val, val_ptr });

        const old_sum = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ old_sum, sum_ptr });
        const new_sum = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fadd double {s}, {s}\n", .{ new_sum, old_sum, val });
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_sum, sum_ptr });
    }

    // Check for zero sum (avoid division by zero)
    const sum = try ctx.newTemp();
    try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ sum, sum_ptr });
    const is_zero = try ctx.newTemp();
    try ctx.emitFmt("    {s} = fcmp oeq double {s}, 0.0\n", .{ is_zero, sum });
    const safe_sum = try ctx.newTemp();
    try ctx.emitFmt("    {s} = select i1 {s}, double 1.0, double {s}\n", .{ safe_sum, is_zero, sum });

    // Second pass: divide by sum
    for (0..total) |i| {
        const val_ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ val_ptr, info.llvm_ptr, i });
        const val = try ctx.newTemp();
        try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ val, val_ptr });
        const normalized = try ctx.newTemp();
        try ctx.emitFmt("    {s} = fdiv double {s}, {s}\n", .{ normalized, val, safe_sum });
        try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ normalized, val_ptr });
    }
}

/// Compute belief as product of incoming messages
/// B(x) = Π_{f∈neighbors} μ_{f→x}(x)
pub fn computeBelief(
    ctx: *CodegenContext,
    belief_name: []const u8,
    message_names: []const []const u8,
) !void {
    const belief_info = ctx.tensors.get(belief_name) orelse return;
    const total = belief_info.totalSize();

    try ctx.emitFmt("\n    ; Compute belief '{s}' from {d} messages\n", .{ belief_name, message_names.len });

    // Initialize belief to 1.0
    for (0..total) |i| {
        const ptr = try ctx.newTemp();
        try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ ptr, belief_info.llvm_ptr, i });
        try ctx.emitFmt("    store double 1.0, ptr {s}\n", .{ptr});
    }

    // Multiply in each message
    for (message_names) |msg_name| {
        const msg_info = ctx.tensors.get(msg_name) orelse continue;

        for (0..total) |i| {
            // Load current belief
            const belief_ptr = try ctx.newTemp();
            try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ belief_ptr, belief_info.llvm_ptr, i });
            const belief_val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ belief_val, belief_ptr });

            // Load message value
            const msg_ptr = try ctx.newTemp();
            try ctx.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ msg_ptr, msg_info.llvm_ptr, i });
            const msg_val = try ctx.newTemp();
            try ctx.emitFmt("    {s} = load double, ptr {s}\n", .{ msg_val, msg_ptr });

            // Multiply
            const new_belief = try ctx.newTemp();
            try ctx.emitFmt("    {s} = fmul double {s}, {s}\n", .{ new_belief, belief_val, msg_val });
            try ctx.emitFmt("    store double {s}, ptr {s}\n", .{ new_belief, belief_ptr });
        }
    }

    // Normalize
    try normalizeMessage(ctx, belief_name);
}

/// Generate damped message update
/// new_msg = α * computed_msg + (1-α) * old_msg
/// Damping helps convergence in loopy BP
pub fn dampedUpdate(
    ctx: *CodegenContext,
    msg_name: []const u8,
    damping: f64, // α in (0, 1], typically 0.5
) !void {
    const info = ctx.tensors.get(msg_name) orelse return;
    const total = info.totalSize();

    try ctx.emitFmt("\n    ; Damped update for '{s}' (α={e})\n", .{ msg_name, damping });

    // This would require storing old messages - for now just emit a comment
    // In practice, damping would be implemented by the fixpoint iterator
    try ctx.emitFmt("    ; TODO: damped update requires storing old messages\n", .{});
    _ = total;
}

/// Check BP convergence (max absolute change in messages)
/// Returns LLVM value with 1 if converged, 0 otherwise
pub fn checkConvergence(
    ctx: *CodegenContext,
    msg_names: []const []const u8,
    threshold: f64,
) ![]const u8 {
    try ctx.emitFmt("\n    ; Check BP convergence (threshold={e})\n", .{threshold});

    // This would require comparing old vs new messages
    // For now, return constant 0 (not converged) - actual convergence
    // is handled by the fixpoint iterator in fixpoint.zig
    try ctx.emit("    ; Using fixpoint iteration for convergence\n");
    _ = msg_names;
    return "0";
}
