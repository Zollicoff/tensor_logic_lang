// Shared types for LLVM code generation
//
// These types are used across multiple codegen modules.

const std = @import("std");

/// Information about a tensor for code generation
pub const TensorInfo = struct {
    name: []const u8,
    llvm_ptr: []const u8, // LLVM SSA name for data pointer
    rank: usize,
    dims: []const usize,
    strides: []const usize,

    pub fn totalSize(self: TensorInfo) usize {
        var size: usize = 1;
        for (self.dims) |d| size *= d;
        return size;
    }
};

/// Index variable info for loop generation
pub const IndexVar = struct {
    name: []const u8,
    size: usize,
    llvm_var: []const u8, // LLVM SSA name for loop variable
    is_contracted: bool, // True if summed out (on RHS but not LHS)
};

/// Loop label set for generated loops
pub const LoopLabels = struct {
    start: []const u8,
    body: []const u8,
    end: []const u8,
};
