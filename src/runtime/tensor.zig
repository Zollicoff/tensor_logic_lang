// Tensor Logic Runtime - Tensor Storage and Operations
//
// Core insight from the paper: Relations are sparse Boolean tensors.
// We support both dense and sparse storage, automatically choosing based on sparsity.

const std = @import("std");

/// Tensor element types
pub const DType = enum {
    bool_type,
    int32,
    int64,
    float32,
    float64,

    pub fn size(self: DType) usize {
        return switch (self) {
            .bool_type => 1,
            .int32 => 4,
            .int64 => 8,
            .float32 => 4,
            .float64 => 8,
        };
    }

    pub fn name(self: DType) []const u8 {
        return switch (self) {
            .bool_type => "bool",
            .int32 => "i32",
            .int64 => "i64",
            .float32 => "f32",
            .float64 => "f64",
        };
    }
};

/// Shape of a tensor - list of dimension sizes
pub const Shape = struct {
    dims: []const usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, dims: []const usize) !Shape {
        const dims_copy = try allocator.alloc(usize, dims.len);
        @memcpy(dims_copy, dims);
        return Shape{
            .dims = dims_copy,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Shape) void {
        self.allocator.free(self.dims);
    }

    pub fn rank(self: Shape) usize {
        return self.dims.len;
    }

    pub fn numel(self: Shape) usize {
        var n: usize = 1;
        for (self.dims) |d| {
            n *= d;
        }
        return n;
    }

    pub fn eql(self: Shape, other: Shape) bool {
        if (self.dims.len != other.dims.len) return false;
        for (self.dims, other.dims) |a, b| {
            if (a != b) return false;
        }
        return true;
    }
};

/// Dense tensor storage - contiguous array
pub fn DenseTensor(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        shape: Shape,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, shape_dims: []const usize) !Self {
            const shape = try Shape.init(allocator, shape_dims);
            const data = try allocator.alloc(T, shape.numel());
            @memset(data, std.mem.zeroes(T));
            return Self{
                .data = data,
                .shape = shape,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            var shape = self.shape;
            shape.deinit();
        }

        pub fn get(self: *const Self, indices: []const usize) T {
            const flat_idx = self.flatIndex(indices);
            return self.data[flat_idx];
        }

        pub fn set(self: *Self, indices: []const usize, value: T) void {
            const flat_idx = self.flatIndex(indices);
            self.data[flat_idx] = value;
        }

        fn flatIndex(self: *const Self, indices: []const usize) usize {
            var idx: usize = 0;
            var stride: usize = 1;
            var i = self.shape.dims.len;
            while (i > 0) {
                i -= 1;
                idx += indices[i] * stride;
                stride *= self.shape.dims[i];
            }
            return idx;
        }

        pub fn fill(self: *Self, value: T) void {
            @memset(self.data, value);
        }

        /// Element-wise addition
        pub fn add(self: *Self, other: *const Self) void {
            for (self.data, other.data) |*a, b| {
                a.* += b;
            }
        }

        /// Element-wise multiplication
        pub fn mul(self: *Self, other: *const Self) void {
            for (self.data, other.data) |*a, b| {
                a.* *= b;
            }
        }

        /// Scalar multiplication
        pub fn scale(self: *Self, scalar: T) void {
            for (self.data) |*a| {
                a.* *= scalar;
            }
        }
    };
}

/// Sparse tensor using coordinate format (COO)
/// Efficient for very sparse tensors like Boolean relations
pub fn SparseTensor(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Indices for each non-zero element: indices[i] is the multi-index for element i
        indices: std.ArrayListUnmanaged([]usize),
        /// Values for each non-zero element
        values: std.ArrayListUnmanaged(T),
        shape: Shape,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, shape_dims: []const usize) !Self {
            const shape = try Shape.init(allocator, shape_dims);
            return Self{
                .indices = .{},
                .values = .{},
                .shape = shape,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.indices.items) |idx| {
                self.allocator.free(idx);
            }
            self.indices.deinit(self.allocator);
            self.values.deinit(self.allocator);
            var shape = self.shape;
            shape.deinit();
        }

        pub fn nnz(self: *const Self) usize {
            return self.values.items.len;
        }

        pub fn set(self: *Self, indices: []const usize, value: T) !void {
            // Check if index already exists
            for (self.indices.items, 0..) |existing, i| {
                if (indexEqual(existing, indices)) {
                    self.values.items[i] = value;
                    return;
                }
            }

            // Add new entry
            const idx_copy = try self.allocator.alloc(usize, indices.len);
            @memcpy(idx_copy, indices);
            try self.indices.append(self.allocator, idx_copy);
            try self.values.append(self.allocator, value);
        }

        pub fn get(self: *const Self, indices: []const usize) T {
            for (self.indices.items, 0..) |existing, i| {
                if (indexEqual(existing, indices)) {
                    return self.values.items[i];
                }
            }
            return std.mem.zeroes(T);
        }

        fn indexEqual(a: []const usize, b: []const usize) bool {
            if (a.len != b.len) return false;
            for (a, b) |x, y| {
                if (x != y) return false;
            }
            return true;
        }
    };
}

/// Dynamic tensor that can hold any type
pub const Tensor = union(enum) {
    bool_dense: DenseTensor(bool),
    bool_sparse: SparseTensor(bool),
    i32_dense: DenseTensor(i32),
    i64_dense: DenseTensor(i64),
    f32_dense: DenseTensor(f32),
    f32_sparse: SparseTensor(f32),
    f64_dense: DenseTensor(f64),
    f64_sparse: SparseTensor(f64),

    pub fn deinit(self: *Tensor) void {
        switch (self.*) {
            .bool_dense => |*t| t.deinit(),
            .bool_sparse => |*t| t.deinit(),
            .i32_dense => |*t| t.deinit(),
            .i64_dense => |*t| t.deinit(),
            .f32_dense => |*t| t.deinit(),
            .f32_sparse => |*t| t.deinit(),
            .f64_dense => |*t| t.deinit(),
            .f64_sparse => |*t| t.deinit(),
        }
    }

    pub fn shape(self: *const Tensor) Shape {
        return switch (self.*) {
            .bool_dense => |t| t.shape,
            .bool_sparse => |t| t.shape,
            .i32_dense => |t| t.shape,
            .i64_dense => |t| t.shape,
            .f32_dense => |t| t.shape,
            .f32_sparse => |t| t.shape,
            .f64_dense => |t| t.shape,
            .f64_sparse => |t| t.shape,
        };
    }

    pub fn dtype(self: *const Tensor) DType {
        return switch (self.*) {
            .bool_dense, .bool_sparse => .bool_type,
            .i32_dense => .int32,
            .i64_dense => .int64,
            .f32_dense, .f32_sparse => .float32,
            .f64_dense, .f64_sparse => .float64,
        };
    }

    pub fn isSparse(self: *const Tensor) bool {
        return switch (self.*) {
            .bool_sparse, .f32_sparse, .f64_sparse => true,
            else => false,
        };
    }
};

/// Create a dense f64 tensor
pub fn zeros(allocator: std.mem.Allocator, shape_dims: []const usize) !Tensor {
    return Tensor{ .f64_dense = try DenseTensor(f64).init(allocator, shape_dims) };
}

/// Create a dense f64 tensor filled with ones
pub fn ones(allocator: std.mem.Allocator, shape_dims: []const usize) !Tensor {
    var t = try DenseTensor(f64).init(allocator, shape_dims);
    t.fill(1.0);
    return Tensor{ .f64_dense = t };
}

/// Create a sparse boolean tensor (for relations)
pub fn sparseRelation(allocator: std.mem.Allocator, shape_dims: []const usize) !Tensor {
    return Tensor{ .bool_sparse = try SparseTensor(bool).init(allocator, shape_dims) };
}

// ============================================================================
// Tests
// ============================================================================

test "dense tensor creation" {
    const allocator = std.testing.allocator;

    var t = try DenseTensor(f64).init(allocator, &[_]usize{ 3, 4 });
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 2), t.shape.rank());
    try std.testing.expectEqual(@as(usize, 12), t.shape.numel());
}

test "dense tensor get/set" {
    const allocator = std.testing.allocator;

    var t = try DenseTensor(f64).init(allocator, &[_]usize{ 2, 3 });
    defer t.deinit();

    t.set(&[_]usize{ 0, 0 }, 1.0);
    t.set(&[_]usize{ 1, 2 }, 5.0);

    try std.testing.expectEqual(@as(f64, 1.0), t.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 5.0), t.get(&[_]usize{ 1, 2 }));
    try std.testing.expectEqual(@as(f64, 0.0), t.get(&[_]usize{ 0, 1 }));
}

test "sparse tensor" {
    const allocator = std.testing.allocator;

    var t = try SparseTensor(bool).init(allocator, &[_]usize{ 100, 100 });
    defer t.deinit();

    try t.set(&[_]usize{ 5, 10 }, true);
    try t.set(&[_]usize{ 50, 75 }, true);

    try std.testing.expectEqual(@as(usize, 2), t.nnz());
    try std.testing.expectEqual(true, t.get(&[_]usize{ 5, 10 }));
    try std.testing.expectEqual(true, t.get(&[_]usize{ 50, 75 }));
    try std.testing.expectEqual(false, t.get(&[_]usize{ 0, 0 }));
}

test "tensor union" {
    const allocator = std.testing.allocator;

    var t = try zeros(allocator, &[_]usize{ 2, 3 });
    defer t.deinit();

    try std.testing.expectEqual(DType.float64, t.dtype());
    try std.testing.expectEqual(false, t.isSparse());
}
