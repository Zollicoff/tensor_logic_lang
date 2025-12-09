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

        /// Clone the sparse tensor
        pub fn clone(self: *const Self) !Self {
            var new_tensor = try Self.init(self.allocator, self.shape.dims);

            for (self.indices.items, self.values.items) |idx, val| {
                try new_tensor.set(idx, val);
            }

            return new_tensor;
        }

        /// Convert to dense tensor
        pub fn toDense(self: *const Self, allocator: std.mem.Allocator) !DenseTensor(T) {
            var dense = try DenseTensor(T).init(allocator, self.shape.dims);

            for (self.indices.items, self.values.items) |idx, val| {
                dense.set(idx, val);
            }

            return dense;
        }

        /// Create sparse tensor from dense (only keeping non-zeros)
        pub fn fromDense(allocator: std.mem.Allocator, dense: *const DenseTensor(T)) !Self {
            var sparse = try Self.init(allocator, dense.shape.dims);

            // Iterate through all elements and add non-zeros
            const numel = dense.shape.numel();
            const rank = dense.shape.dims.len;

            // Pre-compute strides
            var strides = try allocator.alloc(usize, rank);
            defer allocator.free(strides);
            var stride: usize = 1;
            var i = rank;
            while (i > 0) {
                i -= 1;
                strides[i] = stride;
                stride *= dense.shape.dims[i];
            }

            var flat_idx: usize = 0;
            while (flat_idx < numel) : (flat_idx += 1) {
                const val = dense.data[flat_idx];
                if (val != std.mem.zeroes(T)) {
                    // Convert flat index to multi-index
                    var multi_idx = try allocator.alloc(usize, rank);
                    defer allocator.free(multi_idx);

                    var remaining = flat_idx;
                    for (0..rank) |d| {
                        multi_idx[d] = remaining / strides[d];
                        remaining = remaining % strides[d];
                    }

                    try sparse.set(multi_idx, val);
                }
            }

            return sparse;
        }

        /// Remove zero entries (useful after operations that might create zeros)
        pub fn compact(self: *Self) void {
            var write_idx: usize = 0;
            for (self.indices.items, self.values.items, 0..) |idx, val, read_idx| {
                if (val != std.mem.zeroes(T)) {
                    if (write_idx != read_idx) {
                        self.indices.items[write_idx] = idx;
                        self.values.items[write_idx] = val;
                    }
                    write_idx += 1;
                } else {
                    // Free the index array for removed entries
                    self.allocator.free(idx);
                }
            }
            self.indices.shrinkRetainingCapacity(write_idx);
            self.values.shrinkRetainingCapacity(write_idx);
        }

        /// Iterator over non-zero entries
        pub const Entry = struct {
            indices: []const usize,
            value: T,
        };

        pub fn iterator(self: *const Self) Iterator {
            return Iterator{ .tensor = self, .pos = 0 };
        }

        pub const Iterator = struct {
            tensor: *const Self,
            pos: usize,

            pub fn next(self: *Iterator) ?Entry {
                if (self.pos >= self.tensor.values.items.len) return null;
                const entry = Entry{
                    .indices = self.tensor.indices.items[self.pos],
                    .value = self.tensor.values.items[self.pos],
                };
                self.pos += 1;
                return entry;
            }
        };
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

/// Create a sparse f64 tensor
pub fn sparseF64(allocator: std.mem.Allocator, shape_dims: []const usize) !Tensor {
    return Tensor{ .f64_sparse = try SparseTensor(f64).init(allocator, shape_dims) };
}

/// Count non-zero elements in a tensor
pub fn countNonZero(t: *const Tensor) usize {
    return switch (t.*) {
        .f64_dense => |dense| blk: {
            var count: usize = 0;
            for (dense.data) |val| {
                if (val != 0.0) count += 1;
            }
            break :blk count;
        },
        .f64_sparse => |sparse| sparse.values.items.len,
        .bool_sparse => |sparse| sparse.values.items.len,
        else => 0,
    };
}

/// Calculate sparsity ratio (0.0 = all zeros, 1.0 = all non-zero)
pub fn sparsityRatio(t: *const Tensor) f64 {
    const nnz = countNonZero(t);
    const total = t.shape().numel();
    if (total == 0) return 0.0;
    return @as(f64, @floatFromInt(nnz)) / @as(f64, @floatFromInt(total));
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

test "sparse tensor clone" {
    const allocator = std.testing.allocator;

    var t = try SparseTensor(f64).init(allocator, &[_]usize{ 10, 10 });
    defer t.deinit();

    try t.set(&[_]usize{ 1, 2 }, 3.0);
    try t.set(&[_]usize{ 5, 7 }, 9.0);

    var clone = try t.clone();
    defer clone.deinit();

    try std.testing.expectEqual(@as(usize, 2), clone.nnz());
    try std.testing.expectEqual(@as(f64, 3.0), clone.get(&[_]usize{ 1, 2 }));
    try std.testing.expectEqual(@as(f64, 9.0), clone.get(&[_]usize{ 5, 7 }));
}

test "sparse to dense conversion" {
    const allocator = std.testing.allocator;

    var sparse = try SparseTensor(f64).init(allocator, &[_]usize{ 3, 3 });
    defer sparse.deinit();

    try sparse.set(&[_]usize{ 0, 0 }, 1.0);
    try sparse.set(&[_]usize{ 1, 1 }, 2.0);
    try sparse.set(&[_]usize{ 2, 2 }, 3.0);

    var dense = try sparse.toDense(allocator);
    defer dense.deinit();

    try std.testing.expectEqual(@as(f64, 1.0), dense.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 2.0), dense.get(&[_]usize{ 1, 1 }));
    try std.testing.expectEqual(@as(f64, 3.0), dense.get(&[_]usize{ 2, 2 }));
    try std.testing.expectEqual(@as(f64, 0.0), dense.get(&[_]usize{ 0, 1 }));
}

test "dense to sparse conversion" {
    const allocator = std.testing.allocator;

    var dense = try DenseTensor(f64).init(allocator, &[_]usize{ 3, 3 });
    defer dense.deinit();

    dense.set(&[_]usize{ 0, 0 }, 1.0);
    dense.set(&[_]usize{ 1, 1 }, 2.0);
    dense.set(&[_]usize{ 2, 2 }, 3.0);

    var sparse = try SparseTensor(f64).fromDense(allocator, &dense);
    defer sparse.deinit();

    try std.testing.expectEqual(@as(usize, 3), sparse.nnz());
    try std.testing.expectEqual(@as(f64, 1.0), sparse.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 2.0), sparse.get(&[_]usize{ 1, 1 }));
    try std.testing.expectEqual(@as(f64, 3.0), sparse.get(&[_]usize{ 2, 2 }));
}

test "sparse tensor iterator" {
    const allocator = std.testing.allocator;

    var t = try SparseTensor(f64).init(allocator, &[_]usize{ 5, 5 });
    defer t.deinit();

    try t.set(&[_]usize{ 1, 2 }, 10.0);
    try t.set(&[_]usize{ 3, 4 }, 20.0);

    var iter = t.iterator();
    var count: usize = 0;
    var sum: f64 = 0.0;
    while (iter.next()) |entry| {
        count += 1;
        sum += entry.value;
    }

    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expectEqual(@as(f64, 30.0), sum);
}
