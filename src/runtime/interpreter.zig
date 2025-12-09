// Tensor Logic Interpreter
//
// Executes parsed Tensor Logic programs by:
// 1. Building a symbol table of tensors/relations
// 2. Evaluating equations in order
// 3. Supporting fixpoint iteration for recursive rules

const std = @import("std");
const ast = @import("../frontend/ast.zig");
const tensor = @import("tensor.zig");
const einsum = @import("einsum.zig");

const DenseTensor = tensor.DenseTensor;
const SparseTensor = tensor.SparseTensor;
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const Index = ast.Index;

pub const InterpreterError = error{
    UndefinedTensor,
    ShapeMismatch,
    InvalidIndex,
    DomainNotDefined,
    OutOfMemory,
    NotImplemented,
    EinsumError,
};

/// Extract the base name from an index (for matching contracted indices)
fn getIndexName(idx: Index) ?[]const u8 {
    return switch (idx) {
        .name => |n| n,
        .primed => |p| p,
        .arithmetic => |a| a.base,
        .virtual => |v| v,
        .normalize => |n| n,
        .div => |d| d.index,
        .constant, .slice => null,
    };
}

/// Runtime value - either a tensor or a scalar
pub const Value = union(enum) {
    tensor_val: Tensor,
    f64_val: f64,
    i64_val: i64,
    bool_val: bool,

    pub fn deinit(self: *Value) void {
        switch (self.*) {
            .tensor_val => |*t| t.deinit(),
            else => {},
        }
    }
};

/// Domain definition - size and type of an index variable
pub const Domain = struct {
    name: []const u8,
    size: usize,
};

/// Schema for a tensor - maps index positions to domain names
pub const TensorSchema = struct {
    /// Domain name for each index position
    index_domains: []const []const u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *TensorSchema) void {
        self.allocator.free(self.index_domains);
    }
};

/// Interpreter state
pub const Interpreter = struct {
    allocator: std.mem.Allocator,
    /// Named tensors/relations
    tensors: std.StringHashMap(Tensor),
    /// Domain sizes
    domains: std.StringHashMap(usize),
    /// Tensor schemas (variable to domain bindings)
    schemas: std.StringHashMap(TensorSchema),
    /// Default domain size (for undeclared domains)
    default_domain_size: usize,

    pub fn init(allocator: std.mem.Allocator) Interpreter {
        return Interpreter{
            .allocator = allocator,
            .tensors = std.StringHashMap(Tensor).init(allocator),
            .domains = std.StringHashMap(usize).init(allocator),
            .schemas = std.StringHashMap(TensorSchema).init(allocator),
            .default_domain_size = 100, // Default domain size
        };
    }

    pub fn deinit(self: *Interpreter) void {
        var tensor_iter = self.tensors.valueIterator();
        while (tensor_iter.next()) |t| {
            var tensor_copy = t.*;
            tensor_copy.deinit();
        }
        self.tensors.deinit();
        self.domains.deinit();

        var schema_iter = self.schemas.valueIterator();
        while (schema_iter.next()) |s| {
            var schema_copy = s.*;
            schema_copy.deinit();
        }
        self.schemas.deinit();
    }

    /// Define a domain with a specific size
    pub fn defineDomain(self: *Interpreter, name: []const u8, size: usize) !void {
        try self.domains.put(name, size);
    }

    /// Get domain size (returns default if not defined)
    pub fn getDomainSize(self: *Interpreter, name: []const u8) usize {
        return self.domains.get(name) orelse self.default_domain_size;
    }

    /// Get domain size for a variable in a tensor context
    /// Looks up the schema if available, otherwise falls back to domain lookup
    pub fn getIndexDomainSize(self: *Interpreter, tensor_name: []const u8, var_name: []const u8, position: usize) usize {
        // First, check if there's a schema for this tensor
        if (self.schemas.get(tensor_name)) |schema| {
            if (position < schema.index_domains.len) {
                return self.getDomainSize(schema.index_domains[position]);
            }
        }
        // Fall back to looking up the variable name as a domain
        return self.getDomainSize(var_name);
    }

    /// Define a tensor with given shape
    /// If is_sparse is true, uses sparse storage (efficient for relations)
    pub fn defineTensor(self: *Interpreter, name: []const u8, shape: []const usize, is_sparse: bool) !void {
        const t = if (is_sparse)
            try tensor.sparseF64(self.allocator, shape)
        else
            try tensor.zeros(self.allocator, shape);
        try self.tensors.put(name, t);
    }

    /// Define a sparse tensor
    pub fn defineSparseTensor(self: *Interpreter, name: []const u8, shape: []const usize) !void {
        const t = try tensor.sparseF64(self.allocator, shape);
        try self.tensors.put(name, t);
    }

    /// Get a tensor by name
    pub fn getTensor(self: *Interpreter, name: []const u8) ?*Tensor {
        return self.tensors.getPtr(name);
    }

    /// Execute a full program
    pub fn execute(self: *Interpreter, program: *const ast.Program) !void {
        for (program.statements) |stmt| {
            try self.executeStatement(&stmt);
        }
    }

    /// Execute a single statement
    pub fn executeStatement(self: *Interpreter, stmt: *const ast.Statement) !void {
        switch (stmt.*) {
            .equation => |eq| try self.executeEquation(&eq),
            .domain_decl => |d| {
                if (d.size) |size| {
                    try self.defineDomain(d.name, @intCast(size));
                }
            },
            .sparse_decl => |s| {
                // Skip if tensor already exists (idempotent declaration)
                if (self.tensors.contains(s.name)) {
                    return;
                }

                // Determine shape from indices and build schema
                var shape = std.ArrayListUnmanaged(usize){};
                defer shape.deinit(self.allocator);

                var index_domains = std.ArrayListUnmanaged([]const u8){};

                for (s.indices) |idx| {
                    const domain = idx.domain orelse idx.name; // Use variable name as domain if not specified
                    const size = self.getDomainSize(domain);
                    shape.append(self.allocator, size) catch return InterpreterError.OutOfMemory;
                    index_domains.append(self.allocator, domain) catch return InterpreterError.OutOfMemory;
                }

                // Register schema for this tensor
                const schema = TensorSchema{
                    .index_domains = index_domains.toOwnedSlice(self.allocator) catch return InterpreterError.OutOfMemory,
                    .allocator = self.allocator,
                };
                self.schemas.put(s.name, schema) catch return InterpreterError.OutOfMemory;

                try self.defineTensor(s.name, shape.items, s.is_boolean);
            },
            .import_stmt => {
                // TODO: implement imports
            },
            .export_stmt => {
                // TODO: implement exports
            },
            .comment => {
                // Comments are no-ops
            },
        }
    }

    /// Execute a tensor equation
    fn executeEquation(self: *Interpreter, eq: *const ast.Equation) !void {
        // Check if this is an element-wise assignment (all constant indices)
        var all_constant = true;
        var element_indices = std.ArrayListUnmanaged(usize){};
        defer element_indices.deinit(self.allocator);

        for (eq.lhs.indices) |idx| {
            switch (idx) {
                .constant => |c| {
                    element_indices.append(self.allocator, @intCast(c)) catch return InterpreterError.OutOfMemory;
                },
                else => {
                    all_constant = false;
                    break;
                },
            }
        }

        if (all_constant and element_indices.items.len > 0) {
            // Element-wise assignment: T[0,1] = value
            return self.executeElementAssignment(eq, element_indices.items);
        }

        // Regular tensor equation
        // Determine output shape from LHS indices
        var shape = std.ArrayListUnmanaged(usize){};
        defer shape.deinit(self.allocator);

        // Collect output index names for einsum
        var output_indices = std.ArrayListUnmanaged([]const u8){};
        defer output_indices.deinit(self.allocator);

        for (eq.lhs.indices, 0..) |idx, pos| {
            const size = switch (idx) {
                .name => |n| self.getIndexDomainSize(eq.lhs.name, n, pos),
                .constant => |c| @as(usize, @intCast(c)) + 1,
                .primed => |p| self.getIndexDomainSize(eq.lhs.name, p, pos),
                else => self.default_domain_size,
            };
            shape.append(self.allocator, size) catch return InterpreterError.OutOfMemory;

            // Also collect the index name for einsum
            if (getIndexName(idx)) |name| {
                output_indices.append(self.allocator, name) catch return InterpreterError.OutOfMemory;
            }
        }

        // Ensure tensor exists
        if (self.tensors.getPtr(eq.lhs.name) == null) {
            try self.defineTensor(eq.lhs.name, shape.items, eq.lhs.is_boolean);
        }

        // Evaluate RHS with output indices for einsum contraction
        var rhs_val = try self.evaluateExprWithIndices(eq.rhs, output_indices.items);
        defer rhs_val.deinit();

        // Apply accumulation operator
        const lhs_tensor = self.tensors.getPtr(eq.lhs.name) orelse return InterpreterError.UndefinedTensor;

        switch (eq.op) {
            .assign => {
                // Replace value
                switch (rhs_val) {
                    .tensor_val => {
                        var old = lhs_tensor.*;
                        old.deinit();
                        lhs_tensor.* = rhs_val.tensor_val;
                        rhs_val = .{ .f64_val = 0 }; // Prevent double-free
                    },
                    .f64_val => |v| {
                        // Scalar assignment - fill tensor with scalar value
                        switch (lhs_tensor.*) {
                            .f64_dense => |*t| t.fill(v),
                            else => {},
                        }
                    },
                    .i64_val => |v| {
                        // Integer scalar - convert to f64 and fill
                        switch (lhs_tensor.*) {
                            .f64_dense => |*t| t.fill(@floatFromInt(v)),
                            else => {},
                        }
                    },
                    .bool_val => |v| {
                        // Boolean - fill with 1.0 or 0.0
                        switch (lhs_tensor.*) {
                            .f64_dense => |*t| t.fill(if (v) 1.0 else 0.0),
                            else => {},
                        }
                    },
                }
            },
            .add => {
                // Add to existing
                switch (lhs_tensor.*) {
                    .f64_dense => |*t| {
                        switch (rhs_val) {
                            .tensor_val => |rv| {
                                if (rv == .f64_dense) {
                                    t.add(&rv.f64_dense);
                                }
                            },
                            .f64_val => |v| {
                                // Add scalar to all elements
                                for (t.data) |*x| {
                                    x.* += v;
                                }
                            },
                            .i64_val => |v| {
                                const fv: f64 = @floatFromInt(v);
                                for (t.data) |*x| {
                                    x.* += fv;
                                }
                            },
                            else => {},
                        }
                    },
                    else => {},
                }
            },
            .max => {
                // Take element-wise max
                switch (lhs_tensor.*) {
                    .f64_dense => |*t| {
                        switch (rhs_val) {
                            .tensor_val => |rv| {
                                if (rv == .f64_dense) {
                                    for (t.data, rv.f64_dense.data) |*a, b| {
                                        a.* = @max(a.*, b);
                                    }
                                }
                            },
                            .f64_val => |v| {
                                for (t.data) |*x| {
                                    x.* = @max(x.*, v);
                                }
                            },
                            .i64_val => |v| {
                                const fv: f64 = @floatFromInt(v);
                                for (t.data) |*x| {
                                    x.* = @max(x.*, fv);
                                }
                            },
                            else => {},
                        }
                    },
                    else => {},
                }
            },
            .min => {
                // Take element-wise min
                switch (lhs_tensor.*) {
                    .f64_dense => |*t| {
                        switch (rhs_val) {
                            .tensor_val => |rv| {
                                if (rv == .f64_dense) {
                                    for (t.data, rv.f64_dense.data) |*a, b| {
                                        a.* = @min(a.*, b);
                                    }
                                }
                            },
                            .f64_val => |v| {
                                for (t.data) |*x| {
                                    x.* = @min(x.*, v);
                                }
                            },
                            .i64_val => |v| {
                                const fv: f64 = @floatFromInt(v);
                                for (t.data) |*x| {
                                    x.* = @min(x.*, fv);
                                }
                            },
                            else => {},
                        }
                    },
                    else => {},
                }
            },
            .avg => {
                // Running average
                // TODO: implement
            },
        }
    }

    /// Execute element-wise assignment: T[0,1] = value
    fn executeElementAssignment(self: *Interpreter, eq: *const ast.Equation, indices: []const usize) !void {
        // Ensure tensor exists with shape that accommodates the indices
        if (self.tensors.getPtr(eq.lhs.name) == null) {
            // Create tensor with shape large enough for these indices
            var shape = std.ArrayListUnmanaged(usize){};
            defer shape.deinit(self.allocator);
            for (indices) |idx| {
                shape.append(self.allocator, idx + 1) catch return InterpreterError.OutOfMemory;
            }
            try self.defineTensor(eq.lhs.name, shape.items, eq.lhs.is_boolean);
        }

        // Get the tensor
        const lhs_tensor = self.tensors.getPtr(eq.lhs.name) orelse return InterpreterError.UndefinedTensor;

        // Check if indices are within bounds, resize if needed
        const t_shape = lhs_tensor.shape();
        var need_resize = false;
        for (indices, 0..) |idx, dim| {
            if (dim >= t_shape.dims.len or idx >= t_shape.dims[dim]) {
                need_resize = true;
                break;
            }
        }

        if (need_resize) {
            // For now, return error - proper resize would need more work
            // In a full implementation, we'd resize the tensor
            return InterpreterError.InvalidIndex;
        }

        // Evaluate RHS (should be a scalar)
        var rhs_val = try self.evaluateExprWithIndices(eq.rhs, &[_][]const u8{});
        defer rhs_val.deinit();

        // Get scalar value
        const scalar: f64 = switch (rhs_val) {
            .f64_val => |v| v,
            .i64_val => |v| @floatFromInt(v),
            .bool_val => |v| if (v) 1.0 else 0.0,
            .tensor_val => return InterpreterError.ShapeMismatch, // Can't assign tensor to element
        };

        // Set the element
        switch (lhs_tensor.*) {
            .f64_dense => |*dense| {
                switch (eq.op) {
                    .assign => dense.set(indices, scalar),
                    .add => {
                        const old = dense.get(indices);
                        dense.set(indices, old + scalar);
                    },
                    .max => {
                        const old = dense.get(indices);
                        dense.set(indices, @max(old, scalar));
                    },
                    .min => {
                        const old = dense.get(indices);
                        dense.set(indices, @min(old, scalar));
                    },
                    .avg => {
                        // For avg, we'd need a count - skip for now
                    },
                }
            },
            else => return InterpreterError.NotImplemented,
        }
    }

    /// Evaluate an expression to a value
    /// output_indices: the indices that should appear in the final result (from LHS)
    fn evaluateExprWithIndices(self: *Interpreter, expr: *const ast.Expr, output_indices: []const []const u8) InterpreterError!Value {
        switch (expr.*) {
            .tensor_ref => |ref| {
                if (self.tensors.get(ref.name)) |t| {
                    // Return a copy of the tensor with actual data
                    switch (t) {
                        .f64_dense => |dense| {
                            const new_dense = DenseTensor(f64).init(self.allocator, dense.shape.dims) catch return InterpreterError.OutOfMemory;
                            @memcpy(new_dense.data, dense.data);
                            return Value{ .tensor_val = Tensor{ .f64_dense = new_dense } };
                        },
                        else => {
                            // For other types, create zeros with same shape
                            var shape_dims = std.ArrayListUnmanaged(usize){};
                            defer shape_dims.deinit(self.allocator);
                            for (t.shape().dims) |d| {
                                shape_dims.append(self.allocator, d) catch return InterpreterError.OutOfMemory;
                            }
                            const new_t = tensor.zeros(self.allocator, shape_dims.items) catch return InterpreterError.OutOfMemory;
                            return Value{ .tensor_val = new_t };
                        },
                    }
                } else {
                    // Auto-create tensor with zeros
                    var shape = std.ArrayListUnmanaged(usize){};
                    defer shape.deinit(self.allocator);

                    for (ref.indices, 0..) |idx, pos| {
                        const size = switch (idx) {
                            .name => |n| self.getIndexDomainSize(ref.name, n, pos),
                            .primed => |p| self.getIndexDomainSize(ref.name, p, pos),
                            else => self.default_domain_size,
                        };
                        shape.append(self.allocator, size) catch return InterpreterError.OutOfMemory;
                    }

                    const new_t = tensor.zeros(self.allocator, shape.items) catch return InterpreterError.OutOfMemory;
                    return Value{ .tensor_val = new_t };
                }
            },

            .literal => |lit| {
                switch (lit) {
                    .integer => |i| return Value{ .i64_val = i },
                    .float => |f| return Value{ .f64_val = f },
                    .boolean => |b| return Value{ .bool_val = b },
                    .string => return InterpreterError.NotImplemented,
                }
            },

            .product => |prod| {
                // Implicit multiplication - this is the core einsum operation
                // A[i,j] B[j,k] contracts over j to produce C[i,k]
                if (prod.factors.len == 0) {
                    return Value{ .f64_val = 1.0 };
                }

                if (prod.factors.len == 1) {
                    return try self.evaluateExprWithIndices(prod.factors[0], output_indices);
                }

                // For two factors, perform einsum contraction
                // We need to find shared indices and contract over them
                return try self.evaluateProductEinsum(prod.factors, output_indices);
            },

            .nonlinearity => |nl| {
                // For nonlinearity, the output indices are the same as input
                // since nonlinearity is applied element-wise
                var arg_val = try self.evaluateExprWithIndices(nl.arg, output_indices);

                // Apply nonlinearity in-place
                switch (arg_val) {
                    .tensor_val => |*t| {
                        switch (t.*) {
                            .f64_dense => |*dense| {
                                switch (nl.func) {
                                    .step => einsum.stepTensor(dense),
                                    .relu => einsum.reluTensor(dense),
                                    .sigmoid => einsum.sigmoidTensor(dense),
                                    .softmax => einsum.softmaxTensor(dense),
                                    .tanh => einsum.tanhTensor(dense),
                                    .exp => einsum.expTensor(dense),
                                    .log => einsum.logTensor(dense),
                                    .abs => einsum.absTensor(dense),
                                    .sqrt => einsum.sqrtTensor(dense),
                                    .sin => einsum.sinTensor(dense),
                                    .cos => einsum.cosTensor(dense),
                                    .norm => {}, // TODO: implement norm
                                }
                            },
                            .f64_sparse => |*sparse| {
                                // Apply nonlinearities to sparse tensors
                                switch (nl.func) {
                                    .step => einsum.stepSparseTensor(sparse),
                                    .relu => einsum.reluSparseTensor(sparse),
                                    .sigmoid => einsum.sigmoidSparseTensor(sparse),
                                    else => {}, // Other ops not yet implemented for sparse
                                }
                            },
                            else => {},
                        }
                    },
                    .f64_val => |*v| {
                        v.* = switch (nl.func) {
                            .step => einsum.step(v.*),
                            .relu => einsum.relu(v.*),
                            .sigmoid => einsum.sigmoid(v.*),
                            .tanh => einsum.tanh_fn(v.*),
                            .exp => einsum.exp_fn(v.*),
                            .log => einsum.log_fn(v.*),
                            .abs => einsum.abs_fn(v.*),
                            .sqrt => einsum.sqrt_fn(v.*),
                            .sin => einsum.sin_fn(v.*),
                            .cos => einsum.cos_fn(v.*),
                            .softmax, .norm => v.*, // No-op for scalars
                        };
                    },
                    else => {},
                }

                return arg_val;
            },

            .binary => |bin| {
                var left = try self.evaluateExprWithIndices(bin.left, output_indices);
                defer left.deinit();
                var right = try self.evaluateExprWithIndices(bin.right, output_indices);
                defer right.deinit();

                // Handle scalar operations
                if (left == .f64_val and right == .f64_val) {
                    const l = left.f64_val;
                    const r = right.f64_val;
                    const result: f64 = switch (bin.op) {
                        .add => l + r,
                        .sub => l - r,
                        .mul => l * r,
                        .div => l / r,
                        .pow => std.math.pow(f64, l, r),
                        else => l,
                    };
                    return Value{ .f64_val = result };
                }

                return InterpreterError.NotImplemented;
            },

            .unary => |un| {
                var operand = try self.evaluateExprWithIndices(un.operand, output_indices);

                switch (un.op) {
                    .negate => {
                        if (operand == .f64_val) {
                            operand.f64_val = -operand.f64_val;
                        }
                    },
                    .not => {
                        if (operand == .bool_val) {
                            operand.bool_val = !operand.bool_val;
                        }
                    },
                }

                return operand;
            },

            .group => |inner| {
                return self.evaluateExprWithIndices(inner, output_indices);
            },

            .embed, .conditional => {
                return InterpreterError.NotImplemented;
            },
        }
    }

    /// Evaluate a product of tensors using einsum contraction
    /// This is THE core tensor logic operation
    fn evaluateProductEinsum(self: *Interpreter, factors: []*ast.Expr, output_indices: []const []const u8) InterpreterError!Value {
        // For now, handle the common case of two tensor references
        // A[i,j] B[j,k] with output indices [i, k] contracts over j

        if (factors.len < 2) {
            return InterpreterError.NotImplemented;
        }

        // Extract tensor refs from factors
        const factor0 = factors[0];
        const factor1 = factors[1];

        // Both must be tensor refs for einsum
        if (factor0.* != .tensor_ref or factor1.* != .tensor_ref) {
            // Fall back to evaluating each and returning product
            // (scalar multiplication case)
            var val0 = try self.evaluateExprWithIndices(factor0, output_indices);
            defer val0.deinit();
            var val1 = try self.evaluateExprWithIndices(factor1, output_indices);
            defer val1.deinit();

            if (val0 == .f64_val and val1 == .f64_val) {
                return Value{ .f64_val = val0.f64_val * val1.f64_val };
            }
            return InterpreterError.NotImplemented;
        }

        const ref0 = factor0.tensor_ref;
        const ref1 = factor1.tensor_ref;

        // Get or create tensors
        const t0 = self.tensors.get(ref0.name) orelse blk: {
            // Create tensor if it doesn't exist
            var shape = std.ArrayListUnmanaged(usize){};
            defer shape.deinit(self.allocator);
            for (ref0.indices, 0..) |idx, pos| {
                const size = if (getIndexName(idx)) |n| self.getIndexDomainSize(ref0.name, n, pos) else self.default_domain_size;
                shape.append(self.allocator, size) catch return InterpreterError.OutOfMemory;
            }
            self.defineTensor(ref0.name, shape.items, ref0.is_boolean) catch return InterpreterError.OutOfMemory;
            break :blk self.tensors.get(ref0.name).?;
        };

        const t1 = self.tensors.get(ref1.name) orelse blk: {
            var shape = std.ArrayListUnmanaged(usize){};
            defer shape.deinit(self.allocator);
            for (ref1.indices, 0..) |idx, pos| {
                const size = if (getIndexName(idx)) |n| self.getIndexDomainSize(ref1.name, n, pos) else self.default_domain_size;
                shape.append(self.allocator, size) catch return InterpreterError.OutOfMemory;
            }
            self.defineTensor(ref1.name, shape.items, ref1.is_boolean) catch return InterpreterError.OutOfMemory;
            break :blk self.tensors.get(ref1.name).?;
        };

        // Extract index names from tensor refs
        var indices0 = std.ArrayListUnmanaged([]const u8){};
        defer indices0.deinit(self.allocator);
        for (ref0.indices) |idx| {
            if (getIndexName(idx)) |name| {
                indices0.append(self.allocator, name) catch return InterpreterError.OutOfMemory;
            }
        }

        var indices1 = std.ArrayListUnmanaged([]const u8){};
        defer indices1.deinit(self.allocator);
        for (ref1.indices) |idx| {
            if (getIndexName(idx)) |name| {
                indices1.append(self.allocator, name) catch return InterpreterError.OutOfMemory;
            }
        }

        // Dispatch based on tensor types (sparse vs dense)
        const is_sparse0 = t0.isSparse();
        const is_sparse1 = t1.isSparse();

        if (is_sparse0 and is_sparse1) {
            // Sparse-sparse einsum
            const sparse0 = switch (t0) {
                .f64_sparse => |s| s,
                else => return InterpreterError.NotImplemented,
            };
            const sparse1 = switch (t1) {
                .f64_sparse => |s| s,
                else => return InterpreterError.NotImplemented,
            };

            const result_sparse = einsum.sparseEinsum2(
                self.allocator,
                &sparse0,
                indices0.items,
                &sparse1,
                indices1.items,
                output_indices,
            ) catch return InterpreterError.EinsumError;

            return Value{ .tensor_val = Tensor{ .f64_sparse = result_sparse } };
        } else if (is_sparse0 and !is_sparse1) {
            // Sparse-dense einsum
            const sparse0 = switch (t0) {
                .f64_sparse => |s| s,
                else => return InterpreterError.NotImplemented,
            };
            const dense1 = switch (t1) {
                .f64_dense => |d| d,
                else => return InterpreterError.NotImplemented,
            };

            const result_sparse = einsum.sparseDenseEinsum2(
                self.allocator,
                &sparse0,
                indices0.items,
                &dense1,
                indices1.items,
                output_indices,
            ) catch return InterpreterError.EinsumError;

            return Value{ .tensor_val = Tensor{ .f64_sparse = result_sparse } };
        } else {
            // Dense-dense einsum (or dense-sparse, converted to dense)
            const dense0 = switch (t0) {
                .f64_dense => |d| d,
                .f64_sparse => |s| s.toDense(self.allocator) catch return InterpreterError.OutOfMemory,
                else => return InterpreterError.NotImplemented,
            };
            const dense1 = switch (t1) {
                .f64_dense => |d| d,
                .f64_sparse => |s| s.toDense(self.allocator) catch return InterpreterError.OutOfMemory,
                else => return InterpreterError.NotImplemented,
            };

            // Call einsum2 to perform the contraction
            const result_dense = einsum.einsum2(
                self.allocator,
                &dense0,
                indices0.items,
                &dense1,
                indices1.items,
                output_indices,
            ) catch return InterpreterError.EinsumError;

            return Value{ .tensor_val = Tensor{ .f64_dense = result_dense } };
        }
    }

    /// Run fixpoint iteration for recursive rules
    /// Continues until tensors stop changing or max_iters reached
    /// Returns the number of iterations performed
    pub fn runFixpoint(self: *Interpreter, program: *const ast.Program, max_iters: usize) !usize {
        var iter: usize = 0;

        while (iter < max_iters) : (iter += 1) {
            // Save current state checksums
            const old_checksum = self.computeStateChecksum();

            // Execute one iteration
            try self.execute(program);

            // Check for convergence
            const new_checksum = self.computeStateChecksum();
            if (old_checksum == new_checksum) {
                // Converged - no changes this iteration
                return iter + 1;
            }
        }

        return max_iters; // Did not converge within limit
    }

    /// Compute a checksum of all tensor values for convergence detection
    fn computeStateChecksum(self: *Interpreter) u64 {
        var checksum: u64 = 0;
        var iter = self.tensors.iterator();
        while (iter.next()) |entry| {
            const t = entry.value_ptr.*;
            switch (t) {
                .f64_dense => |dense| {
                    for (dense.data) |val| {
                        // Hash the float bits
                        const bits: u64 = @bitCast(val);
                        checksum = checksum *% 31 +% bits;
                    }
                },
                .f64_sparse => |sparse| {
                    // Hash sparse tensor: include indices and values
                    for (sparse.indices.items, sparse.values.items) |idx, val| {
                        for (idx) |i| {
                            checksum = checksum *% 31 +% i;
                        }
                        const bits: u64 = @bitCast(val);
                        checksum = checksum *% 31 +% bits;
                    }
                },
                else => {},
            }
        }
        return checksum;
    }

    /// Execute program until fixpoint, with default max iterations
    pub fn executeToFixpoint(self: *Interpreter, program: *const ast.Program) !usize {
        return self.runFixpoint(program, 100);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "interpreter basic" {
    const allocator = std.testing.allocator;

    var interp = Interpreter.init(allocator);
    defer interp.deinit();

    // Define a domain
    try interp.defineDomain("Person", 10);

    // Define a sparse relation
    try interp.defineTensor("Parent", &[_]usize{ 10, 10 }, true);

    try std.testing.expect(interp.getTensor("Parent") != null);
}

test "interpreter domain size" {
    const allocator = std.testing.allocator;

    var interp = Interpreter.init(allocator);
    defer interp.deinit();

    try interp.defineDomain("Person", 50);
    try std.testing.expectEqual(@as(usize, 50), interp.getDomainSize("Person"));
    try std.testing.expectEqual(@as(usize, 100), interp.getDomainSize("Unknown")); // default
}

test "einsum matmul" {
    // Test C[i,k] = A[i,j] B[j,k] where A is [2,3], B is [3,2]
    const allocator = std.testing.allocator;

    // Create tensors directly
    var a = try DenseTensor(f64).init(allocator, &[_]usize{ 2, 3 });
    defer a.deinit();
    // A = [[1, 2, 3], [4, 5, 6]]
    a.set(&[_]usize{ 0, 0 }, 1);
    a.set(&[_]usize{ 0, 1 }, 2);
    a.set(&[_]usize{ 0, 2 }, 3);
    a.set(&[_]usize{ 1, 0 }, 4);
    a.set(&[_]usize{ 1, 1 }, 5);
    a.set(&[_]usize{ 1, 2 }, 6);

    var b = try DenseTensor(f64).init(allocator, &[_]usize{ 3, 2 });
    defer b.deinit();
    // B = [[1, 2], [3, 4], [5, 6]]
    b.set(&[_]usize{ 0, 0 }, 1);
    b.set(&[_]usize{ 0, 1 }, 2);
    b.set(&[_]usize{ 1, 0 }, 3);
    b.set(&[_]usize{ 1, 1 }, 4);
    b.set(&[_]usize{ 2, 0 }, 5);
    b.set(&[_]usize{ 2, 1 }, 6);

    // C[i,k] = A[i,j] B[j,k]
    const a_indices = [_][]const u8{ "i", "j" };
    const b_indices = [_][]const u8{ "j", "k" };
    const out_indices = [_][]const u8{ "i", "k" };

    var c = try einsum.einsum2(allocator, &a, &a_indices, &b, &b_indices, &out_indices);
    defer c.deinit();

    // C = A @ B = [[22, 28], [49, 64]]
    try std.testing.expectEqual(@as(f64, 22), c.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 28), c.get(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f64, 49), c.get(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f64, 64), c.get(&[_]usize{ 1, 1 }));
}

test "transitive closure" {
    // Test: Ancestor[i,k] = step(Parent[i,j] * Ancestor[j,k])
    // This is the core tensor logic operation for Datalog rules
    //
    // Family tree: 0 -> 1 -> 2 (0 is grandparent of 2)
    // Parent[0,1] = 1, Parent[1,2] = 1
    //
    // After one iteration:
    // Ancestor[0,1] = 1 (direct)
    // Ancestor[1,2] = 1 (direct)
    // Ancestor[0,2] = step(Parent[0,1] * Ancestor[1,2]) = step(1*1) = 1

    const allocator = std.testing.allocator;

    // Create Parent relation (3x3)
    var parent = try DenseTensor(f64).init(allocator, &[_]usize{ 3, 3 });
    defer parent.deinit();
    parent.set(&[_]usize{ 0, 1 }, 1.0); // 0 is parent of 1
    parent.set(&[_]usize{ 1, 2 }, 1.0); // 1 is parent of 2

    // Create Ancestor relation, initially copy of Parent
    var ancestor = try DenseTensor(f64).init(allocator, &[_]usize{ 3, 3 });
    defer ancestor.deinit();
    ancestor.set(&[_]usize{ 0, 1 }, 1.0);
    ancestor.set(&[_]usize{ 1, 2 }, 1.0);

    // Now compute: Ancestor[i,k] += step(Parent[i,j] * Ancestor[j,k])
    // This is einsum contraction over j
    const parent_indices = [_][]const u8{ "i", "j" };
    const ancestor_indices = [_][]const u8{ "j", "k" };
    const out_indices = [_][]const u8{ "i", "k" };

    var product = try einsum.einsum2(allocator, &parent, &parent_indices, &ancestor, &ancestor_indices, &out_indices);
    defer product.deinit();

    // Apply step function
    einsum.stepTensor(&product);

    // Add to ancestor
    ancestor.add(&product);

    // Verify results
    try std.testing.expectEqual(@as(f64, 1), ancestor.get(&[_]usize{ 0, 1 })); // direct: 0 -> 1
    try std.testing.expectEqual(@as(f64, 1), ancestor.get(&[_]usize{ 1, 2 })); // direct: 1 -> 2
    try std.testing.expectEqual(@as(f64, 1), ancestor.get(&[_]usize{ 0, 2 })); // transitive: 0 -> 1 -> 2
    try std.testing.expectEqual(@as(f64, 0), ancestor.get(&[_]usize{ 2, 0 })); // no relation
    try std.testing.expectEqual(@as(f64, 0), ancestor.get(&[_]usize{ 0, 0 })); // no self-loop
}
