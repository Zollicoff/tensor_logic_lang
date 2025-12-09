// Type system for Tensor Logic
//
// Tracks two main aspects:
// 1. Value type: Boolean (0/1) vs Real (any float)
// 2. Shape: Domain sizes for each index dimension
//
// Type inference rules:
// - Boolean relations: Ancestor(x,y) - uses parentheses, values are 0 or 1
// - Real tensors: W[i,j] - uses brackets, values are any float
// - step(x) -> Boolean (thresholds at 0)
// - sigmoid(x) -> Real in (0,1)
// - Product of Booleans -> Boolean (logical AND in semiring)
// - Sum of Booleans -> Real (counts)

const std = @import("std");
const ast = @import("ast.zig");
const tokens = @import("tokens.zig");

/// Value type of a tensor/expression
pub const ValueType = enum {
    boolean, // 0 or 1 only (for relations/predicates)
    real, // Any real number
    unknown, // Not yet inferred

    pub fn format(self: ValueType) []const u8 {
        return switch (self) {
            .boolean => "Boolean",
            .real => "Real",
            .unknown => "Unknown",
        };
    }
};

/// Shape information for a tensor
pub const Shape = struct {
    /// Domain names for each dimension (e.g., ["Person", "Person"] for a relation)
    domains: []const []const u8,
    /// Sizes for each dimension (null if not yet known)
    sizes: []const ?usize,

    pub fn rank(self: Shape) usize {
        return self.domains.len;
    }
};

/// Type information for a tensor
pub const TensorType = struct {
    value_type: ValueType,
    shape: ?Shape,
    is_sparse: bool,

    pub fn boolean() TensorType {
        return .{ .value_type = .boolean, .shape = null, .is_sparse = false };
    }

    pub fn real() TensorType {
        return .{ .value_type = .real, .shape = null, .is_sparse = false };
    }

    pub fn unknown() TensorType {
        return .{ .value_type = .unknown, .shape = null, .is_sparse = false };
    }
};

/// Type error information
pub const TypeError = struct {
    message: []const u8,
    location: tokens.SourceLocation,
    expected: ?ValueType,
    found: ?ValueType,
};

/// Type environment tracking known tensor types
pub const TypeEnv = struct {
    allocator: std.mem.Allocator,
    /// Maps tensor names to their types
    tensor_types: std.StringHashMap(TensorType),
    /// Maps domain names to their sizes
    domain_sizes: std.StringHashMap(usize),
    /// Collected type errors
    errors: std.ArrayListUnmanaged(TypeError),

    pub fn init(allocator: std.mem.Allocator) TypeEnv {
        return .{
            .allocator = allocator,
            .tensor_types = std.StringHashMap(TensorType).init(allocator),
            .domain_sizes = std.StringHashMap(usize).init(allocator),
            .errors = std.ArrayListUnmanaged(TypeError){},
        };
    }

    pub fn deinit(self: *TypeEnv) void {
        self.tensor_types.deinit();
        self.domain_sizes.deinit();
        self.errors.deinit(self.allocator);
    }

    pub fn setTensorType(self: *TypeEnv, name: []const u8, typ: TensorType) !void {
        try self.tensor_types.put(name, typ);
    }

    pub fn getTensorType(self: *TypeEnv, name: []const u8) ?TensorType {
        return self.tensor_types.get(name);
    }

    pub fn setDomainSize(self: *TypeEnv, name: []const u8, size: usize) !void {
        try self.domain_sizes.put(name, size);
    }

    pub fn getDomainSize(self: *TypeEnv, name: []const u8) ?usize {
        return self.domain_sizes.get(name);
    }

    pub fn addError(self: *TypeEnv, err: TypeError) !void {
        try self.errors.append(self.allocator, err);
    }

    pub fn hasErrors(self: *TypeEnv) bool {
        return self.errors.items.len > 0;
    }
};

/// Type checker for Tensor Logic programs
pub const TypeChecker = struct {
    env: TypeEnv,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) TypeChecker {
        return .{
            .env = TypeEnv.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TypeChecker) void {
        self.env.deinit();
    }

    /// Check types for an entire program
    pub fn check(self: *TypeChecker, program: *const ast.Program) !void {
        // First pass: collect domain declarations and tensor declarations
        for (program.statements) |stmt| {
            switch (stmt) {
                .domain_decl => |d| {
                    if (d.size) |size| {
                        try self.env.setDomainSize(d.name, @intCast(size));
                    }
                },
                .sparse_decl => |s| {
                    // Sparse declarations define Boolean relations
                    try self.env.setTensorType(s.name, TensorType{
                        .value_type = if (s.is_boolean) .boolean else .real,
                        .shape = null,
                        .is_sparse = true,
                    });
                },
                else => {},
            }
        }

        // Second pass: infer types from equations
        for (program.statements) |stmt| {
            switch (stmt) {
                .equation => |eq| {
                    try self.checkEquation(&eq);
                },
                else => {},
            }
        }
    }

    /// Check types for a single equation
    fn checkEquation(self: *TypeChecker, eq: *const ast.Equation) !void {
        // Infer LHS type from syntax (parentheses = Boolean, brackets = Real)
        const lhs_type: ValueType = if (eq.lhs.is_boolean) .boolean else .real;

        // Infer RHS type from expression
        const rhs_type = try self.inferExprType(eq.rhs);

        // Check compatibility
        if (lhs_type == .boolean and rhs_type == .real) {
            // Warning: assigning real to boolean - will be truncated
            try self.env.addError(.{
                .message = "Assigning real value to Boolean relation - values will be truncated to 0/1",
                .location = eq.location,
                .expected = .boolean,
                .found = .real,
            });
        }

        // Register the tensor type
        try self.env.setTensorType(eq.lhs.name, TensorType{
            .value_type = lhs_type,
            .shape = null,
            .is_sparse = false,
        });
    }

    /// Infer the type of an expression
    fn inferExprType(self: *TypeChecker, expr: *const ast.Expr) !ValueType {
        return switch (expr.*) {
            .literal => |lit| switch (lit) {
                .integer => |v| if (v == 0 or v == 1) .boolean else .real,
                .float => .real,
                .boolean => .boolean,
                .string => .real, // Strings are not really supported in tensor ops
            },

            .tensor_ref => |ref| {
                // Check if we already know this tensor's type
                if (self.env.getTensorType(ref.name)) |known| {
                    return known.value_type;
                }
                // Infer from syntax: parentheses = Boolean
                return if (ref.is_boolean) .boolean else .real;
            },

            .binary => |bin| {
                const left_type = try self.inferExprType(bin.left);
                const right_type = try self.inferExprType(bin.right);

                return switch (bin.op) {
                    // Arithmetic ops produce real results
                    .add, .sub, .div, .pow => .real,
                    // Multiplication: Boolean * Boolean = Boolean (AND), otherwise Real
                    .mul => if (left_type == .boolean and right_type == .boolean) .boolean else .real,
                    // Comparison ops produce boolean results
                    .eq, .ne, .lt, .le, .gt, .ge => .boolean,
                    // Logical ops produce boolean results
                    .@"and", .@"or" => .boolean,
                };
            },

            .unary => |un| {
                _ = try self.inferExprType(un.operand);
                return switch (un.op) {
                    .negate => .real, // Negation produces real
                    .not => .boolean, // Logical not produces boolean
                };
            },

            .product => |prod| {
                // Product of all factors
                var all_boolean = true;
                for (prod.factors) |factor| {
                    const factor_type = try self.inferExprType(factor);
                    if (factor_type != .boolean) {
                        all_boolean = false;
                    }
                }
                return if (all_boolean) .boolean else .real;
            },

            .nonlinearity => |nl| {
                // Different nonlinearities have different output types
                return switch (nl.func) {
                    .step => .boolean, // step produces 0 or 1
                    // All other nonlinearities produce real values
                    .sigmoid, .relu, .tanh, .softmax, .exp, .log, .abs, .sqrt, .sin, .cos, .norm, .lnorm, .concat => .real,
                };
            },

            // Embed and conditional produce real by default
            .embed => .real,
            .conditional => |cond| {
                const then_type = try self.inferExprType(cond.then_branch);
                if (cond.else_branch) |else_branch| {
                    const else_type = try self.inferExprType(else_branch);
                    if (then_type == .boolean and else_type == .boolean) return .boolean;
                }
                return then_type;
            },
            .group => |g| try self.inferExprType(g),
        };
    }

    /// Get all type errors
    pub fn getErrors(self: *TypeChecker) []const TypeError {
        return self.errors.items;
    }

    /// Format type errors for display
    pub fn formatErrors(self: *TypeChecker, allocator: std.mem.Allocator) ![]u8 {
        var result = std.ArrayList(u8).init(allocator);
        const writer = result.writer();

        for (self.env.errors.items) |err| {
            try writer.print("{d}:{d}: type warning: {s}\n", .{
                err.location.line,
                err.location.column,
                err.message,
            });
            if (err.expected) |expected| {
                if (err.found) |found| {
                    try writer.print("  expected: {s}, found: {s}\n", .{
                        expected.format(),
                        found.format(),
                    });
                }
            }
        }

        return result.toOwnedSlice();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "type env basic" {
    const allocator = std.testing.allocator;
    var env = TypeEnv.init(allocator);
    defer env.deinit();

    try env.setTensorType("A", TensorType.real());
    try env.setTensorType("Parent", TensorType.boolean());
    try env.setDomainSize("Person", 100);

    try std.testing.expectEqual(ValueType.real, env.getTensorType("A").?.value_type);
    try std.testing.expectEqual(ValueType.boolean, env.getTensorType("Parent").?.value_type);
    try std.testing.expectEqual(@as(usize, 100), env.getDomainSize("Person").?);
}

test "infer boolean from syntax" {
    // Parent(x,y) should be boolean
    const allocator = std.testing.allocator;
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();

    // Simulate a boolean tensor ref
    var ref_expr = ast.Expr{
        .tensor_ref = .{
            .name = "Parent",
            .indices = &[_]ast.Index{},
            .is_boolean = true,
            .location = .{ .line = 1, .column = 1, .offset = 0 },
        },
    };

    const inferred = try checker.inferExprType(&ref_expr);
    try std.testing.expectEqual(ValueType.boolean, inferred);
}

test "infer real from syntax" {
    // W[i,j] should be real
    const allocator = std.testing.allocator;
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();

    var ref_expr = ast.Expr{
        .tensor_ref = .{
            .name = "W",
            .indices = &[_]ast.Index{},
            .is_boolean = false,
            .location = .{ .line = 1, .column = 1, .offset = 0 },
        },
    };

    const inferred = try checker.inferExprType(&ref_expr);
    try std.testing.expectEqual(ValueType.real, inferred);
}

test "step produces boolean" {
    const allocator = std.testing.allocator;
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();

    var inner = ast.Expr{
        .literal = .{ .float = 0.5 },
    };

    var step_expr = ast.Expr{
        .nonlinearity = .{
            .func = .step,
            .arg = &inner,
        },
    };

    const inferred = try checker.inferExprType(&step_expr);
    try std.testing.expectEqual(ValueType.boolean, inferred);
}

test "sigmoid produces real" {
    const allocator = std.testing.allocator;
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();

    var inner = ast.Expr{
        .literal = .{ .float = 0.5 },
    };

    var sigmoid_expr = ast.Expr{
        .nonlinearity = .{
            .func = .sigmoid,
            .arg = &inner,
        },
    };

    const inferred = try checker.inferExprType(&sigmoid_expr);
    try std.testing.expectEqual(ValueType.real, inferred);
}
