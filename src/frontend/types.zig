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

/// Tracks inferred domain for an index symbol
/// Used to validate that the same index symbol is used consistently across all tensors
pub const IndexConstraint = struct {
    /// Index symbol name (e.g., "i", "x")
    name: []const u8,
    /// Domain inferred from tensor position (e.g., "Person" from sparse Parent[Person, Person])
    inferred_domain: ?[]const u8,
    /// Size of that domain (e.g., 10 from domain Person: 10)
    inferred_size: ?usize,
    /// Source location where first used (for error messages)
    first_used_at: tokens.SourceLocation,
    /// Tensor where first seen (for error messages)
    first_tensor: []const u8,
    /// Which dimension in that tensor (0-indexed)
    first_dimension: usize,
};

/// Type information for a tensor
pub const TensorType = struct {
    value_type: ValueType,
    shape: ?Shape,
    is_sparse: bool,
    rank: ?usize = null,
    first_defined_at: ?tokens.SourceLocation = null,

    pub fn boolean() TensorType {
        return .{ .value_type = .boolean, .shape = null, .is_sparse = false };
    }

    pub fn real() TensorType {
        return .{ .value_type = .real, .shape = null, .is_sparse = false };
    }

    pub fn unknown() TensorType {
        return .{ .value_type = .unknown, .shape = null, .is_sparse = false };
    }

    pub fn withRank(value_type: ValueType, rank: usize, location: tokens.SourceLocation) TensorType {
        return .{
            .value_type = value_type,
            .shape = null,
            .is_sparse = false,
            .rank = rank,
            .first_defined_at = location,
        };
    }
};

/// Error severity levels
pub const Severity = enum {
    @"error", // Blocks compilation
    warning, // Allowed but suspicious
    hint, // Informational

    pub fn format(self: Severity) []const u8 {
        return switch (self) {
            .@"error" => "error",
            .warning => "warning",
            .hint => "note",
        };
    }
};

/// Type error information
pub const TypeError = struct {
    message: []const u8,
    location: tokens.SourceLocation,
    expected: ?ValueType = null,
    found: ?ValueType = null,
    // Severity level
    severity: Severity = .@"error",
    // For rank mismatches
    expected_rank: ?usize = null,
    found_rank: ?usize = null,
    tensor_name: ?[]const u8 = null,
    // For index errors
    index_name: ?[]const u8 = null,
    // Cross-reference to original definition
    original_location: ?tokens.SourceLocation = null,
    // For domain/size errors (M1.1)
    expected_domain: ?[]const u8 = null,
    found_domain: ?[]const u8 = null,
    expected_size: ?usize = null,
    found_size: ?usize = null,
};

/// Type environment tracking known tensor types
pub const TypeEnv = struct {
    allocator: std.mem.Allocator,
    /// Maps tensor names to their types
    tensor_types: std.StringHashMap(TensorType),
    /// Maps domain names to their sizes
    domain_sizes: std.StringHashMap(usize),
    /// Tracks index symbol constraints (M1.1 - validates consistent domain usage)
    index_constraints: std.StringHashMap(IndexConstraint),
    /// Collected type errors
    errors: std.ArrayListUnmanaged(TypeError),

    pub fn init(allocator: std.mem.Allocator) TypeEnv {
        return .{
            .allocator = allocator,
            .tensor_types = std.StringHashMap(TensorType).init(allocator),
            .domain_sizes = std.StringHashMap(usize).init(allocator),
            .index_constraints = std.StringHashMap(IndexConstraint).init(allocator),
            .errors = std.ArrayListUnmanaged(TypeError){},
        };
    }

    pub fn deinit(self: *TypeEnv) void {
        self.tensor_types.deinit();
        self.domain_sizes.deinit();
        self.index_constraints.deinit();
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

    /// Get an existing index constraint
    pub fn getIndexConstraint(self: *TypeEnv, name: []const u8) ?IndexConstraint {
        return self.index_constraints.get(name);
    }

    /// Record use of an index symbol and validate consistency
    /// If the same index was used before with a different size, emit an error
    /// If the same index was used with a different domain name (but same size), emit a warning
    pub fn recordIndexUse(
        self: *TypeEnv,
        index_name: []const u8,
        domain: ?[]const u8,
        size: ?usize,
        tensor_name: []const u8,
        dimension: usize,
        location: tokens.SourceLocation,
    ) !void {
        if (self.index_constraints.getPtr(index_name)) |existing| {
            // Index already seen - validate consistency
            if (existing.inferred_size) |existing_size| {
                if (size) |new_size| {
                    if (existing_size != new_size) {
                        // Inconsistent use - same index symbol used with different sizes
                        try self.addError(.{
                            .message = "index used with inconsistent domain sizes",
                            .location = location,
                            .severity = .@"error",
                            .index_name = index_name,
                            .tensor_name = tensor_name,
                            .expected_size = existing_size,
                            .found_size = new_size,
                            .expected_domain = existing.inferred_domain,
                            .found_domain = domain,
                            .original_location = existing.first_used_at,
                        });
                    } else {
                        // Sizes match - but check if domain names differ
                        const domains_differ = blk: {
                            if (existing.inferred_domain) |existing_dom| {
                                if (domain) |new_dom| {
                                    break :blk !std.mem.eql(u8, existing_dom, new_dom);
                                }
                            }
                            break :blk false;
                        };
                        if (domains_differ) {
                            try self.addError(.{
                                .message = "index used with different domains (sizes match but semantics may differ)",
                                .location = location,
                                .severity = .warning,
                                .index_name = index_name,
                                .tensor_name = tensor_name,
                                .expected_size = existing_size,
                                .found_size = new_size,
                                .expected_domain = existing.inferred_domain,
                                .found_domain = domain,
                                .original_location = existing.first_used_at,
                            });
                        }
                    }
                }
            } else if (size != null) {
                // Existing had no size but we now have one - update the constraint
                existing.inferred_size = size;
                existing.inferred_domain = domain;
                existing.first_used_at = location;
                existing.first_tensor = tensor_name;
                existing.first_dimension = dimension;
            }
        } else {
            // First use of this index - record the constraint
            try self.index_constraints.put(index_name, .{
                .name = index_name,
                .inferred_domain = domain,
                .inferred_size = size,
                .first_used_at = location,
                .first_tensor = tensor_name,
                .first_dimension = dimension,
            });
        }
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
                    // Sparse declarations define Boolean relations with explicit domain info
                    // Build shape from declared domains
                    var domains = try self.allocator.alloc([]const u8, s.indices.len);
                    var sizes = try self.allocator.alloc(?usize, s.indices.len);

                    for (s.indices, 0..) |idx_decl, i| {
                        // Use declared domain if present, otherwise use index name
                        const domain_name = idx_decl.domain orelse idx_decl.name;
                        domains[i] = domain_name;
                        sizes[i] = self.env.getDomainSize(domain_name);
                    }

                    const shape = Shape{
                        .domains = domains,
                        .sizes = sizes,
                    };

                    try self.env.setTensorType(s.name, TensorType{
                        .value_type = if (s.is_boolean) .boolean else .real,
                        .shape = shape,
                        .is_sparse = true,
                        .rank = s.indices.len,
                        .first_defined_at = s.location,
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
                .severity = .warning,
            });
        }

        // Check rank consistency and register tensor type
        try self.checkTensorRank(&eq.lhs);

        // Check indices in RHS expression
        try self.checkExprRanks(eq.rhs);
    }

    /// Extract simple name from Index union (returns null for constants/slices)
    fn getIndexName(idx: ast.Index) ?[]const u8 {
        return switch (idx) {
            .name => |n| n,
            .arithmetic => |a| a.base,
            .virtual => |v| v,
            .normalize => |n| n,
            .primed => |p| p,
            .div => |d| d.index,
            .constant, .slice => null,
        };
    }

    /// Check tensor reference for consistent rank and record index uses
    fn checkTensorRank(self: *TypeChecker, ref: *const ast.TensorRef) !void {
        const ref_rank = ref.indices.len;
        const value_type: ValueType = if (ref.is_boolean) .boolean else .real;

        // Get existing tensor type (may have shape from sparse_decl)
        const existing_type = self.env.getTensorType(ref.name);

        if (existing_type) |existing| {
            // Tensor seen before - check rank consistency
            if (existing.rank) |expected_rank| {
                if (ref_rank != expected_rank) {
                    try self.env.addError(.{
                        .message = "tensor rank mismatch",
                        .location = ref.location,
                        .tensor_name = ref.name,
                        .expected_rank = expected_rank,
                        .found_rank = ref_rank,
                        .original_location = existing.first_defined_at,
                        .severity = .@"error",
                    });
                }
            } else {
                // Existing type has no rank info - update it
                try self.env.setTensorType(ref.name, TensorType{
                    .value_type = existing.value_type,
                    .shape = existing.shape,
                    .is_sparse = existing.is_sparse,
                    .rank = ref_rank,
                    .first_defined_at = ref.location,
                });
            }
        } else {
            // First use - record the type with rank
            try self.env.setTensorType(ref.name, TensorType.withRank(
                value_type,
                ref_rank,
                ref.location,
            ));
        }

        // Check individual indices for validity and record index uses
        for (ref.indices, 0..) |idx, dim| {
            try self.checkArithmeticIndex(idx, ref.location);

            // Record index use for domain tracking (M1.1)
            if (idx.getBaseName()) |index_name| {
                // Try to get domain info from tensor's shape (if available)
                var domain: ?[]const u8 = null;
                var size: ?usize = null;

                if (existing_type) |existing| {
                    if (existing.shape) |shape| {
                        if (dim < shape.domains.len) {
                            domain = shape.domains[dim];
                            size = shape.sizes[dim];
                        }
                    }
                }

                // If no size from shape, try fallback to domain lookup by index name
                // This is "name-based coupling" - emit a warning to make it visible
                if (size == null) {
                    if (self.env.getDomainSize(index_name)) |fallback_size| {
                        size = fallback_size;
                        domain = index_name; // Mark that domain came from name matching
                        try self.env.addError(.{
                            .message = "index domain inferred from name (not tensor position)",
                            .location = ref.location,
                            .severity = .warning,
                            .index_name = index_name,
                            .tensor_name = ref.name,
                            .expected_size = fallback_size,
                        });
                    }
                }

                try self.env.recordIndexUse(
                    index_name,
                    domain,
                    size,
                    ref.name,
                    dim,
                    ref.location,
                );
            }
        }
    }

    /// Validate arithmetic index expressions are well-defined
    fn checkArithmeticIndex(self: *TypeChecker, idx: ast.Index, location: tokens.SourceLocation) !void {
        switch (idx) {
            .div => |d| {
                if (d.divisor <= 0) {
                    try self.env.addError(.{
                        .message = "index division by non-positive number",
                        .location = location,
                        .index_name = d.index,
                        .severity = .@"error",
                    });
                }
            },
            else => {},
        }
    }

    /// Check ranks recursively in an expression
    fn checkExprRanks(self: *TypeChecker, expr: *const ast.Expr) !void {
        switch (expr.*) {
            .tensor_ref => |ref| {
                try self.checkTensorRank(&ref);
            },
            .binary => |bin| {
                try self.checkExprRanks(bin.left);
                try self.checkExprRanks(bin.right);
            },
            .unary => |un| {
                try self.checkExprRanks(un.operand);
            },
            .product => |prod| {
                for (prod.factors) |factor| {
                    try self.checkExprRanks(factor);
                }
            },
            .nonlinearity => |nl| {
                try self.checkExprRanks(nl.arg);
            },
            .conditional => |cond| {
                try self.checkExprRanks(cond.condition);
                try self.checkExprRanks(cond.then_branch);
                if (cond.else_branch) |else_br| {
                    try self.checkExprRanks(else_br);
                }
            },
            .group => |g| try self.checkExprRanks(g),
            .embed => |e| {
                try self.checkExprRanks(e.embedding);
                try self.checkExprRanks(e.index);
            },
            .literal => {},
        }
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

    /// Count errors by severity
    pub fn countBySeverity(self: *TypeChecker) struct { errors: usize, warnings: usize } {
        var errors: usize = 0;
        var warnings: usize = 0;
        for (self.env.errors.items) |err| {
            switch (err.severity) {
                .@"error" => errors += 1,
                .warning => warnings += 1,
                .hint => {},
            }
        }
        return .{ .errors = errors, .warnings = warnings };
    }

    /// Format type errors for display
    pub fn formatErrors(self: *TypeChecker, allocator: std.mem.Allocator) ![]u8 {
        var result = std.ArrayListUnmanaged(u8){};
        defer result.deinit(allocator);
        const writer = result.writer(allocator);

        for (self.env.errors.items) |err| {
            // Severity prefix
            try writer.print("{d}:{d}: {s}: {s}\n", .{
                err.location.line,
                err.location.column,
                err.severity.format(),
                err.message,
            });

            // Tensor name context
            if (err.tensor_name) |name| {
                try writer.print("  tensor: {s}\n", .{name});
            }

            // Index name context
            if (err.index_name) |name| {
                try writer.print("  index: {s}\n", .{name});
            }

            // Rank mismatch details
            if (err.expected_rank) |expected| {
                if (err.found_rank) |found| {
                    try writer.print("  expected rank: {d}, found: {d}\n", .{ expected, found });
                }
            }

            // Value type mismatch details
            if (err.expected) |expected| {
                if (err.found) |found| {
                    try writer.print("  expected: {s}, found: {s}\n", .{
                        expected.format(),
                        found.format(),
                    });
                }
            }

            // Domain size mismatch details (M1.1)
            if (err.expected_size) |expected| {
                if (err.found_size) |found| {
                    try writer.print("  expected size: {d}, found: {d}\n", .{ expected, found });
                }
            }

            // Domain name mismatch details (M1.1)
            if (err.expected_domain) |expected| {
                if (err.found_domain) |found| {
                    try writer.print("  expected domain: {s}, found: {s}\n", .{ expected, found });
                } else {
                    try writer.print("  expected domain: {s}\n", .{expected});
                }
            }

            // Cross-reference to original location
            if (err.original_location) |orig| {
                try writer.print("  first defined at {d}:{d}\n", .{ orig.line, orig.column });
            }
        }

        return result.toOwnedSlice(allocator);
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

test "rank consistency - same tensor same rank" {
    const allocator = std.testing.allocator;
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();

    // First use at rank 2
    var indices1 = [_]ast.Index{ .{ .name = "i" }, .{ .name = "j" } };
    const ref1 = ast.TensorRef{
        .name = "A",
        .indices = &indices1,
        .is_boolean = false,
        .location = .{ .line = 1, .column = 1, .offset = 0 },
    };
    try checker.checkTensorRank(&ref1);

    // Second use also at rank 2 - should be fine
    var indices2 = [_]ast.Index{ .{ .name = "x" }, .{ .name = "y" } };
    const ref2 = ast.TensorRef{
        .name = "A",
        .indices = &indices2,
        .is_boolean = false,
        .location = .{ .line = 2, .column = 1, .offset = 10 },
    };
    try checker.checkTensorRank(&ref2);

    // No errors expected
    try std.testing.expect(!checker.env.hasErrors());
}

test "rank consistency - same tensor different ranks" {
    const allocator = std.testing.allocator;
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();

    // First use at rank 2
    var indices1 = [_]ast.Index{ .{ .name = "i" }, .{ .name = "j" } };
    const ref1 = ast.TensorRef{
        .name = "A",
        .indices = &indices1,
        .is_boolean = false,
        .location = .{ .line = 1, .column = 1, .offset = 0 },
    };
    try checker.checkTensorRank(&ref1);

    // Second use at rank 3 - should produce error
    var indices2 = [_]ast.Index{ .{ .name = "i" }, .{ .name = "j" }, .{ .name = "k" } };
    const ref2 = ast.TensorRef{
        .name = "A",
        .indices = &indices2,
        .is_boolean = false,
        .location = .{ .line = 2, .column = 1, .offset = 10 },
    };
    try checker.checkTensorRank(&ref2);

    // Should have error
    try std.testing.expect(checker.env.hasErrors());
    try std.testing.expectEqual(@as(usize, 1), checker.env.errors.items.len);
    const err = checker.env.errors.items[0];
    try std.testing.expectEqual(@as(?usize, 2), err.expected_rank);
    try std.testing.expectEqual(@as(?usize, 3), err.found_rank);
    try std.testing.expectEqual(Severity.@"error", err.severity);
}

test "arithmetic index - division by zero" {
    const allocator = std.testing.allocator;
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();

    // Division by zero should produce error
    const div_idx = ast.Index{ .div = .{ .index = "i", .divisor = 0 } };
    try checker.checkArithmeticIndex(div_idx, .{ .line = 1, .column = 1, .offset = 0 });

    try std.testing.expect(checker.env.hasErrors());
    try std.testing.expectEqual(Severity.@"error", checker.env.errors.items[0].severity);
}

test "arithmetic index - valid division" {
    const allocator = std.testing.allocator;
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();

    // Division by positive number should be fine
    const div_idx = ast.Index{ .div = .{ .index = "i", .divisor = 2 } };
    try checker.checkArithmeticIndex(div_idx, .{ .line = 1, .column = 1, .offset = 0 });

    try std.testing.expect(!checker.env.hasErrors());
}

test "error count by severity" {
    const allocator = std.testing.allocator;
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();

    // Add one error
    try checker.env.addError(.{
        .message = "test error",
        .location = .{ .line = 1, .column = 1, .offset = 0 },
        .severity = .@"error",
    });

    // Add one warning
    try checker.env.addError(.{
        .message = "test warning",
        .location = .{ .line = 2, .column = 1, .offset = 10 },
        .severity = .warning,
    });

    const counts = checker.countBySeverity();
    try std.testing.expectEqual(@as(usize, 1), counts.errors);
    try std.testing.expectEqual(@as(usize, 1), counts.warnings);
}
