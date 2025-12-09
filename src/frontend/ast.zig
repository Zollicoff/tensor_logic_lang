// Tensor Logic Abstract Syntax Tree
// AST nodes representing the structure of Tensor Logic programs
//
// Based on the grammar from Pedro Domingos' "Tensor Logic: The Language of AI"
// Key insight: Everything is a tensor equation: LHS = [nonlinearity(] RHS [)]

const std = @import("std");
const tokens = @import("tokens.zig");
const SourceLocation = tokens.SourceLocation;

// ============================================================================
// Index Types
// ============================================================================

/// Represents different forms of tensor indices
/// Examples: i, j, i+1, i-1, p', i:10, i/2, ~i (virtual)
pub const Index = union(enum) {
    /// Simple named index: i, j, k
    name: []const u8,

    /// Constant integer index: 0, 1, 42
    constant: i64,

    /// Arithmetic index: i+1, j-2
    arithmetic: struct {
        base: []const u8,
        op: enum { add, sub },
        offset: i64,
    },

    /// Virtual index for embedding: ~i
    virtual: []const u8,

    /// Normalized index for softmax: /i (normalizes over i)
    normalize: []const u8,

    /// Primed index: p' (different dimension with same semantic meaning)
    primed: []const u8,

    /// Slice: 0:10 (range of values)
    slice: struct {
        start: i64,
        end: i64,
    },

    /// Division index: i/2 (used in pooling, strided operations)
    div: struct {
        index: []const u8,
        divisor: i64,
    },

    pub fn format(self: Index, allocator: std.mem.Allocator) ![]u8 {
        return switch (self) {
            .name => |n| std.fmt.allocPrint(allocator, "{s}", .{n}),
            .constant => |c| std.fmt.allocPrint(allocator, "{d}", .{c}),
            .arithmetic => |a| std.fmt.allocPrint(allocator, "{s}{s}{d}", .{
                a.base,
                if (a.op == .add) "+" else "-",
                a.offset,
            }),
            .virtual => |v| std.fmt.allocPrint(allocator, "~{s}", .{v}),
            .normalize => |n| std.fmt.allocPrint(allocator, "/{s}", .{n}),
            .primed => |p| std.fmt.allocPrint(allocator, "{s}'", .{p}),
            .slice => |s| std.fmt.allocPrint(allocator, "{d}:{d}", .{ s.start, s.end }),
            .div => |d| std.fmt.allocPrint(allocator, "{s}/{d}", .{ d.index, d.divisor }),
        };
    }
};

// ============================================================================
// Expression Types
// ============================================================================

/// Reference to a tensor with indices
/// Examples: A[i,j], Parent(x,y), W[i,j,k]
pub const TensorRef = struct {
    /// Name of the tensor/relation
    name: []const u8,

    /// List of indices
    indices: []Index,

    /// True if Boolean relation (uses parentheses), false if numeric tensor (uses brackets)
    is_boolean: bool,

    /// Source location for error reporting
    location: SourceLocation,
};

/// Literal values
pub const Literal = union(enum) {
    integer: i64,
    float: f64,
    boolean: bool,
    string: []const u8,
};

/// Unary operators (applied to single expression)
pub const UnaryOp = enum {
    negate, // -x
    not, // !x (logical negation)
};

/// Binary operators
pub const BinaryOp = enum {
    // Arithmetic
    add, // +
    sub, // -
    mul, // * (explicit multiplication)
    div, // /
    pow, // ^

    // Comparison
    eq, // ==
    ne, // !=
    lt, // <
    le, // <=
    gt, // >
    ge, // >=

    // Logical
    @"and", // & (explicit and)
    @"or", // | (explicit or)
};

/// Nonlinearity/activation functions
pub const Nonlinearity = enum {
    step, // H(x) = 1 if x > 0 else 0 (Heaviside)
    softmax, // exp(x_i) / sum_j exp(x_j)
    relu, // max(0, x)
    sigmoid, // 1 / (1 + exp(-x))
    tanh, // tanh(x)
    exp, // e^x
    log, // ln(x)
    abs, // |x|
    sqrt, // sqrt(x)
    sin, // sin(x)
    cos, // cos(x)
    norm, // ||x||
    lnorm, // layer normalization: (x - mean) / std
    concat, // concatenation along last axis
};

/// Accumulation operators (how to combine multiple equations for same LHS)
pub const AccumulationOp = enum {
    assign, // = (overwrite)
    add, // += (sum)
    mul, // *= (product)
    max, // max= (maximum)
    min, // min= (minimum)
    avg, // avg= (average)
};

/// Expression node - the core of tensor logic
pub const Expr = union(enum) {
    /// Tensor reference: A[i,j] or Parent(x,y)
    tensor_ref: TensorRef,

    /// Literal value: 42, 3.14, true, "hello"
    literal: Literal,

    /// Unary operation: -x, !x
    unary: struct {
        op: UnaryOp,
        operand: *Expr,
    },

    /// Binary operation: x + y, x * y
    binary: struct {
        op: BinaryOp,
        left: *Expr,
        right: *Expr,
    },

    /// Implicit multiplication (adjacent tensors): A[i,j] B[j,k]
    /// This is the key tensor logic operation - product followed by projection
    product: struct {
        factors: []*Expr,
    },

    /// Nonlinearity application: step(x), softmax(x), relu(x)
    nonlinearity: struct {
        func: Nonlinearity,
        arg: *Expr,
    },

    /// Embedding lookup: embed(E, x) where E is embedding matrix
    embed: struct {
        embedding: *Expr,
        index: *Expr,
    },

    /// Conditional: if cond then_expr else else_expr
    conditional: struct {
        condition: *Expr,
        then_branch: *Expr,
        else_branch: ?*Expr,
    },

    /// Grouped expression (parenthesized for precedence)
    group: *Expr,
};

// ============================================================================
// Statement Types
// ============================================================================

/// Tensor equation: LHS = RHS or LHS += RHS, etc.
/// This is THE fundamental construct of tensor logic
pub const Equation = struct {
    /// Left-hand side (what we're defining)
    lhs: TensorRef,

    /// Accumulation operator
    op: AccumulationOp,

    /// Right-hand side (the definition)
    rhs: *Expr,

    /// Source location for error reporting
    location: SourceLocation,
};

/// Domain declaration: defines the size/type of an index domain
/// Example: domain Person: 1000
pub const DomainDecl = struct {
    name: []const u8,
    size: ?i64, // None means inferred
    type_hint: ?[]const u8, // Optional type annotation
    location: SourceLocation,
};

/// Sparse tensor declaration
/// Example: sparse Parent(x: Person, y: Person)
pub const SparseDecl = struct {
    pub const IndexDecl = struct {
        name: []const u8,
        domain: ?[]const u8,
    };

    name: []const u8,
    indices: []IndexDecl,
    is_boolean: bool,
    location: SourceLocation,
};

/// Import statement
pub const Import = struct {
    path: []const u8,
    alias: ?[]const u8,
    location: SourceLocation,
};

/// Export statement
pub const Export = struct {
    name: []const u8,
    location: SourceLocation,
};

/// Query statement: Tensor? or Tensor[i,j]?
/// Queries the value of a tensor and prints it
pub const Query = struct {
    tensor: TensorRef,
    location: SourceLocation,
};

/// Save statement: save TensorName "filename"
pub const Save = struct {
    tensor_name: []const u8,
    path: []const u8,
    location: SourceLocation,
};

/// Load statement: load TensorName "filename"
pub const Load = struct {
    tensor_name: []const u8,
    path: []const u8,
    location: SourceLocation,
};

/// Top-level statement
pub const Statement = union(enum) {
    equation: Equation,
    domain_decl: DomainDecl,
    sparse_decl: SparseDecl,
    import_stmt: Import,
    export_stmt: Export,
    query: Query,
    save_stmt: Save,
    load_stmt: Load,
    comment: []const u8,
};

// ============================================================================
// Program
// ============================================================================

/// A complete Tensor Logic program
pub const Program = struct {
    statements: []Statement,
    allocator: std.mem.Allocator,
    /// Optional arena that owns all AST memory - if set, deinit frees it
    arena: ?*std.heap.ArenaAllocator = null,

    pub fn deinit(self: *Program) void {
        if (self.arena) |arena| {
            // Arena owns everything - free it all at once
            const backing = arena.child_allocator;
            arena.deinit();
            backing.destroy(arena);
        } else {
            // Just free the statements slice (caller manages AST memory)
            self.allocator.free(self.statements);
        }
    }
};

// ============================================================================
// AST Builder Helpers
// ============================================================================

pub const AstBuilder = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) AstBuilder {
        return .{ .allocator = allocator };
    }

    pub fn createTensorRef(
        self: *AstBuilder,
        name: []const u8,
        indices: []const Index,
        is_boolean: bool,
        location: SourceLocation,
    ) !TensorRef {
        const indices_copy = try self.allocator.alloc(Index, indices.len);
        @memcpy(indices_copy, indices);
        return TensorRef{
            .name = name,
            .indices = indices_copy,
            .is_boolean = is_boolean,
            .location = location,
        };
    }

    pub fn createExpr(self: *AstBuilder, expr: Expr) !*Expr {
        const ptr = try self.allocator.create(Expr);
        ptr.* = expr;
        return ptr;
    }

    pub fn createTensorRefExpr(self: *AstBuilder, ref: TensorRef) !*Expr {
        return self.createExpr(.{ .tensor_ref = ref });
    }

    pub fn createLiteralExpr(self: *AstBuilder, lit: Literal) !*Expr {
        return self.createExpr(.{ .literal = lit });
    }

    pub fn createBinaryExpr(self: *AstBuilder, op: BinaryOp, left: *Expr, right: *Expr) !*Expr {
        return self.createExpr(.{ .binary = .{ .op = op, .left = left, .right = right } });
    }

    pub fn createUnaryExpr(self: *AstBuilder, op: UnaryOp, operand: *Expr) !*Expr {
        return self.createExpr(.{ .unary = .{ .op = op, .operand = operand } });
    }

    pub fn createNonlinearityExpr(self: *AstBuilder, func: Nonlinearity, arg: *Expr) !*Expr {
        return self.createExpr(.{ .nonlinearity = .{ .func = func, .arg = arg } });
    }

    pub fn createProductExpr(self: *AstBuilder, factors: []const *Expr) !*Expr {
        const factors_copy = try self.allocator.alloc(*Expr, factors.len);
        @memcpy(factors_copy, factors);
        return self.createExpr(.{ .product = .{ .factors = factors_copy } });
    }

    pub fn createEquation(
        self: *AstBuilder,
        lhs: TensorRef,
        op: AccumulationOp,
        rhs: *Expr,
        location: SourceLocation,
    ) Equation {
        _ = self;
        return Equation{
            .lhs = lhs,
            .op = op,
            .rhs = rhs,
            .location = location,
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "create simple tensor ref" {
    const allocator = std.testing.allocator;
    var builder = AstBuilder.init(allocator);

    const indices = [_]Index{
        .{ .name = "i" },
        .{ .name = "j" },
    };

    const ref = try builder.createTensorRef(
        "A",
        &indices,
        false,
        .{ .line = 1, .column = 1, .offset = 0 },
    );
    defer allocator.free(ref.indices);

    try std.testing.expectEqualStrings("A", ref.name);
    try std.testing.expectEqual(@as(usize, 2), ref.indices.len);
    try std.testing.expectEqual(false, ref.is_boolean);
}

test "create boolean relation ref" {
    const allocator = std.testing.allocator;
    var builder = AstBuilder.init(allocator);

    const indices = [_]Index{
        .{ .name = "x" },
        .{ .name = "y" },
    };

    const ref = try builder.createTensorRef(
        "Parent",
        &indices,
        true,
        .{ .line = 1, .column = 1, .offset = 0 },
    );
    defer allocator.free(ref.indices);

    try std.testing.expectEqualStrings("Parent", ref.name);
    try std.testing.expectEqual(true, ref.is_boolean);
}

test "create literal expression" {
    const allocator = std.testing.allocator;
    var builder = AstBuilder.init(allocator);

    const expr = try builder.createLiteralExpr(.{ .integer = 42 });
    defer allocator.destroy(expr);

    try std.testing.expectEqual(Literal{ .integer = 42 }, expr.literal);
}

test "create binary expression" {
    const allocator = std.testing.allocator;
    var builder = AstBuilder.init(allocator);

    const left = try builder.createLiteralExpr(.{ .integer = 1 });
    defer allocator.destroy(left);
    const right = try builder.createLiteralExpr(.{ .integer = 2 });
    defer allocator.destroy(right);

    const expr = try builder.createBinaryExpr(.add, left, right);
    defer allocator.destroy(expr);

    try std.testing.expectEqual(BinaryOp.add, expr.binary.op);
}

test "create nonlinearity expression" {
    const allocator = std.testing.allocator;
    var builder = AstBuilder.init(allocator);

    const arg = try builder.createLiteralExpr(.{ .float = 0.5 });
    defer allocator.destroy(arg);

    const expr = try builder.createNonlinearityExpr(.step, arg);
    defer allocator.destroy(expr);

    try std.testing.expectEqual(Nonlinearity.step, expr.nonlinearity.func);
}

test "index formatting" {
    const allocator = std.testing.allocator;

    const idx1 = Index{ .name = "i" };
    const s1 = try idx1.format(allocator);
    defer allocator.free(s1);
    try std.testing.expectEqualStrings("i", s1);

    const idx2 = Index{ .primed = "p" };
    const s2 = try idx2.format(allocator);
    defer allocator.free(s2);
    try std.testing.expectEqualStrings("p'", s2);

    const idx3 = Index{ .arithmetic = .{ .base = "i", .op = .add, .offset = 1 } };
    const s3 = try idx3.format(allocator);
    defer allocator.free(s3);
    try std.testing.expectEqualStrings("i+1", s3);
}
