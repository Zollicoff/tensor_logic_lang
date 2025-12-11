// AST Optimizer for Tensor Logic
//
// Performs optimizations on the AST before interpretation:
// - Constant folding: evaluate constant expressions at compile time
// - Dead code elimination: remove unreachable statements
// - Strength reduction: replace expensive operations with cheaper ones
//

const std = @import("std");
const ast = @import("ast.zig");

/// Optimizer state
pub const Optimizer = struct {
    allocator: std.mem.Allocator,
    /// Statistics
    constants_folded: usize,
    dead_eliminated: usize,

    pub fn init(allocator: std.mem.Allocator) Optimizer {
        return .{
            .allocator = allocator,
            .constants_folded = 0,
            .dead_eliminated = 0,
        };
    }

    /// Optimize a program
    pub fn optimize(self: *Optimizer, program: *ast.Program) !void {
        for (program.statements) |*stmt| {
            try self.optimizeStatement(stmt);
        }
    }

    /// Optimize a single statement
    fn optimizeStatement(self: *Optimizer, stmt: *ast.Statement) !void {
        switch (stmt.*) {
            .equation => |*eq| {
                // Optimize the RHS expression
                eq.rhs = try self.optimizeExpr(eq.rhs);
            },
            else => {},
        }
    }

    /// Optimize an expression, returning potentially simplified version
    fn optimizeExpr(self: *Optimizer, expr: *ast.Expr) !*ast.Expr {
        switch (expr.*) {
            .binary => |bin| {
                // First optimize children
                const left = try self.optimizeExpr(bin.left);
                const right = try self.optimizeExpr(bin.right);

                // Constant folding: if both operands are literals, compute result
                if (left.* == .literal and right.* == .literal) {
                    if (self.foldBinaryLiteral(bin.op, left.literal, right.literal)) |result| {
                        self.constants_folded += 1;
                        const new_expr = try self.allocator.create(ast.Expr);
                        new_expr.* = .{ .literal = result };
                        return new_expr;
                    }
                }

                // Strength reduction: x * 1 = x, x + 0 = x, etc.
                if (right.* == .literal) {
                    if (self.identityReduction(bin.op, right.literal)) |_| {
                        // Return left operand directly
                        return left;
                    }
                }
                if (left.* == .literal) {
                    if (self.identityReductionLeft(bin.op, left.literal)) |_| {
                        return right;
                    }
                }

                // Zero reduction: x * 0 = 0
                if (right.* == .literal and self.isZero(right.literal)) {
                    if (bin.op == .mul) {
                        self.constants_folded += 1;
                        const new_expr = try self.allocator.create(ast.Expr);
                        new_expr.* = .{ .literal = .{ .integer = 0 } };
                        return new_expr;
                    }
                }
                if (left.* == .literal and self.isZero(left.literal)) {
                    if (bin.op == .mul) {
                        self.constants_folded += 1;
                        const new_expr = try self.allocator.create(ast.Expr);
                        new_expr.* = .{ .literal = .{ .integer = 0 } };
                        return new_expr;
                    }
                }

                // Update expression with optimized children
                expr.binary.left = left;
                expr.binary.right = right;
                return expr;
            },

            .unary => |un| {
                const operand = try self.optimizeExpr(un.operand);

                // Constant folding: -5 -> -5
                if (operand.* == .literal) {
                    if (self.foldUnaryLiteral(un.op, operand.literal)) |result| {
                        self.constants_folded += 1;
                        const new_expr = try self.allocator.create(ast.Expr);
                        new_expr.* = .{ .literal = result };
                        return new_expr;
                    }
                }

                // Double negation: --x = x
                if (un.op == .negate and operand.* == .unary and operand.unary.op == .negate) {
                    self.constants_folded += 1;
                    return operand.unary.operand;
                }

                expr.unary.operand = operand;
                return expr;
            },

            .nonlinearity => |nl| {
                const arg = try self.optimizeExpr(nl.arg);

                // Constant folding for known nonlinearities
                if (arg.* == .literal) {
                    if (self.foldNonlinearity(nl.func, arg.literal)) |result| {
                        self.constants_folded += 1;
                        const new_expr = try self.allocator.create(ast.Expr);
                        new_expr.* = .{ .literal = result };
                        return new_expr;
                    }
                }

                expr.nonlinearity.arg = arg;
                return expr;
            },

            .product => |prod| {
                // Optimize each factor
                for (prod.factors) |*factor| {
                    factor.* = try self.optimizeExpr(factor.*);
                }
                return expr;
            },

            .group => |inner| {
                // Optimize the inner expression
                const optimized_inner = try self.optimizeExpr(inner);

                // If inner is a literal, unwrap the group
                if (optimized_inner.* == .literal) {
                    return optimized_inner;
                }

                expr.* = .{ .group = optimized_inner };
                return expr;
            },

            .conditional => |cond| {
                const condition = try self.optimizeExpr(cond.condition);
                const then_branch = try self.optimizeExpr(cond.then_branch);
                const else_branch = if (cond.else_branch) |eb| try self.optimizeExpr(eb) else null;

                // If condition is a known constant, fold the conditional
                if (condition.* == .literal) {
                    if (self.isTruthy(condition.literal)) {
                        self.constants_folded += 1;
                        return then_branch;
                    } else if (else_branch) |eb| {
                        self.constants_folded += 1;
                        return eb;
                    }
                }

                expr.conditional.condition = condition;
                expr.conditional.then_branch = then_branch;
                expr.conditional.else_branch = else_branch;
                return expr;
            },

            // These don't need optimization
            .tensor_ref, .literal, .embed => return expr,
        }
    }

    /// Try to fold a binary operation on two literals
    fn foldBinaryLiteral(self: *Optimizer, op: ast.BinaryOp, left: ast.Literal, right: ast.Literal) ?ast.Literal {
        _ = self;

        // Extract numeric values
        const l: f64 = switch (left) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            .boolean => |b| if (b) 1.0 else 0.0,
            .string => return null,
        };
        const r: f64 = switch (right) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            .boolean => |b| if (b) 1.0 else 0.0,
            .string => return null,
        };

        const result: f64 = switch (op) {
            .add => l + r,
            .sub => l - r,
            .mul => l * r,
            .div => if (r != 0) l / r else return null,
            .pow => std.math.pow(f64, l, r),
            else => return null,
        };

        // Return as float or int depending on original types
        if (left == .integer and right == .integer and op != .div) {
            return .{ .integer = @intFromFloat(result) };
        }
        return .{ .float = result };
    }

    /// Try to fold a unary operation on a literal
    fn foldUnaryLiteral(self: *Optimizer, op: ast.UnaryOp, operand: ast.Literal) ?ast.Literal {
        _ = self;

        switch (op) {
            .negate => {
                return switch (operand) {
                    .integer => |i| .{ .integer = -i },
                    .float => |f| .{ .float = -f },
                    else => null,
                };
            },
            .not => {
                return switch (operand) {
                    .boolean => |b| .{ .boolean = !b },
                    else => null,
                };
            },
        }
    }

    /// Try to fold a nonlinearity on a literal
    fn foldNonlinearity(self: *Optimizer, func: ast.Nonlinearity, arg: ast.Literal) ?ast.Literal {
        _ = self;

        const x: f64 = switch (arg) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => return null,
        };

        const result: f64 = switch (func) {
            .step => if (x > 0) 1.0 else 0.0,
            .relu => @max(0.0, x),
            .sigmoid => 1.0 / (1.0 + @exp(-x)),
            .tanh => std.math.tanh(x),
            .exp => @exp(x),
            .log => if (x > 0) @log(x) else return null,
            .abs => @abs(x),
            .sqrt => if (x >= 0) @sqrt(x) else return null,
            .sin => @sin(x),
            .cos => @cos(x),
            .softmax, .norm, .lnorm, .concat => return null, // Can't fold these for scalars
        };

        return .{ .float = result };
    }

    /// Check if literal is an identity element for the operation on the right
    fn identityReduction(self: *Optimizer, op: ast.BinaryOp, right: ast.Literal) ?void {
        _ = self;
        const val: f64 = switch (right) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => return null,
        };

        // x + 0 = x, x - 0 = x
        if ((op == .add or op == .sub) and val == 0) return {};
        // x * 1 = x, x / 1 = x
        if ((op == .mul or op == .div) and val == 1) return {};
        // x ^ 1 = x
        if (op == .pow and val == 1) return {};

        return null;
    }

    /// Check if literal is an identity element on the left
    fn identityReductionLeft(self: *Optimizer, op: ast.BinaryOp, left: ast.Literal) ?void {
        _ = self;
        const val: f64 = switch (left) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => return null,
        };

        // 0 + x = x
        if (op == .add and val == 0) return {};
        // 1 * x = x
        if (op == .mul and val == 1) return {};

        return null;
    }

    /// Check if literal is zero
    fn isZero(self: *Optimizer, lit: ast.Literal) bool {
        _ = self;
        return switch (lit) {
            .integer => |i| i == 0,
            .float => |f| f == 0,
            else => false,
        };
    }

    /// Check if literal is truthy
    fn isTruthy(self: *Optimizer, lit: ast.Literal) bool {
        _ = self;
        return switch (lit) {
            .boolean => |b| b,
            .integer => |i| i != 0,
            .float => |f| f != 0,
            .string => |s| s.len > 0,
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "constant folding arithmetic" {
    const allocator = std.testing.allocator;
    var builder = ast.AstBuilder.init(allocator);

    // Create 2 + 3
    const left = try builder.createLiteralExpr(.{ .integer = 2 });
    defer allocator.destroy(left);
    const right = try builder.createLiteralExpr(.{ .integer = 3 });
    defer allocator.destroy(right);

    const bin = try builder.createBinaryExpr(.add, left, right);
    defer allocator.destroy(bin);

    var opt = Optimizer.init(allocator);
    const result = try opt.optimizeExpr(bin);

    // Should fold to 5
    try std.testing.expect(result.* == .literal);
    try std.testing.expectEqual(@as(i64, 5), result.literal.integer);
    try std.testing.expectEqual(@as(usize, 1), opt.constants_folded);

    allocator.destroy(result);
}

test "identity reduction" {
    const allocator = std.testing.allocator;
    var builder = ast.AstBuilder.init(allocator);

    // Create x * 1 (represented as literal(42) * 1)
    const x = try builder.createLiteralExpr(.{ .integer = 42 });
    defer allocator.destroy(x);
    const one = try builder.createLiteralExpr(.{ .integer = 1 });
    defer allocator.destroy(one);

    const mul = try builder.createBinaryExpr(.mul, x, one);
    defer allocator.destroy(mul);

    var opt = Optimizer.init(allocator);
    const result = try opt.optimizeExpr(mul);

    // Should return just 42 (the identity was folded first since both are constants)
    try std.testing.expect(result.* == .literal);
    try std.testing.expectEqual(@as(i64, 42), result.literal.integer);

    allocator.destroy(result);
}

test "nonlinearity folding" {
    const allocator = std.testing.allocator;
    var builder = ast.AstBuilder.init(allocator);

    // Create relu(-5)
    const arg = try builder.createLiteralExpr(.{ .integer = -5 });
    defer allocator.destroy(arg);

    const relu_expr = try builder.createNonlinearityExpr(.relu, arg);
    defer allocator.destroy(relu_expr);

    var opt = Optimizer.init(allocator);
    const result = try opt.optimizeExpr(relu_expr);

    // relu(-5) = 0
    try std.testing.expect(result.* == .literal);
    try std.testing.expectEqual(@as(f64, 0.0), result.literal.float);

    allocator.destroy(result);
}
