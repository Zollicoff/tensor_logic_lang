// Tensor Logic Parser
// Parses token stream into AST
//
// Grammar (simplified EBNF):
//   program     = { statement }
//   statement   = equation | domain_decl | sparse_decl | import | export | comment
//   equation    = tensor_ref (= | += | max= | min= | avg=) expr NEWLINE
//   expr        = term { term }                    -- implicit multiplication
//   term        = factor { (+ | -) factor }
//   factor      = unary { (* | / | ^) unary }
//   unary       = (- | !)? primary
//   primary     = tensor_ref | literal | nonlinearity "(" expr ")" | "(" expr ")"
//   tensor_ref  = IDENTIFIER ( "[" indices "]" | "(" indices ")" )
//   indices     = index { "," index }
//   index       = IDENTIFIER ["'"] | INTEGER | IDENTIFIER (+ | -) INTEGER

const std = @import("std");
const tokens = @import("tokens.zig");
const ast = @import("ast.zig");
const Token = tokens.Token;
const TokenType = tokens.TokenType;
const SourceLocation = tokens.SourceLocation;

pub const ParseError = error{
    UnexpectedToken,
    ExpectedExpression,
    ExpectedIdentifier,
    ExpectedEquals,
    ExpectedClosingBracket,
    ExpectedClosingParen,
    ExpectedComma,
    InvalidIndex,
    OutOfMemory,
};

pub const Parser = struct {
    tokens_list: []const Token,
    current: usize,
    allocator: std.mem.Allocator,
    builder: ast.AstBuilder,
    errors: std.ArrayListUnmanaged(ParserError),

    pub const ParserError = struct {
        message: []const u8,
        location: SourceLocation,

        pub fn format(self: ParserError, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.print("Parse error at {d}:{d}: {s}", .{ self.location.line, self.location.column, self.message });
        }
    };

    pub fn init(allocator: std.mem.Allocator, tokens_list: []const Token) Parser {
        return Parser{
            .tokens_list = tokens_list,
            .current = 0,
            .allocator = allocator,
            .builder = ast.AstBuilder.init(allocator),
            .errors = .{},
        };
    }

    pub fn deinit(self: *Parser) void {
        self.errors.deinit(self.allocator);
    }

    /// Get the last recorded error (if any)
    pub fn getLastError(self: *Parser) ?ParserError {
        if (self.errors.items.len > 0) {
            return self.errors.items[self.errors.items.len - 1];
        }
        return null;
    }

    /// Get all recorded errors
    pub fn getErrors(self: *Parser) []const ParserError {
        return self.errors.items;
    }

    /// Record an error with current token location
    fn recordError(self: *Parser, message: []const u8) void {
        const loc = if (self.current < self.tokens_list.len)
            self.tokens_list[self.current].location
        else if (self.tokens_list.len > 0)
            self.tokens_list[self.tokens_list.len - 1].location
        else
            SourceLocation{ .line = 1, .column = 1, .offset = 0 };

        self.errors.append(self.allocator, ParserError{
            .message = message,
            .location = loc,
        }) catch {};
    }

    pub fn parse(self: *Parser) ParseError!ast.Program {
        var statements = std.ArrayListUnmanaged(ast.Statement){};

        while (!self.isAtEnd()) {
            // Skip newlines between statements
            while (self.check(.newline) or self.check(.comment)) {
                if (self.check(.comment)) {
                    const tok = self.advance();
                    statements.append(self.allocator, .{ .comment = tok.lexeme }) catch return ParseError.OutOfMemory;
                } else {
                    _ = self.advance();
                }
            }

            if (self.isAtEnd()) break;

            const stmt = try self.parseStatement();
            statements.append(self.allocator, stmt) catch return ParseError.OutOfMemory;
        }

        return ast.Program{
            .statements = statements.toOwnedSlice(self.allocator) catch return ParseError.OutOfMemory,
            .allocator = self.allocator,
        };
    }

    fn parseStatement(self: *Parser) ParseError!ast.Statement {
        // Check for keywords first
        if (self.check(.kw_domain)) {
            return self.parseDomainDecl();
        }
        if (self.check(.kw_sparse)) {
            return self.parseSparseDecl();
        }
        if (self.check(.kw_import)) {
            return self.parseImport();
        }
        if (self.check(.kw_export)) {
            return self.parseExport();
        }
        if (self.check(.kw_save)) {
            return self.parseSave();
        }
        if (self.check(.kw_load)) {
            return self.parseLoad();
        }
        if (self.check(.kw_backward)) {
            return self.parseBackward();
        }

        // Otherwise, it should be an equation
        return self.parseEquation();
    }

    fn parseSave(self: *Parser) ParseError!ast.Statement {
        const location = self.peek().location;
        _ = self.advance(); // consume 'save'

        // Expect tensor name
        if (!self.check(.identifier)) {
            self.recordError("expected tensor name after 'save'");
            return ParseError.ExpectedIdentifier;
        }
        const name = self.advance().lexeme;

        // Expect string path
        if (!self.check(.string)) {
            self.recordError("expected file path string after tensor name");
            return ParseError.UnexpectedToken;
        }
        const path = self.advance().lexeme;

        // Consume newline
        if (self.check(.newline)) _ = self.advance();

        return ast.Statement{
            .save_stmt = ast.Save{
                .tensor_name = name,
                .path = path,
                .location = location,
            },
        };
    }

    fn parseLoad(self: *Parser) ParseError!ast.Statement {
        const location = self.peek().location;
        _ = self.advance(); // consume 'load'

        // Expect tensor name
        if (!self.check(.identifier)) {
            self.recordError("expected tensor name after 'load'");
            return ParseError.ExpectedIdentifier;
        }
        const name = self.advance().lexeme;

        // Expect string path
        if (!self.check(.string)) {
            self.recordError("expected file path string after tensor name");
            return ParseError.UnexpectedToken;
        }
        const path = self.advance().lexeme;

        // Consume newline
        if (self.check(.newline)) _ = self.advance();

        return ast.Statement{
            .load_stmt = ast.Load{
                .tensor_name = name,
                .path = path,
                .location = location,
            },
        };
    }

    fn parseBackward(self: *Parser) ParseError!ast.Statement {
        const location = self.peek().location;
        _ = self.advance(); // consume 'backward'

        // Expect loss tensor name
        if (!self.check(.identifier)) {
            self.recordError("expected loss tensor name after 'backward'");
            return ParseError.ExpectedIdentifier;
        }
        const loss = self.advance().lexeme;

        // Expect 'wrt' keyword
        if (!self.check(.kw_wrt)) {
            self.recordError("expected 'wrt' after loss tensor");
            return ParseError.UnexpectedToken;
        }
        _ = self.advance(); // consume 'wrt'

        // Parse parameter list (comma-separated identifiers)
        var params = std.ArrayListUnmanaged([]const u8){};

        if (!self.check(.identifier)) {
            self.recordError("expected parameter name after 'wrt'");
            return ParseError.ExpectedIdentifier;
        }
        params.append(self.allocator, self.advance().lexeme) catch return ParseError.OutOfMemory;

        // Parse additional parameters
        while (self.match(.comma)) {
            if (!self.check(.identifier)) {
                self.recordError("expected parameter name after ','");
                return ParseError.ExpectedIdentifier;
            }
            params.append(self.allocator, self.advance().lexeme) catch return ParseError.OutOfMemory;
        }

        // Consume newline
        if (self.check(.newline)) _ = self.advance();

        return ast.Statement{
            .backward_stmt = ast.Backward{
                .loss = loss,
                .params = params.toOwnedSlice(self.allocator) catch return ParseError.OutOfMemory,
                .location = location,
            },
        };
    }

    fn parseEquation(self: *Parser) ParseError!ast.Statement {
        const location = self.peek().location;

        // Parse LHS (must be a tensor reference)
        const lhs = try self.parseTensorRef();

        // Check for query: Tensor? or Tensor[i,j]?
        if (self.match(.question)) {
            // Consume trailing newline/comments
            while (self.check(.comment)) {
                _ = self.advance();
            }
            if (self.check(.newline)) {
                _ = self.advance();
            }
            return ast.Statement{
                .query = ast.Query{
                    .tensor = lhs,
                    .location = location,
                },
            };
        }

        // Parse operator
        const op = try self.parseAccumulationOp();

        // Parse RHS expression
        const rhs = try self.parseExpr();

        // Consume newline if present (equation ends at newline or EOF)
        // Also accept comment before newline
        while (self.check(.comment)) {
            _ = self.advance();
        }
        if (self.check(.newline)) {
            _ = self.advance();
        }
        // At this point we should be at newline, EOF, or next statement

        return ast.Statement{
            .equation = self.builder.createEquation(lhs, op, rhs, location),
        };
    }

    fn parseAccumulationOp(self: *Parser) ParseError!ast.AccumulationOp {
        if (self.match(.equals)) {
            return .assign;
        }
        if (self.match(.plus_equals)) {
            return .add;
        }
        if (self.match(.star_equals)) {
            return .mul;
        }
        if (self.match(.max_equals)) {
            return .max;
        }
        if (self.match(.min_equals)) {
            return .min;
        }
        if (self.match(.avg_equals)) {
            return .avg;
        }
        self.recordError("expected '=', '+=', '*=', 'max=', 'min=', or 'avg='");
        return ParseError.ExpectedEquals;
    }

    fn parseTensorRef(self: *Parser) ParseError!ast.TensorRef {
        const name_tok = self.advance();
        if (name_tok.type != .identifier) {
            self.recordError("expected tensor name");
            return ParseError.ExpectedIdentifier;
        }

        var indices = std.ArrayListUnmanaged(ast.Index){};
        var is_boolean = false;

        if (self.check(.lbracket)) {
            // Numeric tensor: A[i,j]
            _ = self.advance();
            try self.parseIndices(&indices);
            if (!self.match(.rbracket)) {
                self.recordError("expected ']' after indices");
                return ParseError.ExpectedClosingBracket;
            }
        } else if (self.check(.lparen)) {
            // Boolean relation: Parent(x,y)
            is_boolean = true;
            _ = self.advance();
            try self.parseIndices(&indices);
            if (!self.match(.rparen)) {
                self.recordError("expected ')' after indices");
                return ParseError.ExpectedClosingParen;
            }
        }
        // No indices is also valid (scalar)

        return ast.TensorRef{
            .name = name_tok.lexeme,
            .indices = indices.toOwnedSlice(self.allocator) catch return ParseError.OutOfMemory,
            .is_boolean = is_boolean,
            .location = name_tok.location,
        };
    }

    fn parseIndices(self: *Parser, indices: *std.ArrayListUnmanaged(ast.Index)) ParseError!void {
        // Parse first index
        const idx = try self.parseIndex();
        indices.append(self.allocator, idx) catch return ParseError.OutOfMemory;

        // Parse remaining indices
        while (self.match(.comma)) {
            const next_idx = try self.parseIndex();
            indices.append(self.allocator, next_idx) catch return ParseError.OutOfMemory;
        }
    }

    fn parseIndex(self: *Parser) ParseError!ast.Index {
        // Check for virtual index: *t (use tensor t's values as indices)
        if (self.check(.star)) {
            _ = self.advance();
            const name = self.advance();
            if (name.type != .identifier) {
                return ParseError.ExpectedIdentifier;
            }
            return ast.Index{ .virtual = name.lexeme };
        }

        // Check for normalize index: /i
        if (self.check(.slash)) {
            _ = self.advance();
            const name = self.advance();
            if (name.type != .identifier) {
                return ParseError.ExpectedIdentifier;
            }
            return ast.Index{ .normalize = name.lexeme };
        }

        // Check for constant: 42
        if (self.check(.integer)) {
            const tok = self.advance();
            const val = std.fmt.parseInt(i64, tok.lexeme, 10) catch return ParseError.InvalidIndex;

            // Check for slice: 0:10
            if (self.match(.colon)) {
                if (self.check(.integer)) {
                    const end_tok = self.advance();
                    const end_val = std.fmt.parseInt(i64, end_tok.lexeme, 10) catch return ParseError.InvalidIndex;
                    return ast.Index{ .slice = .{ .start = val, .end = end_val } };
                }
                return ParseError.InvalidIndex;
            }

            return ast.Index{ .constant = val };
        }

        // Identifier-based index
        if (self.check(.identifier)) {
            const name_tok = self.advance();

            // Check for primed: p'
            if (self.match(.prime)) {
                return ast.Index{ .primed = name_tok.lexeme };
            }

            // Check for normalization axis: i. (softmax normalizes over this index)
            if (self.match(.dot)) {
                return ast.Index{ .normalize = name_tok.lexeme };
            }

            // Check for arithmetic: i+1, i-1
            if (self.check(.plus)) {
                _ = self.advance();
                if (self.check(.integer)) {
                    const offset_tok = self.advance();
                    const offset = std.fmt.parseInt(i64, offset_tok.lexeme, 10) catch return ParseError.InvalidIndex;
                    return ast.Index{ .arithmetic = .{
                        .base = name_tok.lexeme,
                        .op = .add,
                        .offset = offset,
                    } };
                }
                return ParseError.InvalidIndex;
            }
            if (self.check(.minus)) {
                _ = self.advance();
                if (self.check(.integer)) {
                    const offset_tok = self.advance();
                    const offset = std.fmt.parseInt(i64, offset_tok.lexeme, 10) catch return ParseError.InvalidIndex;
                    return ast.Index{ .arithmetic = .{
                        .base = name_tok.lexeme,
                        .op = .sub,
                        .offset = offset,
                    } };
                }
                return ParseError.InvalidIndex;
            }

            // Check for division: i/2
            if (self.check(.slash)) {
                _ = self.advance();
                if (self.check(.integer)) {
                    const div_tok = self.advance();
                    const divisor = std.fmt.parseInt(i64, div_tok.lexeme, 10) catch return ParseError.InvalidIndex;
                    return ast.Index{ .div = .{
                        .index = name_tok.lexeme,
                        .divisor = divisor,
                    } };
                }
                return ParseError.InvalidIndex;
            }

            // Simple named index
            return ast.Index{ .name = name_tok.lexeme };
        }

        return ParseError.InvalidIndex;
    }

    fn parseExpr(self: *Parser) ParseError!*ast.Expr {
        return self.parseImplicitProduct();
    }

    // Implicit multiplication: A[i,j] B[j,k] (adjacent tensors)
    fn parseImplicitProduct(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseAdditive();

        // Continue while we see another term that could be part of implicit multiplication
        // Only identifiers and nonlinearity keywords can start a new implicit factor
        // We need to be careful not to consume operators like + or -
        while (self.couldStartImplicitFactor()) {
            const right = try self.parseAdditive();

            // Create product node
            const factors = [_]*ast.Expr{ left, right };
            left = self.builder.createProductExpr(&factors) catch return ParseError.OutOfMemory;
        }

        return left;
    }

    fn couldStartImplicitFactor(self: *Parser) bool {
        const t = self.peek().type;
        // Only identifiers (tensor refs) and nonlinearity functions can start implicit factors
        // NOT: numbers, strings, operators, parens (those need explicit context)
        return t == .identifier or
            t == .kw_step or
            t == .kw_softmax or
            t == .kw_relu or
            t == .kw_sigmoid or
            t == .kw_tanh or
            t == .kw_exp or
            t == .kw_log or
            t == .kw_abs or
            t == .kw_sqrt or
            t == .kw_sin or
            t == .kw_cos or
            t == .kw_norm or
            t == .kw_embed;
    }

    // Addition/subtraction
    fn parseAdditive(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseMultiplicative();

        while (true) {
            if (self.match(.plus)) {
                const right = try self.parseMultiplicative();
                left = self.builder.createBinaryExpr(.add, left, right) catch return ParseError.OutOfMemory;
            } else if (self.match(.minus)) {
                const right = try self.parseMultiplicative();
                left = self.builder.createBinaryExpr(.sub, left, right) catch return ParseError.OutOfMemory;
            } else {
                break;
            }
        }

        return left;
    }

    // Explicit multiplication, division, power
    fn parseMultiplicative(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseUnary();

        while (true) {
            if (self.match(.star)) {
                const right = try self.parseUnary();
                left = self.builder.createBinaryExpr(.mul, left, right) catch return ParseError.OutOfMemory;
            } else if (self.match(.slash)) {
                const right = try self.parseUnary();
                left = self.builder.createBinaryExpr(.div, left, right) catch return ParseError.OutOfMemory;
            } else if (self.match(.caret)) {
                const right = try self.parseUnary();
                left = self.builder.createBinaryExpr(.pow, left, right) catch return ParseError.OutOfMemory;
            } else {
                break;
            }
        }

        return left;
    }

    fn parseUnary(self: *Parser) ParseError!*ast.Expr {
        if (self.match(.minus)) {
            const operand = try self.parseUnary();
            return self.builder.createUnaryExpr(.negate, operand) catch return ParseError.OutOfMemory;
        }

        return self.parsePrimary();
    }

    fn parsePrimary(self: *Parser) ParseError!*ast.Expr {
        // Conditional: if cond then_expr else else_expr
        if (self.match(.kw_if)) {
            const condition = try self.parseExpr();

            // 'then' is optional - can be if cond expr else expr
            // For now, require expression after condition
            const then_branch = try self.parseExpr();

            var else_branch: ?*ast.Expr = null;
            if (self.match(.kw_else)) {
                else_branch = try self.parseExpr();
            }

            const cond_expr = self.builder.createExpr(.{
                .conditional = .{
                    .condition = condition,
                    .then_branch = then_branch,
                    .else_branch = else_branch,
                },
            }) catch return ParseError.OutOfMemory;
            return cond_expr;
        }

        // Nonlinearity function: step(x), relu(x), etc.
        if (self.matchNonlinearity()) |func| {
            if (!self.match(.lparen)) {
                self.recordError("expected '(' after function name");
                return ParseError.ExpectedExpression;
            }
            const arg = try self.parseExpr();
            if (!self.match(.rparen)) {
                self.recordError("expected ')' after function argument");
                return ParseError.ExpectedClosingParen;
            }
            return self.builder.createNonlinearityExpr(func, arg) catch return ParseError.OutOfMemory;
        }

        // Parenthesized expression
        if (self.check(.lparen)) {
            // Could be grouped expression or tensor ref (boolean relation)
            // Peek ahead to decide
            if (self.isGroupedExpr()) {
                _ = self.advance(); // consume (
                const expr = try self.parseExpr();
                if (!self.match(.rparen)) {
                    self.recordError("expected ')' to close grouped expression");
                    return ParseError.ExpectedClosingParen;
                }
                const grouped = self.builder.createExpr(.{ .group = expr }) catch return ParseError.OutOfMemory;
                return grouped;
            }
        }

        // Tensor reference: A[i,j] or Parent(x,y)
        if (self.check(.identifier)) {
            const ref = try self.parseTensorRef();
            return self.builder.createTensorRefExpr(ref) catch return ParseError.OutOfMemory;
        }

        // Integer literal
        if (self.check(.integer)) {
            const tok = self.advance();
            const val = std.fmt.parseInt(i64, tok.lexeme, 10) catch return ParseError.ExpectedExpression;
            return self.builder.createLiteralExpr(.{ .integer = val }) catch return ParseError.OutOfMemory;
        }

        // Float literal
        if (self.check(.float)) {
            const tok = self.advance();
            const val = std.fmt.parseFloat(f64, tok.lexeme) catch return ParseError.ExpectedExpression;
            return self.builder.createLiteralExpr(.{ .float = val }) catch return ParseError.OutOfMemory;
        }

        // Boolean literals
        if (self.match(.kw_true)) {
            return self.builder.createLiteralExpr(.{ .boolean = true }) catch return ParseError.OutOfMemory;
        }
        if (self.match(.kw_false)) {
            return self.builder.createLiteralExpr(.{ .boolean = false }) catch return ParseError.OutOfMemory;
        }

        // String literal
        if (self.check(.string)) {
            const tok = self.advance();
            return self.builder.createLiteralExpr(.{ .string = tok.lexeme }) catch return ParseError.OutOfMemory;
        }

        self.recordError("expected expression (tensor, number, or function call)");
        return ParseError.ExpectedExpression;
    }

    fn isGroupedExpr(self: *Parser) bool {
        // If we see ( and the token after is not an identifier or the pattern
        // doesn't look like a relation, it's a grouped expression
        // For simplicity, check if second token is an operator or )
        if (self.current + 1 >= self.tokens_list.len) return true;

        const next = self.tokens_list[self.current + 1];
        // If the next token after ( is an operator or (, it's grouped
        return next.type != .identifier;
    }

    fn matchNonlinearity(self: *Parser) ?ast.Nonlinearity {
        if (self.match(.kw_step)) return .step;
        if (self.match(.kw_softmax)) return .softmax;
        if (self.match(.kw_relu)) return .relu;
        if (self.match(.kw_sigmoid)) return .sigmoid;
        if (self.match(.kw_tanh)) return .tanh;
        if (self.match(.kw_exp)) return .exp;
        if (self.match(.kw_log)) return .log;
        if (self.match(.kw_abs)) return .abs;
        if (self.match(.kw_sqrt)) return .sqrt;
        if (self.match(.kw_sin)) return .sin;
        if (self.match(.kw_cos)) return .cos;
        if (self.match(.kw_norm)) return .norm;
        if (self.match(.kw_lnorm)) return .lnorm;
        if (self.match(.kw_concat)) return .concat;
        return null;
    }

    fn parseDomainDecl(self: *Parser) ParseError!ast.Statement {
        const location = self.peek().location;
        _ = self.advance(); // consume 'domain'

        if (!self.check(.identifier)) {
            return ParseError.ExpectedIdentifier;
        }
        const name_tok = self.advance();

        var size: ?i64 = null;
        var type_hint: ?[]const u8 = null;

        if (self.match(.colon)) {
            if (self.check(.integer)) {
                const size_tok = self.advance();
                size = std.fmt.parseInt(i64, size_tok.lexeme, 10) catch return ParseError.InvalidIndex;
            } else if (self.check(.identifier)) {
                const hint_tok = self.advance();
                type_hint = hint_tok.lexeme;
            }
        }

        return ast.Statement{
            .domain_decl = .{
                .name = name_tok.lexeme,
                .size = size,
                .type_hint = type_hint,
                .location = location,
            },
        };
    }

    fn parseSparseDecl(self: *Parser) ParseError!ast.Statement {
        const location = self.peek().location;
        _ = self.advance(); // consume 'sparse'

        if (!self.check(.identifier)) {
            return ParseError.ExpectedIdentifier;
        }
        const name_tok = self.advance();

        var is_boolean = false;
        var indices = std.ArrayListUnmanaged(ast.SparseDecl.IndexDecl){};

        if (self.check(.lbracket) or self.check(.lparen)) {
            is_boolean = self.check(.lparen);
            _ = self.advance();

            // Parse index declarations
            while (!self.check(.rbracket) and !self.check(.rparen) and !self.isAtEnd()) {
                if (self.check(.identifier)) {
                    const idx_name = self.advance().lexeme;
                    var domain: ?[]const u8 = null;

                    if (self.match(.colon)) {
                        if (self.check(.identifier)) {
                            domain = self.advance().lexeme;
                        }
                    }

                    indices.append(self.allocator, .{ .name = idx_name, .domain = domain }) catch return ParseError.OutOfMemory;

                    if (!self.check(.rbracket) and !self.check(.rparen)) {
                        if (!self.match(.comma)) {
                            break;
                        }
                    }
                } else {
                    break;
                }
            }

            if (is_boolean) {
                if (!self.match(.rparen)) return ParseError.ExpectedClosingParen;
            } else {
                if (!self.match(.rbracket)) return ParseError.ExpectedClosingBracket;
            }
        }

        return ast.Statement{
            .sparse_decl = .{
                .name = name_tok.lexeme,
                .indices = indices.toOwnedSlice(self.allocator) catch return ParseError.OutOfMemory,
                .is_boolean = is_boolean,
                .location = location,
            },
        };
    }

    fn parseImport(self: *Parser) ParseError!ast.Statement {
        const location = self.peek().location;
        _ = self.advance(); // consume 'import'

        if (!self.check(.string) and !self.check(.identifier)) {
            return ParseError.ExpectedExpression;
        }
        const path_tok = self.advance();

        // Handle 'as' for aliases: import "file.tl" as foo
        var alias: ?[]const u8 = null;
        if (self.check(.kw_as)) {
            _ = self.advance(); // consume 'as'
            if (!self.check(.identifier)) {
                return ParseError.ExpectedIdentifier;
            }
            alias = self.advance().lexeme;
        }

        return ast.Statement{
            .import_stmt = .{
                .path = path_tok.lexeme,
                .alias = alias,
                .location = location,
            },
        };
    }

    fn parseExport(self: *Parser) ParseError!ast.Statement {
        const location = self.peek().location;
        _ = self.advance(); // consume 'export'

        if (!self.check(.identifier)) {
            return ParseError.ExpectedIdentifier;
        }
        const name_tok = self.advance();

        return ast.Statement{
            .export_stmt = .{
                .name = name_tok.lexeme,
                .location = location,
            },
        };
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    fn peek(self: *Parser) Token {
        return self.tokens_list[self.current];
    }

    fn previous(self: *Parser) Token {
        return self.tokens_list[self.current - 1];
    }

    fn isAtEnd(self: *Parser) bool {
        return self.peek().type == .eof;
    }

    fn advance(self: *Parser) Token {
        if (!self.isAtEnd()) {
            self.current += 1;
        }
        return self.previous();
    }

    fn check(self: *Parser, expected: TokenType) bool {
        if (self.isAtEnd()) return false;
        return self.peek().type == expected;
    }

    fn match(self: *Parser, expected: TokenType) bool {
        if (self.check(expected)) {
            _ = self.advance();
            return true;
        }
        return false;
    }
};

// ============================================================================
// Tests
// ============================================================================

const lexer = @import("lexer.zig");

fn parseSource(backing_allocator: std.mem.Allocator, source: []const u8) !ast.Program {
    // Create arena to own all AST memory
    const arena = try backing_allocator.create(std.heap.ArenaAllocator);
    arena.* = std.heap.ArenaAllocator.init(backing_allocator);
    const allocator = arena.allocator();

    // Copy source to arena (lexer needs stable memory)
    const src_copy = try allocator.dupe(u8, source);

    var lex = lexer.Lexer.init(allocator, src_copy);
    const toks = try lex.scanTokens();

    var p = Parser.init(allocator, toks);
    var program = try p.parse();
    program.arena = arena; // Program now owns the arena
    return program;
}

test "parse simple equation" {
    const allocator = std.testing.allocator;
    var program = try parseSource(allocator, "C[i,k] = A[i,j] B[j,k]");
    defer program.deinit();

    try std.testing.expectEqual(@as(usize, 1), program.statements.len);

    const stmt = program.statements[0];
    try std.testing.expectEqual(ast.Statement.equation, @as(std.meta.Tag(ast.Statement), stmt));

    const eq = stmt.equation;
    try std.testing.expectEqualStrings("C", eq.lhs.name);
    try std.testing.expectEqual(@as(usize, 2), eq.lhs.indices.len);
    try std.testing.expectEqual(ast.AccumulationOp.assign, eq.op);
}

test "parse boolean relation" {
    const allocator = std.testing.allocator;
    var program = try parseSource(allocator, "Ancestor(x,z) = Parent(x,z)");
    defer program.deinit();

    try std.testing.expectEqual(@as(usize, 1), program.statements.len);

    const eq = program.statements[0].equation;
    try std.testing.expectEqualStrings("Ancestor", eq.lhs.name);
    try std.testing.expectEqual(true, eq.lhs.is_boolean);
}

test "parse step function" {
    const allocator = std.testing.allocator;
    var program = try parseSource(allocator, "Ancestor(x,z) = step(Parent(x,y) Ancestor(y,z))");
    defer program.deinit();

    const eq = program.statements[0].equation;
    try std.testing.expectEqual(ast.Expr.nonlinearity, @as(std.meta.Tag(ast.Expr), eq.rhs.*));
    try std.testing.expectEqual(ast.Nonlinearity.step, eq.rhs.nonlinearity.func);
}

test "parse accumulation" {
    const allocator = std.testing.allocator;
    var program = try parseSource(allocator, "A[i] += B[i]");
    defer program.deinit();

    const eq = program.statements[0].equation;
    try std.testing.expectEqual(ast.AccumulationOp.add, eq.op);
}

test "parse primed index" {
    const allocator = std.testing.allocator;
    var program = try parseSource(allocator, "H[p,p'] = W[p,p']");
    defer program.deinit();

    const eq = program.statements[0].equation;
    try std.testing.expectEqual(@as(usize, 2), eq.lhs.indices.len);
    try std.testing.expectEqual(ast.Index.primed, @as(std.meta.Tag(ast.Index), eq.lhs.indices[1]));
}

test "parse arithmetic index" {
    const allocator = std.testing.allocator;
    var program = try parseSource(allocator, "Y[i+1] = X[i]");
    defer program.deinit();

    const eq = program.statements[0].equation;
    const idx = eq.lhs.indices[0];
    try std.testing.expectEqual(ast.Index.arithmetic, @as(std.meta.Tag(ast.Index), idx));
    try std.testing.expectEqualStrings("i", idx.arithmetic.base);
    try std.testing.expectEqual(@as(i64, 1), idx.arithmetic.offset);
}

test "parse multiple statements" {
    const allocator = std.testing.allocator;
    const source =
        \\Ancestor(x,z) = Parent(x,z)
        \\Ancestor(x,z) += step(Parent(x,y) Ancestor(y,z))
    ;
    var program = try parseSource(allocator, source);
    defer program.deinit();

    // Count non-comment statements
    var eq_count: usize = 0;
    for (program.statements) |stmt| {
        if (stmt == .equation) eq_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), eq_count);
}
