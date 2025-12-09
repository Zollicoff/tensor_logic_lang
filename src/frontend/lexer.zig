// Tensor Logic Lexer
// Tokenizes .tl source files into a stream of tokens

const std = @import("std");
const tokens = @import("tokens.zig");
const Token = tokens.Token;
const TokenType = tokens.TokenType;
const SourceLocation = tokens.SourceLocation;

pub const LexerError = error{
    UnterminatedString,
    InvalidCharacter,
    InvalidNumber,
    OutOfMemory,
};

pub const Lexer = struct {
    source: []const u8,
    start: usize,
    current: usize,
    line: u32,
    column: u32,
    line_start: usize,
    allocator: std.mem.Allocator,
    token_list: std.ArrayListUnmanaged(Token),

    pub fn init(allocator: std.mem.Allocator, source: []const u8) Lexer {
        return Lexer{
            .source = source,
            .start = 0,
            .current = 0,
            .line = 1,
            .column = 1,
            .line_start = 0,
            .allocator = allocator,
            .token_list = .{},
        };
    }

    pub fn deinit(self: *Lexer) void {
        self.token_list.deinit(self.allocator);
    }

    pub fn scanTokens(self: *Lexer) LexerError![]Token {
        while (!self.isAtEnd()) {
            self.start = self.current;
            try self.scanToken();
        }

        // Add EOF token
        self.token_list.append(self.allocator, Token.init(
            .eof,
            "",
            self.currentLocation(),
        )) catch return LexerError.OutOfMemory;

        return self.token_list.items;
    }

    fn scanToken(self: *Lexer) LexerError!void {
        const c = self.advance();

        switch (c) {
            // Single character tokens
            '[' => try self.addToken(.lbracket),
            ']' => try self.addToken(.rbracket),
            '(' => try self.addToken(.lparen),
            ')' => try self.addToken(.rparen),
            '{' => try self.addToken(.lbrace),
            '}' => try self.addToken(.rbrace),
            '*' => {
                if (self.match('=')) {
                    try self.addToken(.star_equals);
                } else {
                    try self.addToken(.star);
                }
            },
            '^' => try self.addToken(.caret),
            '?' => try self.addToken(.question),
            '@' => try self.addToken(.at),
            '&' => try self.addToken(.ampersand),
            '|' => try self.addToken(.pipe),
            ',' => try self.addToken(.comma),
            ':' => try self.addToken(.colon),
            '.' => try self.addToken(.dot),
            '\'' => try self.addToken(.prime),
            ';' => try self.addToken(.semicolon),

            // Two-character tokens
            '+' => {
                if (self.match('=')) {
                    try self.addToken(.plus_equals);
                } else {
                    try self.addToken(.plus);
                }
            },
            '-' => try self.addToken(.minus),
            '/' => {
                if (self.match('/')) {
                    // Line comment
                    while (self.peek() != '\n' and !self.isAtEnd()) {
                        _ = self.advance();
                    }
                    try self.addToken(.comment);
                } else {
                    try self.addToken(.slash);
                }
            },
            '=' => {
                if (self.match('=')) {
                    try self.addToken(.equal_equal);
                } else {
                    try self.addToken(.equals);
                }
            },
            '<' => {
                if (self.match('=')) {
                    try self.addToken(.less_equal);
                } else {
                    try self.addToken(.less);
                }
            },
            '>' => {
                if (self.match('=')) {
                    try self.addToken(.greater_equal);
                } else {
                    try self.addToken(.greater);
                }
            },
            '!' => {
                if (self.match('=')) {
                    try self.addToken(.not_equal);
                } else {
                    return LexerError.InvalidCharacter;
                }
            },

            // Whitespace
            ' ', '\r', '\t' => {},

            // Newline
            '\n' => {
                try self.addToken(.newline);
                self.line += 1;
                self.column = 1;
                self.line_start = self.current;
            },

            // String literals
            '"' => try self.string(),

            // Numbers and identifiers
            else => {
                if (isDigit(c)) {
                    try self.number();
                } else if (isAlpha(c)) {
                    try self.identifier();
                } else {
                    return LexerError.InvalidCharacter;
                }
            },
        }
    }

    fn string(self: *Lexer) LexerError!void {
        while (self.peek() != '"' and !self.isAtEnd()) {
            if (self.peek() == '\n') {
                self.line += 1;
                self.column = 1;
            }
            _ = self.advance();
        }

        if (self.isAtEnd()) {
            return LexerError.UnterminatedString;
        }

        // Consume closing quote
        _ = self.advance();

        // Trim quotes from lexeme (store just the content)
        try self.addToken(.string);
    }

    fn number(self: *Lexer) LexerError!void {
        while (isDigit(self.peek())) {
            _ = self.advance();
        }

        var is_float = false;

        // Look for fractional part
        if (self.peek() == '.' and isDigit(self.peekNext())) {
            is_float = true;
            // Consume the dot
            _ = self.advance();

            while (isDigit(self.peek())) {
                _ = self.advance();
            }
        }

        // Scientific notation (can appear with or without decimal point)
        if (self.peek() == 'e' or self.peek() == 'E') {
            is_float = true;
            _ = self.advance();
            if (self.peek() == '+' or self.peek() == '-') {
                _ = self.advance();
            }
            while (isDigit(self.peek())) {
                _ = self.advance();
            }
        }

        if (is_float) {
            try self.addToken(.float);
        } else {
            try self.addToken(.integer);
        }
    }

    fn identifier(self: *Lexer) LexerError!void {
        while (isAlphaNumeric(self.peek())) {
            _ = self.advance();
        }

        const text = self.source[self.start..self.current];

        // Check for compound assignment keywords (max=, min=, avg=)
        if (std.mem.eql(u8, text, "max") and self.match('=')) {
            try self.addToken(.max_equals);
        } else if (std.mem.eql(u8, text, "min") and self.match('=')) {
            try self.addToken(.min_equals);
        } else if (std.mem.eql(u8, text, "avg") and self.match('=')) {
            try self.addToken(.avg_equals);
        } else {
            const token_type = tokens.lookupKeyword(text);
            try self.addToken(token_type);
        }
    }

    fn advance(self: *Lexer) u8 {
        const c = self.source[self.current];
        self.current += 1;
        self.column += 1;
        return c;
    }

    fn peek(self: *Lexer) u8 {
        if (self.isAtEnd()) return 0;
        return self.source[self.current];
    }

    fn peekNext(self: *Lexer) u8 {
        if (self.current + 1 >= self.source.len) return 0;
        return self.source[self.current + 1];
    }

    fn match(self: *Lexer, expected: u8) bool {
        if (self.isAtEnd()) return false;
        if (self.source[self.current] != expected) return false;
        self.current += 1;
        self.column += 1;
        return true;
    }

    fn isAtEnd(self: *Lexer) bool {
        return self.current >= self.source.len;
    }

    fn currentLocation(self: *Lexer) SourceLocation {
        const col: u32 = if (self.start >= self.line_start)
            @intCast(self.start - self.line_start + 1)
        else
            1;
        return SourceLocation{
            .line = self.line,
            .column = col,
            .offset = self.start,
        };
    }

    fn addToken(self: *Lexer, token_type: TokenType) LexerError!void {
        const lexeme = self.source[self.start..self.current];
        self.token_list.append(self.allocator, Token.init(
            token_type,
            lexeme,
            self.currentLocation(),
        )) catch return LexerError.OutOfMemory;
    }
};

fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

fn isAlpha(c: u8) bool {
    return (c >= 'a' and c <= 'z') or
        (c >= 'A' and c <= 'Z') or
        c == '_';
}

fn isAlphaNumeric(c: u8) bool {
    return isAlpha(c) or isDigit(c);
}

// ============================================================================
// Tests
// ============================================================================

test "lexer basic tokens" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "[ ] ( ) + - * /");
    defer lexer.deinit();

    const toks = try lexer.scanTokens();

    try std.testing.expectEqual(@as(usize, 9), toks.len); // 8 tokens + EOF
    try std.testing.expectEqual(TokenType.lbracket, toks[0].type);
    try std.testing.expectEqual(TokenType.rbracket, toks[1].type);
    try std.testing.expectEqual(TokenType.lparen, toks[2].type);
    try std.testing.expectEqual(TokenType.rparen, toks[3].type);
    try std.testing.expectEqual(TokenType.plus, toks[4].type);
    try std.testing.expectEqual(TokenType.minus, toks[5].type);
    try std.testing.expectEqual(TokenType.star, toks[6].type);
    try std.testing.expectEqual(TokenType.slash, toks[7].type);
    try std.testing.expectEqual(TokenType.eof, toks[8].type);
}

test "lexer tensor equation" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "C[i,k] = A[i,j] B[j,k]");
    defer lexer.deinit();

    const toks = try lexer.scanTokens();

    try std.testing.expectEqual(TokenType.identifier, toks[0].type);
    try std.testing.expectEqualStrings("C", toks[0].lexeme);
    try std.testing.expectEqual(TokenType.lbracket, toks[1].type);
    try std.testing.expectEqual(TokenType.identifier, toks[2].type);
    try std.testing.expectEqualStrings("i", toks[2].lexeme);
    try std.testing.expectEqual(TokenType.comma, toks[3].type);
    try std.testing.expectEqual(TokenType.identifier, toks[4].type);
    try std.testing.expectEqualStrings("k", toks[4].lexeme);
    try std.testing.expectEqual(TokenType.rbracket, toks[5].type);
    try std.testing.expectEqual(TokenType.equals, toks[6].type);
}

test "lexer boolean relation" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "Ancestor(x,z) = step(Parent(x,y) Ancestor(y,z))");
    defer lexer.deinit();

    const toks = try lexer.scanTokens();

    try std.testing.expectEqual(TokenType.identifier, toks[0].type);
    try std.testing.expectEqualStrings("Ancestor", toks[0].lexeme);
    try std.testing.expectEqual(TokenType.lparen, toks[1].type);
    try std.testing.expectEqual(TokenType.identifier, toks[2].type);
    try std.testing.expectEqualStrings("x", toks[2].lexeme);
    try std.testing.expectEqual(TokenType.comma, toks[3].type);
    try std.testing.expectEqual(TokenType.identifier, toks[4].type);
    try std.testing.expectEqualStrings("z", toks[4].lexeme);
    try std.testing.expectEqual(TokenType.rparen, toks[5].type);
    try std.testing.expectEqual(TokenType.equals, toks[6].type);
    try std.testing.expectEqual(TokenType.kw_step, toks[7].type);
}

test "lexer numbers" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "42 3.14 1e-5 2.5e10");
    defer lexer.deinit();

    const toks = try lexer.scanTokens();

    try std.testing.expectEqual(TokenType.integer, toks[0].type);
    try std.testing.expectEqualStrings("42", toks[0].lexeme);
    try std.testing.expectEqual(TokenType.float, toks[1].type);
    try std.testing.expectEqualStrings("3.14", toks[1].lexeme);
    try std.testing.expectEqual(TokenType.float, toks[2].type);
    try std.testing.expectEqualStrings("1e-5", toks[2].lexeme);
    try std.testing.expectEqual(TokenType.float, toks[3].type);
    try std.testing.expectEqualStrings("2.5e10", toks[3].lexeme);
}

test "lexer prime operator" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "H[p,p']");
    defer lexer.deinit();

    const toks = try lexer.scanTokens();

    try std.testing.expectEqual(TokenType.identifier, toks[0].type);
    try std.testing.expectEqualStrings("H", toks[0].lexeme);
    try std.testing.expectEqual(TokenType.lbracket, toks[1].type);
    try std.testing.expectEqual(TokenType.identifier, toks[2].type);
    try std.testing.expectEqualStrings("p", toks[2].lexeme);
    try std.testing.expectEqual(TokenType.comma, toks[3].type);
    try std.testing.expectEqual(TokenType.identifier, toks[4].type);
    try std.testing.expectEqualStrings("p", toks[4].lexeme);
    try std.testing.expectEqual(TokenType.prime, toks[5].type);
    try std.testing.expectEqual(TokenType.rbracket, toks[6].type);
}

test "lexer accumulation operators" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "+= max= min= avg=");
    defer lexer.deinit();

    const toks = try lexer.scanTokens();

    try std.testing.expectEqual(TokenType.plus_equals, toks[0].type);
    try std.testing.expectEqual(TokenType.max_equals, toks[1].type);
    try std.testing.expectEqual(TokenType.min_equals, toks[2].type);
    try std.testing.expectEqual(TokenType.avg_equals, toks[3].type);
}

test "lexer comments" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "A[i] = B[i] // this is a comment\nC[j] = D[j]");
    defer lexer.deinit();

    const toks = try lexer.scanTokens();

    // Should have: A [ i ] = B [ i ] comment newline C [ j ] = D [ j ] eof
    var found_comment = false;
    var found_C = false;
    for (toks) |tok| {
        if (tok.type == .comment) found_comment = true;
        if (tok.type == .identifier and std.mem.eql(u8, tok.lexeme, "C")) found_C = true;
    }
    try std.testing.expect(found_comment);
    try std.testing.expect(found_C);
}

test "lexer string literals" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "\"hello world\"");
    defer lexer.deinit();

    const toks = try lexer.scanTokens();

    try std.testing.expectEqual(TokenType.string, toks[0].type);
    try std.testing.expectEqualStrings("\"hello world\"", toks[0].lexeme);
}

test "lexer multiline" {
    const allocator = std.testing.allocator;

    const source =
        \\// Transitive closure
        \\Ancestor(x,z) = Parent(x,z)
        \\Ancestor(x,z) += step(Parent(x,y) Ancestor(y,z))
    ;

    var lexer = Lexer.init(allocator, source);
    defer lexer.deinit();

    const toks = try lexer.scanTokens();

    // Count newlines
    var newline_count: usize = 0;
    for (toks) |tok| {
        if (tok.type == .newline) newline_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), newline_count);
}

test "lexer location tracking" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "A[i]\nB[j]");
    defer lexer.deinit();

    const toks = try lexer.scanTokens();

    // A is at line 1, column 1
    try std.testing.expectEqual(@as(u32, 1), toks[0].location.line);
    try std.testing.expectEqual(@as(u32, 1), toks[0].location.column);

    // B is at line 2, column 1
    // Find B token
    for (toks) |tok| {
        if (tok.type == .identifier and std.mem.eql(u8, tok.lexeme, "B")) {
            try std.testing.expectEqual(@as(u32, 2), tok.location.line);
            try std.testing.expectEqual(@as(u32, 1), tok.location.column);
            break;
        }
    }
}
