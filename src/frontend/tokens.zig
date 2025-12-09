// Tensor Logic Token Definitions
// Based on the grammar from Pedro Domingos' "Tensor Logic: The Language of AI"

const std = @import("std");

pub const TokenType = enum {
    // Literals
    integer,
    float,
    string,
    identifier,

    // Brackets and parens
    lbracket, // [
    rbracket, // ]
    lparen, // (
    rparen, // )
    lbrace, // {
    rbrace, // }

    // Operators
    equals, // =
    plus, // +
    minus, // -
    star, // *
    slash, // /
    caret, // ^
    question, // ?
    at, // @
    ampersand, // &
    pipe, // |
    less, // <
    greater, // >
    less_equal, // <=
    greater_equal, // >=
    equal_equal, // ==
    not_equal, // !=

    // Punctuation
    comma, // ,
    colon, // :
    dot, // .
    prime, // '
    semicolon, // ;

    // Compound assignment (accumulation operators)
    plus_equals, // +=
    star_equals, // *=
    max_equals, // max=
    min_equals, // min=
    avg_equals, // avg=

    // Keywords
    kw_step, // step (Heaviside function)
    kw_softmax,
    kw_relu,
    kw_sigmoid,
    kw_tanh,
    kw_exp,
    kw_log,
    kw_abs,
    kw_sqrt,
    kw_sin,
    kw_cos,
    kw_norm,
    kw_lnorm, // layer normalization
    kw_concat, // concatenation
    kw_embed, // embedding lookup
    kw_sparse, // sparse tensor declaration
    kw_domain, // domain declaration
    kw_import,
    kw_export,
    kw_as,
    kw_if,
    kw_else,
    kw_for,
    kw_in,
    kw_true,
    kw_false,
    kw_save,
    kw_load,

    // Type keywords
    kw_real,
    kw_int,
    kw_bool,
    kw_complex,

    // Control
    newline,
    eof,
    comment,

    // Error
    invalid,

    pub fn symbol(self: TokenType) []const u8 {
        return switch (self) {
            .integer => "integer",
            .float => "float",
            .string => "string",
            .identifier => "identifier",
            .lbracket => "[",
            .rbracket => "]",
            .lparen => "(",
            .rparen => ")",
            .lbrace => "{",
            .rbrace => "}",
            .equals => "=",
            .plus => "+",
            .minus => "-",
            .star => "*",
            .slash => "/",
            .caret => "^",
            .question => "?",
            .at => "@",
            .ampersand => "&",
            .pipe => "|",
            .less => "<",
            .greater => ">",
            .less_equal => "<=",
            .greater_equal => ">=",
            .equal_equal => "==",
            .not_equal => "!=",
            .comma => ",",
            .colon => ":",
            .dot => ".",
            .prime => "'",
            .semicolon => ";",
            .plus_equals => "+=",
            .star_equals => "*=",
            .max_equals => "max=",
            .min_equals => "min=",
            .avg_equals => "avg=",
            .kw_step => "step",
            .kw_softmax => "softmax",
            .kw_relu => "relu",
            .kw_sigmoid => "sigmoid",
            .kw_tanh => "tanh",
            .kw_exp => "exp",
            .kw_log => "log",
            .kw_abs => "abs",
            .kw_sqrt => "sqrt",
            .kw_sin => "sin",
            .kw_cos => "cos",
            .kw_norm => "norm",
            .kw_lnorm => "lnorm",
            .kw_concat => "concat",
            .kw_embed => "embed",
            .kw_sparse => "sparse",
            .kw_domain => "domain",
            .kw_import => "import",
            .kw_export => "export",
            .kw_as => "as",
            .kw_if => "if",
            .kw_else => "else",
            .kw_for => "for",
            .kw_in => "in",
            .kw_true => "true",
            .kw_false => "false",
            .kw_save => "save",
            .kw_load => "load",
            .kw_real => "real",
            .kw_int => "int",
            .kw_bool => "bool",
            .kw_complex => "complex",
            .newline => "newline",
            .eof => "eof",
            .comment => "comment",
            .invalid => "invalid",
        };
    }
};

pub const SourceLocation = struct {
    line: u32,
    column: u32,
    offset: usize,

    pub fn format(
        self: SourceLocation,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{}:{}", .{ self.line, self.column });
    }
};

pub const Token = struct {
    type: TokenType,
    lexeme: []const u8,
    location: SourceLocation,

    pub fn init(token_type: TokenType, lexeme: []const u8, location: SourceLocation) Token {
        return Token{
            .type = token_type,
            .lexeme = lexeme,
            .location = location,
        };
    }

    pub fn formatToken(self: Token, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator, "Token({s}, \"{s}\", {d}:{d})", .{
            @tagName(self.type),
            self.lexeme,
            self.location.line,
            self.location.column,
        });
    }
};

// Keyword lookup table
pub const keywords = std.StaticStringMap(TokenType).initComptime(.{
    .{ "step", .kw_step },
    .{ "softmax", .kw_softmax },
    .{ "relu", .kw_relu },
    .{ "sigmoid", .kw_sigmoid },
    .{ "tanh", .kw_tanh },
    .{ "exp", .kw_exp },
    .{ "log", .kw_log },
    .{ "abs", .kw_abs },
    .{ "sqrt", .kw_sqrt },
    .{ "sin", .kw_sin },
    .{ "cos", .kw_cos },
    .{ "norm", .kw_norm },
    .{ "lnorm", .kw_lnorm },
    .{ "concat", .kw_concat },
    .{ "embed", .kw_embed },
    .{ "sparse", .kw_sparse },
    .{ "domain", .kw_domain },
    .{ "import", .kw_import },
    .{ "export", .kw_export },
    .{ "as", .kw_as },
    .{ "if", .kw_if },
    .{ "else", .kw_else },
    .{ "for", .kw_for },
    .{ "in", .kw_in },
    .{ "true", .kw_true },
    .{ "false", .kw_false },
    .{ "save", .kw_save },
    .{ "load", .kw_load },
    .{ "real", .kw_real },
    .{ "int", .kw_int },
    .{ "bool", .kw_bool },
    .{ "complex", .kw_complex },
});

pub fn lookupKeyword(identifier: []const u8) TokenType {
    return keywords.get(identifier) orelse .identifier;
}

test "keyword lookup" {
    const testing = std.testing;
    try testing.expectEqual(TokenType.kw_step, lookupKeyword("step"));
    try testing.expectEqual(TokenType.kw_softmax, lookupKeyword("softmax"));
    try testing.expectEqual(TokenType.identifier, lookupKeyword("foo"));
    try testing.expectEqual(TokenType.identifier, lookupKeyword("MyTensor"));
}
