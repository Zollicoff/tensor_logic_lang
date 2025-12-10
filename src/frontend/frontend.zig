// Frontend module - exports all frontend components for external use

pub const lexer = @import("lexer.zig");
pub const parser = @import("parser.zig");
pub const ast = @import("ast.zig");
pub const tokens = @import("tokens.zig");
pub const types = @import("types.zig");
pub const optimize = @import("optimize.zig");

pub const Lexer = lexer.Lexer;
pub const Parser = parser.Parser;
pub const SourceLocation = tokens.SourceLocation;
pub const Token = tokens.Token;
pub const TokenType = tokens.TokenType;
