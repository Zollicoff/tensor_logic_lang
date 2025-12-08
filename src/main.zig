// Tensor Logic Compiler (tlc)
// A compiler for the Tensor Logic language based on Pedro Domingos' paper

const std = @import("std");
const lexer = @import("frontend/lexer.zig");
const parser = @import("frontend/parser.zig");
const ast = @import("frontend/ast.zig");
const interpreter = @import("runtime/interpreter.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printUsage();
        return;
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "lex")) {
        if (args.len < 3) {
            std.debug.print("Error: 'lex' command requires a file path\n", .{});
            return;
        }
        try lexFile(allocator, args[2]);
    } else if (std.mem.eql(u8, command, "parse")) {
        if (args.len < 3) {
            std.debug.print("Error: 'parse' command requires a file path\n", .{});
            return;
        }
        try parseFile(allocator, args[2]);
    } else if (std.mem.eql(u8, command, "run")) {
        if (args.len < 3) {
            std.debug.print("Error: 'run' command requires a file path\n", .{});
            return;
        }
        // Check for --fixpoint or -f flag
        var use_fixpoint = false;
        var file_path: []const u8 = args[2];
        if (args.len >= 4) {
            if (std.mem.eql(u8, args[2], "--fixpoint") or std.mem.eql(u8, args[2], "-f")) {
                use_fixpoint = true;
                file_path = args[3];
            } else if (std.mem.eql(u8, args[3], "--fixpoint") or std.mem.eql(u8, args[3], "-f")) {
                use_fixpoint = true;
            }
        }
        try runFile(allocator, file_path, use_fixpoint);
    } else if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        try printUsage();
    } else if (std.mem.eql(u8, command, "version") or std.mem.eql(u8, command, "--version") or std.mem.eql(u8, command, "-v")) {
        try printVersion();
    } else {
        // Assume it's a file path
        try runFile(allocator, command, false);
    }
}

fn printUsage() !void {
    const stdout = std.fs.File.stdout();
    try stdout.writeAll(
        \\Tensor Logic Compiler (tlc) v0.1.0
        \\
        \\A language where everything is a tensor equation.
        \\Based on Pedro Domingos' "Tensor Logic: The Language of AI"
        \\
        \\Usage: tlc <command> [options] [file]
        \\
        \\Commands:
        \\  lex <file>           Tokenize a .tl file and print tokens
        \\  parse <file>         Parse a .tl file and print AST summary
        \\  run <file>           Execute a .tl program (single pass)
        \\  run -f <file>        Execute with fixpoint iteration (for recursive rules)
        \\  help                 Show this help message
        \\  version              Show version information
        \\
        \\Options:
        \\  -f, --fixpoint       Run until convergence (for recursive rules like Ancestor)
        \\
        \\Examples:
        \\  tlc lex examples/matmul.tl
        \\  tlc parse examples/ancestor.tl
        \\  tlc run examples/matmul.tl
        \\  tlc run -f examples/ancestor.tl
        \\
    );
}

fn printVersion() !void {
    const stdout = std.fs.File.stdout();
    try stdout.writeAll("tlc version 0.1.0\n");
}

fn lexFile(allocator: std.mem.Allocator, path: []const u8) !void {
    const source = std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024) catch |err| {
        std.debug.print("Error reading file '{s}': {}\n", .{ path, err });
        return;
    };
    defer allocator.free(source);

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = lex.scanTokens() catch |err| {
        std.debug.print("Lexer error: {}\n", .{err});
        return;
    };

    const stdout = std.fs.File.stdout();
    for (tokens) |tok| {
        const line = tok.formatToken(allocator) catch continue;
        defer allocator.free(line);
        stdout.writeAll(line) catch continue;
        stdout.writeAll("\n") catch continue;
    }
}

fn parseFile(allocator: std.mem.Allocator, path: []const u8) !void {
    const source = std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024) catch |err| {
        std.debug.print("Error reading file '{s}': {}\n", .{ path, err });
        return;
    };
    defer allocator.free(source);

    // Use arena for AST nodes
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ast_allocator = arena.allocator();

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = lex.scanTokens() catch |err| {
        std.debug.print("Lexer error: {}\n", .{err});
        return;
    };

    var p = parser.Parser.init(ast_allocator, tokens);
    defer p.deinit();

    const program = p.parse() catch |err| {
        std.debug.print("Parse error: {}\n", .{err});
        return;
    };

    const stdout = std.fs.File.stdout();
    const header = std.fmt.allocPrint(allocator, "Parsed {s}: {d} statement(s)\n\n", .{ path, program.statements.len }) catch return;
    defer allocator.free(header);
    stdout.writeAll(header) catch return;

    for (program.statements, 0..) |stmt, i| {
        const num = std.fmt.allocPrint(allocator, "[{d}] ", .{i + 1}) catch continue;
        defer allocator.free(num);
        stdout.writeAll(num) catch continue;

        switch (stmt) {
            .equation => |eq| {
                const msg = std.fmt.allocPrint(allocator, "Equation: {s}[{d} indices] {s} <expr>\n", .{
                    eq.lhs.name,
                    eq.lhs.indices.len,
                    @tagName(eq.op),
                }) catch continue;
                defer allocator.free(msg);
                stdout.writeAll(msg) catch continue;
            },
            .domain_decl => |d| {
                const msg = std.fmt.allocPrint(allocator, "Domain: {s}\n", .{d.name}) catch continue;
                defer allocator.free(msg);
                stdout.writeAll(msg) catch continue;
            },
            .sparse_decl => |s| {
                const msg = std.fmt.allocPrint(allocator, "Sparse: {s}\n", .{s.name}) catch continue;
                defer allocator.free(msg);
                stdout.writeAll(msg) catch continue;
            },
            .import_stmt => |im| {
                const msg = std.fmt.allocPrint(allocator, "Import: {s}\n", .{im.path}) catch continue;
                defer allocator.free(msg);
                stdout.writeAll(msg) catch continue;
            },
            .export_stmt => |ex| {
                const msg = std.fmt.allocPrint(allocator, "Export: {s}\n", .{ex.name}) catch continue;
                defer allocator.free(msg);
                stdout.writeAll(msg) catch continue;
            },
            .comment => |c| {
                const msg = std.fmt.allocPrint(allocator, "Comment: {s}\n", .{c}) catch continue;
                defer allocator.free(msg);
                stdout.writeAll(msg) catch continue;
            },
        }
    }
}

fn runFile(allocator: std.mem.Allocator, path: []const u8, use_fixpoint: bool) !void {
    const stdout = std.fs.File.stdout();

    const source = std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024) catch |err| {
        std.debug.print("Error reading file '{s}': {}\n", .{ path, err });
        return;
    };
    defer allocator.free(source);

    // Use arena for AST
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ast_allocator = arena.allocator();

    // Lex
    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = lex.scanTokens() catch |err| {
        std.debug.print("Lexer error: {}\n", .{err});
        return;
    };

    // Parse
    var p = parser.Parser.init(ast_allocator, tokens);
    defer p.deinit();

    const program = p.parse() catch |err| {
        std.debug.print("Parse error: {}\n", .{err});
        return;
    };

    // Execute
    var interp = interpreter.Interpreter.init(allocator);
    defer interp.deinit();

    if (use_fixpoint) {
        const msg1 = std.fmt.allocPrint(allocator, "Running {s} with fixpoint iteration...\n", .{path}) catch return;
        defer allocator.free(msg1);
        stdout.writeAll(msg1) catch return;

        const iters = interp.executeToFixpoint(&program) catch |err| {
            const err_msg = std.fmt.allocPrint(allocator, "Runtime error: {}\n", .{err}) catch return;
            defer allocator.free(err_msg);
            stdout.writeAll(err_msg) catch return;
            return;
        };

        const msg2 = std.fmt.allocPrint(allocator, "Converged after {d} iteration(s)\n", .{iters}) catch return;
        defer allocator.free(msg2);
        stdout.writeAll(msg2) catch return;
    } else {
        const msg1 = std.fmt.allocPrint(allocator, "Running {s}...\n", .{path}) catch return;
        defer allocator.free(msg1);
        stdout.writeAll(msg1) catch return;

        interp.execute(&program) catch |err| {
            const err_msg = std.fmt.allocPrint(allocator, "Runtime error: {}\n", .{err}) catch return;
            defer allocator.free(err_msg);
            stdout.writeAll(err_msg) catch return;
            return;
        };
    }

    // Print results
    stdout.writeAll("\nDefined tensors:\n") catch return;

    var tensor_iter = interp.tensors.iterator();
    while (tensor_iter.next()) |entry| {
        const name = entry.key_ptr.*;
        const t = entry.value_ptr.*;
        const shape = t.shape();

        var shape_str = std.ArrayListUnmanaged(u8){};
        defer shape_str.deinit(allocator);

        var writer = shape_str.writer(allocator);
        writer.print("[", .{}) catch continue;
        for (shape.dims, 0..) |d, i| {
            if (i > 0) writer.print(", ", .{}) catch continue;
            writer.print("{d}", .{d}) catch continue;
        }
        writer.print("]", .{}) catch continue;

        const line = std.fmt.allocPrint(allocator, "  {s}: {s} {s}\n", .{
            name,
            t.dtype().name(),
            shape_str.items,
        }) catch continue;
        defer allocator.free(line);
        stdout.writeAll(line) catch continue;

        // For small tensors, print non-zero values
        if (shape.numel() <= 100) {
            switch (t) {
                .f64_dense => |dense| {
                    // Print non-zero elements
                    var has_nonzero = false;
                    for (dense.data, 0..) |val, flat_idx| {
                        if (val != 0.0) {
                            if (!has_nonzero) {
                                stdout.writeAll("    Non-zero values:\n") catch continue;
                                has_nonzero = true;
                            }
                            // Convert flat index to multi-index
                            var idx_str = std.ArrayListUnmanaged(u8){};
                            defer idx_str.deinit(allocator);
                            var idx_writer = idx_str.writer(allocator);

                            var remaining = flat_idx;
                            var strides = std.ArrayListUnmanaged(usize){};
                            defer strides.deinit(allocator);

                            // Compute strides
                            var stride: usize = 1;
                            var dim_idx = shape.dims.len;
                            while (dim_idx > 0) {
                                dim_idx -= 1;
                                strides.insert(allocator, 0, stride) catch break;
                                stride *= shape.dims[dim_idx];
                            }

                            idx_writer.print("[", .{}) catch continue;
                            for (strides.items, 0..) |s, di| {
                                if (di > 0) idx_writer.print(",", .{}) catch continue;
                                const coord = remaining / s;
                                remaining = remaining % s;
                                idx_writer.print("{d}", .{coord}) catch continue;
                            }
                            idx_writer.print("]", .{}) catch continue;

                            const val_line = std.fmt.allocPrint(allocator, "      {s} = {d:.1}\n", .{ idx_str.items, val }) catch continue;
                            defer allocator.free(val_line);
                            stdout.writeAll(val_line) catch continue;
                        }
                    }
                },
                else => {},
            }
        }
    }

    stdout.writeAll("\nExecution complete.\n") catch return;
}

test "main module" {
    // Import tests from submodules
    _ = @import("frontend/lexer.zig");
    _ = @import("frontend/tokens.zig");
    _ = @import("frontend/ast.zig");
    _ = @import("frontend/parser.zig");
    _ = @import("runtime/tensor.zig");
    _ = @import("runtime/einsum.zig");
    _ = @import("runtime/interpreter.zig");
}
