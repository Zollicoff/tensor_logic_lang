// Tensor Logic Compiler (tlc)
// A compiler for the Tensor Logic language based on Pedro Domingos' paper

const std = @import("std");
const lexer = @import("frontend/lexer.zig");
const parser = @import("frontend/parser.zig");
const ast = @import("frontend/ast.zig");
const types = @import("frontend/types.zig");
const interpreter = @import("runtime/interpreter.zig");
const tensor_mod = @import("runtime/tensor.zig");

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
    } else if (std.mem.eql(u8, command, "repl")) {
        try runRepl(allocator);
    } else if (std.mem.eql(u8, command, "check")) {
        if (args.len < 3) {
            std.debug.print("Error: 'check' command requires a file path\n", .{});
            return;
        }
        try checkFile(allocator, args[2]);
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
        \\  repl                 Start interactive REPL (Read-Eval-Print Loop)
        \\  lex <file>           Tokenize a .tl file and print tokens
        \\  parse <file>         Parse a .tl file and print AST summary
        \\  check <file>         Type check a .tl file (Boolean vs Real inference)
        \\  run <file>           Execute a .tl program (single pass)
        \\  run -f <file>        Execute with fixpoint iteration (for recursive rules)
        \\  help                 Show this help message
        \\  version              Show version information
        \\
        \\Options:
        \\  -f, --fixpoint       Run until convergence (for recursive rules like Ancestor)
        \\
        \\Examples:
        \\  tlc repl
        \\  tlc lex examples/matmul.tl
        \\  tlc parse examples/ancestor.tl
        \\  tlc run examples/matmul.tl
        \\  tlc run -f examples/ancestor.tl
        \\
        \\REPL Commands:
        \\  :help                Show REPL help
        \\  :show [name]         Show tensor values (all or by name)
        \\  :clear               Clear all tensors and domains
        \\  :fixpoint            Run fixpoint iteration on current state
        \\  :quit                Exit REPL
        \\
    );
}

fn printVersion() !void {
    const stdout = std.fs.File.stdout();
    try stdout.writeAll("tlc version 0.1.0\n");
}

fn runRepl(allocator: std.mem.Allocator) !void {
    const stdout = std.fs.File.stdout();
    const stdin_file = std.fs.File.stdin();

    // Print banner
    stdout.writeAll(
        \\Tensor Logic REPL v0.1.0
        \\Type :help for commands, :quit to exit
        \\
        \\
    ) catch {};

    // Create interpreter that persists across inputs
    var interp = interpreter.Interpreter.init(allocator);
    defer interp.deinit();

    // Track all arena allocators for cleanup
    var arenas = std.ArrayListUnmanaged(*std.heap.ArenaAllocator){};
    defer {
        for (arenas.items) |arena| {
            arena.deinit();
            allocator.destroy(arena);
        }
        arenas.deinit(allocator);
    }

    // Input buffer for stdin reader
    var read_buf: [4096]u8 = undefined;
    var stdin_reader = stdin_file.reader(&read_buf);

    while (true) {
        stdout.writeAll("tl> ") catch {};

        // Read line using Zig 0.15's new API
        const line = stdin_reader.interface.takeDelimiterExclusive('\n') catch |err| {
            switch (err) {
                error.EndOfStream => {
                    // EOF
                    stdout.writeAll("\nGoodbye!\n") catch {};
                    break;
                },
                else => {
                    printLine(allocator, stdout, "Input error");
                    continue;
                },
            }
        };

        // Trim whitespace
        const trimmed = std.mem.trim(u8, line, " \t\r\n");

        if (trimmed.len == 0) continue;

        // Handle REPL commands
        if (trimmed[0] == ':') {
            if (std.mem.eql(u8, trimmed, ":quit") or std.mem.eql(u8, trimmed, ":q")) {
                stdout.writeAll("Goodbye!\n") catch {};
                break;
            } else if (std.mem.eql(u8, trimmed, ":help") or std.mem.eql(u8, trimmed, ":h")) {
                printReplHelp(stdout);
            } else if (std.mem.eql(u8, trimmed, ":clear")) {
                interp.deinit();
                interp = interpreter.Interpreter.init(allocator);
                stdout.writeAll("Cleared all tensors and domains.\n") catch {};
            } else if (std.mem.eql(u8, trimmed, ":fixpoint") or std.mem.eql(u8, trimmed, ":fp")) {
                runFixpointOnCurrentState(&interp, allocator, stdout);
            } else if (std.mem.startsWith(u8, trimmed, ":show")) {
                const arg = std.mem.trim(u8, trimmed[5..], " \t");
                showTensors(&interp, allocator, stdout, if (arg.len > 0) arg else null);
            } else if (std.mem.startsWith(u8, trimmed, ":load ")) {
                const path = std.mem.trim(u8, trimmed[6..], " \t");
                loadFile(&interp, allocator, &arenas, path, stdout);
            } else {
                printFmt(allocator, stdout, "Unknown command: {s}\n", .{trimmed});
                stdout.writeAll("Type :help for available commands.\n") catch {};
            }
            continue;
        }

        // Parse and execute tensor logic statement
        // Create new arena for this statement's AST
        const arena = allocator.create(std.heap.ArenaAllocator) catch continue;
        arena.* = std.heap.ArenaAllocator.init(allocator);
        arenas.append(allocator, arena) catch continue;
        const ast_allocator = arena.allocator();

        // Make a copy of the input for lexer
        const source = ast_allocator.dupe(u8, trimmed) catch continue;

        // Lex
        var lex = lexer.Lexer.init(ast_allocator, source);

        const tokens = lex.scanTokens() catch |err| {
            printFmt(allocator, stdout, "Lexer error: {}\n", .{err});
            continue;
        };

        // Parse
        var p = parser.Parser.init(ast_allocator, tokens);

        const program = p.parse() catch |err| {
            if (p.getLastError()) |parse_err| {
                printFmt(allocator, stdout, "error at {d}:{d}: {s}\n", .{ parse_err.location.line, parse_err.location.column, parse_err.message });
            } else {
                printFmt(allocator, stdout, "Parse error: {}\n", .{err});
            }
            continue;
        };

        // Execute
        interp.execute(&program) catch |err| {
            printFmt(allocator, stdout, "Runtime error: {}\n", .{err});
            continue;
        };

        // Print brief confirmation
        for (program.statements) |stmt| {
            switch (stmt) {
                .equation => |eq| {
                    printFmt(allocator, stdout, "  {s} defined\n", .{eq.lhs.name});
                },
                .domain_decl => |d| {
                    printFmt(allocator, stdout, "  domain {s} declared\n", .{d.name});
                },
                .sparse_decl => |s| {
                    printFmt(allocator, stdout, "  sparse {s} declared\n", .{s.name});
                },
                .import_stmt => |im| {
                    printFmt(allocator, stdout, "  imported {s}\n", .{im.path});
                },
                .export_stmt => |ex| {
                    printFmt(allocator, stdout, "  exported {s}\n", .{ex.name});
                },
                .comment => {},
            }
        }
    }
}

// Helper to format and print a string
fn printFmt(allocator: std.mem.Allocator, file: std.fs.File, comptime fmt: []const u8, args: anytype) void {
    const msg = std.fmt.allocPrint(allocator, fmt, args) catch return;
    defer allocator.free(msg);
    file.writeAll(msg) catch {};
}

fn printLine(allocator: std.mem.Allocator, file: std.fs.File, msg: []const u8) void {
    _ = allocator;
    file.writeAll(msg) catch {};
    file.writeAll("\n") catch {};
}

fn printReplHelp(file: std.fs.File) void {
    file.writeAll(
        \\REPL Commands:
        \\  :help, :h            Show this help
        \\  :show [name]         Show tensor values (all or by name)
        \\  :clear               Clear all tensors and domains
        \\  :fixpoint, :fp       Run fixpoint iteration
        \\  :load <file>         Load and execute a .tl file
        \\  :quit, :q            Exit REPL
        \\
        \\Tensor Logic Syntax:
        \\  domain i: 10         Declare index domain
        \\  A[i,j] = 0           Initialize tensor
        \\  A[0,1] = 1           Set element
        \\  C[i,k] = A[i,j] B[j,k]   Matrix multiply (Einstein sum)
        \\  Y[i] = softmax(X[i])     Apply nonlinearity
        \\  A[i,k] max= B[i,j] C[j,k]  Accumulate with max
        \\
        \\
    ) catch {};
}

fn showTensors(interp: *interpreter.Interpreter, allocator: std.mem.Allocator, file: std.fs.File, name_filter: ?[]const u8) void {
    if (interp.tensors.count() == 0) {
        file.writeAll("No tensors defined.\n") catch {};
        return;
    }

    var tensor_iter = interp.tensors.iterator();
    while (tensor_iter.next()) |entry| {
        const name = entry.key_ptr.*;
        const t = entry.value_ptr.*;

        // Filter by name if specified
        if (name_filter) |filter| {
            if (!std.mem.eql(u8, name, filter)) continue;
        }

        const shape = t.shape();

        // Build shape string using fixed buffer
        var shape_buf: [256]u8 = undefined;
        var shape_len: usize = 0;
        shape_buf[shape_len] = '[';
        shape_len += 1;
        for (shape.dims, 0..) |d, i| {
            if (i > 0) {
                if (shape_len + 2 < shape_buf.len) {
                    shape_buf[shape_len] = ',';
                    shape_buf[shape_len + 1] = ' ';
                    shape_len += 2;
                }
            }
            const num_str = std.fmt.bufPrint(shape_buf[shape_len..], "{d}", .{d}) catch break;
            shape_len += num_str.len;
        }
        if (shape_len < shape_buf.len) {
            shape_buf[shape_len] = ']';
            shape_len += 1;
        }
        const shape_str = shape_buf[0..shape_len];

        // Calculate sparsity stats
        const nnz = tensor_mod.countNonZero(&t);
        const total = shape.numel();
        const sparsity_pct = if (total > 0)
            100.0 * (1.0 - @as(f64, @floatFromInt(nnz)) / @as(f64, @floatFromInt(total)))
        else
            0.0;

        printFmt(allocator, file, "{s}: {s} {s} ({d} nnz, {d:.1}% sparse)\n", .{
            name,
            t.dtype().name(),
            shape_str,
            nnz,
            sparsity_pct,
        });

        // For small tensors, print non-zero values
        if (total <= 100) {
            printNonZeroValues(t, shape, allocator, file);
        }
    }

    if (name_filter) |filter| {
        if (!interp.tensors.contains(filter)) {
            printFmt(allocator, file, "Tensor '{s}' not found.\n", .{filter});
        }
    }
}

fn printNonZeroValues(t: tensor_mod.Tensor, shape: tensor_mod.Shape, allocator: std.mem.Allocator, file: std.fs.File) void {
    switch (t) {
        .f64_dense => |dense| {
            var has_nonzero = false;
            // Pre-compute strides
            var strides_buf: [16]usize = undefined;
            var stride: usize = 1;
            var dim_idx = shape.dims.len;
            while (dim_idx > 0) {
                dim_idx -= 1;
                strides_buf[dim_idx] = stride;
                stride *= shape.dims[dim_idx];
            }
            const strides = strides_buf[0..shape.dims.len];

            for (dense.data, 0..) |val, flat_idx| {
                if (val != 0.0) {
                    if (!has_nonzero) {
                        file.writeAll("  Values:\n") catch {};
                        has_nonzero = true;
                    }
                    // Convert flat index to multi-index using fixed buffer
                    var idx_buf: [128]u8 = undefined;
                    var idx_len: usize = 0;
                    idx_buf[idx_len] = '[';
                    idx_len += 1;

                    var remaining = flat_idx;
                    for (strides, 0..) |s, di| {
                        if (di > 0) {
                            idx_buf[idx_len] = ',';
                            idx_len += 1;
                        }
                        const coord = remaining / s;
                        remaining = remaining % s;
                        const num_str = std.fmt.bufPrint(idx_buf[idx_len..], "{d}", .{coord}) catch break;
                        idx_len += num_str.len;
                    }
                    idx_buf[idx_len] = ']';
                    idx_len += 1;

                    printFmt(allocator, file, "    {s} = {d:.4}\n", .{ idx_buf[0..idx_len], val });
                }
            }
        },
        .f64_sparse => |sparse| {
            if (sparse.values.items.len > 0) {
                file.writeAll("  Values:\n") catch {};
                for (sparse.indices.items, sparse.values.items) |indices, val| {
                    var idx_buf: [128]u8 = undefined;
                    var idx_len: usize = 0;
                    idx_buf[idx_len] = '[';
                    idx_len += 1;

                    for (indices, 0..) |idx, i| {
                        if (i > 0) {
                            idx_buf[idx_len] = ',';
                            idx_len += 1;
                        }
                        const num_str = std.fmt.bufPrint(idx_buf[idx_len..], "{d}", .{idx}) catch break;
                        idx_len += num_str.len;
                    }
                    idx_buf[idx_len] = ']';
                    idx_len += 1;

                    printFmt(allocator, file, "    {s} = {d:.4}\n", .{ idx_buf[0..idx_len], val });
                }
            }
        },
        else => {},
    }
}

fn runFixpointOnCurrentState(interp: *interpreter.Interpreter, allocator: std.mem.Allocator, file: std.fs.File) void {
    if (interp.last_program) |program| {
        const iters = interp.executeToFixpoint(program) catch |err| {
            printFmt(allocator, file, "Fixpoint error: {}\n", .{err});
            return;
        };
        printFmt(allocator, file, "Converged after {d} iteration(s).\n", .{iters});
    } else {
        file.writeAll("No program to iterate. Enter some equations first.\n") catch {};
    }
}

fn loadFile(interp: *interpreter.Interpreter, allocator: std.mem.Allocator, arenas: *std.ArrayListUnmanaged(*std.heap.ArenaAllocator), path: []const u8, file: std.fs.File) void {
    const source = std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024) catch |err| {
        printFmt(allocator, file, "Error reading file '{s}': {}\n", .{ path, err });
        return;
    };

    // Create arena for AST
    const arena = allocator.create(std.heap.ArenaAllocator) catch {
        allocator.free(source);
        file.writeAll("Out of memory\n") catch {};
        return;
    };
    arena.* = std.heap.ArenaAllocator.init(allocator);
    arenas.append(allocator, arena) catch {
        allocator.free(source);
        file.writeAll("Out of memory\n") catch {};
        return;
    };
    const ast_allocator = arena.allocator();

    // Copy source to arena so it lives long enough
    const source_copy = ast_allocator.dupe(u8, source) catch {
        allocator.free(source);
        file.writeAll("Out of memory\n") catch {};
        return;
    };
    allocator.free(source);

    // Lex
    var lex = lexer.Lexer.init(ast_allocator, source_copy);

    const tokens = lex.scanTokens() catch |err| {
        printFmt(allocator, file, "Lexer error: {}\n", .{err});
        return;
    };

    // Parse
    var p = parser.Parser.init(ast_allocator, tokens);

    const program = p.parse() catch |err| {
        if (p.getLastError()) |parse_err| {
            printFmt(allocator, file, "error at {d}:{d}: {s}\n", .{ parse_err.location.line, parse_err.location.column, parse_err.message });
        } else {
            printFmt(allocator, file, "Parse error: {}\n", .{err});
        }
        return;
    };

    // Execute
    interp.execute(&program) catch |err| {
        printFmt(allocator, file, "Runtime error: {}\n", .{err});
        return;
    };

    printFmt(allocator, file, "Loaded {s}: {d} statement(s)\n", .{ path, program.statements.len });
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
        if (p.getLastError()) |parse_err| {
            std.debug.print("{s}:{d}:{d}: error: {s}\n", .{ path, parse_err.location.line, parse_err.location.column, parse_err.message });
        } else {
            std.debug.print("Parse error: {}\n", .{err});
        }
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

fn checkFile(allocator: std.mem.Allocator, path: []const u8) !void {
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
        if (p.getLastError()) |parse_err| {
            std.debug.print("{s}:{d}:{d}: error: {s}\n", .{ path, parse_err.location.line, parse_err.location.column, parse_err.message });
        } else {
            std.debug.print("Parse error: {}\n", .{err});
        }
        return;
    };

    // Run type checker
    var checker = types.TypeChecker.init(allocator);
    defer checker.deinit();

    checker.check(&program) catch |err| {
        std.debug.print("Type check error: {}\n", .{err});
        return;
    };

    const stdout = std.fs.File.stdout();

    // Print type information for all tensors
    const header = std.fmt.allocPrint(allocator, "Type checking {s}...\n\n", .{path}) catch return;
    defer allocator.free(header);
    stdout.writeAll(header) catch return;

    // Print inferred types
    stdout.writeAll("Inferred tensor types:\n") catch return;
    var iter = checker.env.tensor_types.iterator();
    while (iter.next()) |entry| {
        const type_str = entry.value_ptr.value_type.format();
        const sparse_str: []const u8 = if (entry.value_ptr.is_sparse) " (sparse)" else "";
        const msg = std.fmt.allocPrint(allocator, "  {s}: {s}{s}\n", .{ entry.key_ptr.*, type_str, sparse_str }) catch continue;
        defer allocator.free(msg);
        stdout.writeAll(msg) catch continue;
    }

    // Print any warnings
    if (checker.env.errors.items.len > 0) {
        stdout.writeAll("\nType warnings:\n") catch return;
        for (checker.env.errors.items) |err| {
            const msg = std.fmt.allocPrint(allocator, "  {s}:{d}:{d}: warning: {s}\n", .{
                path,
                err.location.line,
                err.location.column,
                err.message,
            }) catch continue;
            defer allocator.free(msg);
            stdout.writeAll(msg) catch continue;
        }
    } else {
        stdout.writeAll("\nNo type warnings.\n") catch return;
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
        if (p.getLastError()) |parse_err| {
            std.debug.print("{s}:{d}:{d}: error: {s}\n", .{ path, parse_err.location.line, parse_err.location.column, parse_err.message });
        } else {
            std.debug.print("Parse error: {}\n", .{err});
        }
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

        // Calculate sparsity stats
        const nnz = tensor_mod.countNonZero(&t);
        const total = shape.numel();
        const sparsity_pct = if (total > 0)
            100.0 * (1.0 - @as(f64, @floatFromInt(nnz)) / @as(f64, @floatFromInt(total)))
        else
            0.0;

        const line = std.fmt.allocPrint(allocator, "  {s}: {s} {s} ({d} nnz, {d:.1}% sparse)\n", .{
            name,
            t.dtype().name(),
            shape_str.items,
            nnz,
            sparsity_pct,
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
