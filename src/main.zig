// Tensor Logic Compiler (tlc)
// A compiler for the Tensor Logic language based on Pedro Domingos' paper
// "The sole construct in tensor logic is the tensor equation"

const std = @import("std");
const lexer = @import("frontend/lexer.zig");
const parser = @import("frontend/parser.zig");
const ast = @import("frontend/ast.zig");
const types = @import("frontend/types.zig");
const llvm_codegen = @import("codegen/llvm.zig");

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
    } else if (std.mem.eql(u8, command, "check")) {
        if (args.len < 3) {
            std.debug.print("Error: 'check' command requires a file path\n", .{});
            return;
        }
        try checkFile(allocator, args[2]);
    } else if (std.mem.eql(u8, command, "compile")) {
        if (args.len < 3) {
            std.debug.print("Error: 'compile' command requires a file path\n", .{});
            return;
        }
        // Parse compile flags: -o <output>, --emit-llvm
        var output_path: ?[]const u8 = null;
        var file_path: ?[]const u8 = null;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            const arg = args[i];
            if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
                i += 1;
                if (i < args.len) {
                    output_path = args[i];
                }
            } else if (std.mem.eql(u8, arg, "--emit-llvm")) {
                // Default behavior, just emit LLVM IR
            } else if (file_path == null) {
                file_path = arg;
            }
        }
        if (file_path) |path| {
            try compileFile(allocator, path, output_path);
        } else {
            std.debug.print("Error: 'compile' command requires a file path\n", .{});
        }
    } else if (std.mem.eql(u8, command, "build")) {
        if (args.len < 3) {
            std.debug.print("Error: 'build' command requires a file path\n", .{});
            return;
        }
        // Parse build flags: -o <output>
        var output_path: ?[]const u8 = null;
        var file_path: ?[]const u8 = null;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            const arg = args[i];
            if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
                i += 1;
                if (i < args.len) {
                    output_path = args[i];
                }
            } else if (file_path == null) {
                file_path = arg;
            }
        }
        if (file_path) |path| {
            try buildFile(allocator, path, output_path);
        } else {
            std.debug.print("Error: 'build' command requires a file path\n", .{});
        }
    } else if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        try printUsage();
    } else if (std.mem.eql(u8, command, "version") or std.mem.eql(u8, command, "--version") or std.mem.eql(u8, command, "-v")) {
        try printVersion();
    } else {
        // Assume it's a file path - compile it
        try compileFile(allocator, command, null);
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
        \\  build <file> -o out  Build native executable (requires clang)
        \\  compile <file>       Compile to LLVM IR (stdout)
        \\  compile <file> -o f  Compile and write LLVM IR to file
        \\  check <file>         Type check a .tl file
        \\  lex <file>           Tokenize a .tl file
        \\  parse <file>         Parse a .tl file
        \\  help                 Show this help message
        \\  version              Show version information
        \\
        \\Examples:
        \\  tlc build examples/matmul.tl -o matmul && ./matmul
        \\  tlc compile examples/matmul.tl -o matmul.ll
        \\  tlc check examples/ancestor.tl
        \\
    );
}

fn printVersion() !void {
    const stdout = std.fs.File.stdout();
    try stdout.writeAll("tlc version 0.1.0\n");
}

fn buildFile(allocator: std.mem.Allocator, path: []const u8, output_path: ?[]const u8) !void {
    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024) catch |err| {
        switch (err) {
            error.FileNotFound => {
                std.debug.print("Error: file not found: '{s}'\n", .{path});
                std.debug.print("Check that the path is correct and the file exists.\n", .{});
            },
            error.AccessDenied => {
                std.debug.print("Error: permission denied: '{s}'\n", .{path});
            },
            else => {
                std.debug.print("Error reading file '{s}': {}\n", .{ path, err });
            },
        }
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

    // Generate LLVM IR
    const llvm_ir = llvm_codegen.compile(allocator, &program) catch |err| {
        std.debug.print("Code generation error: {}\n", .{err});
        return;
    };
    defer allocator.free(llvm_ir);

    // Determine output name
    const is_windows = @import("builtin").os.tag == .windows;
    const out_name = output_path orelse blk: {
        // Strip .tl extension if present, otherwise use default
        if (std.mem.endsWith(u8, path, ".tl")) {
            break :blk path[0 .. path.len - 3];
        }
        break :blk if (is_windows) "a.exe" else "a.out";
    };

    // Write LLVM IR to temp file (cross-platform)
    const tmp_dir = if (@import("builtin").os.tag == .windows)
        std.process.getEnvVarOwned(allocator, "TEMP") catch std.process.getEnvVarOwned(allocator, "TMP") catch null
    else
        null;
    defer if (tmp_dir) |dir| allocator.free(dir);

    const tmp_prefix = tmp_dir orelse "/tmp";
    const tmp_ll = std.fmt.allocPrint(allocator, "{s}/tlc_{d}.ll", .{ tmp_prefix, std.time.milliTimestamp() }) catch {
        std.debug.print("Error: out of memory\n", .{});
        return;
    };
    defer allocator.free(tmp_ll);

    const tmp_file = std.fs.createFileAbsolute(tmp_ll, .{}) catch |err| {
        std.debug.print("Error creating temp file: {}\n", .{err});
        return;
    };
    tmp_file.writeAll(llvm_ir) catch |err| {
        std.debug.print("Error writing temp file: {}\n", .{err});
        tmp_file.close();
        return;
    };
    tmp_file.close();

    // Invoke clang (Windows doesn't need -lm)
    const clang_args = if (is_windows)
        &[_][]const u8{ "clang", tmp_ll, "-o", out_name }
    else
        &[_][]const u8{ "clang", tmp_ll, "-o", out_name, "-lm" };

    var child = std.process.Child.init(clang_args, allocator);

    child.spawn() catch |err| {
        std.debug.print("Error: failed to run clang: {}\n", .{err});
        std.debug.print("\nClang is required to build native executables.\n", .{});
        std.debug.print("Install clang:\n", .{});
        std.debug.print("  macOS:   xcode-select --install\n", .{});
        std.debug.print("  Ubuntu:  sudo apt install clang\n", .{});
        std.debug.print("  Windows: winget install LLVM.LLVM\n", .{});
        std.debug.print("\nAlternatively, use 'tlc compile' to emit LLVM IR.\n", .{});
        std.fs.deleteFileAbsolute(tmp_ll) catch {};
        return;
    };

    const result = child.wait() catch |err| {
        std.debug.print("Error waiting for clang: {}\n", .{err});
        std.fs.deleteFileAbsolute(tmp_ll) catch {};
        return;
    };

    // Clean up temp file
    std.fs.deleteFileAbsolute(tmp_ll) catch {};

    if (result.Exited == 0) {
        std.debug.print("Built {s} -> {s}\n", .{ path, out_name });
    } else {
        std.debug.print("Error: clang failed with exit code {}\n", .{result.Exited});
    }
}

fn compileFile(allocator: std.mem.Allocator, path: []const u8, output_path: ?[]const u8) !void {
    const stdout = std.fs.File.stdout();

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024) catch |err| {
        switch (err) {
            error.FileNotFound => {
                std.debug.print("Error: file not found: '{s}'\n", .{path});
                std.debug.print("Check that the path is correct and the file exists.\n", .{});
            },
            error.AccessDenied => {
                std.debug.print("Error: permission denied: '{s}'\n", .{path});
            },
            else => {
                std.debug.print("Error reading file '{s}': {}\n", .{ path, err });
            },
        }
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

    // Generate LLVM IR
    const llvm_ir = llvm_codegen.compile(allocator, &program) catch |err| {
        std.debug.print("Code generation error: {}\n", .{err});
        return;
    };
    defer allocator.free(llvm_ir);

    // Output
    if (output_path) |out_path| {
        // Write to file
        const file = std.fs.cwd().createFile(out_path, .{}) catch |err| {
            std.debug.print("Error creating output file '{s}': {}\n", .{ out_path, err });
            return;
        };
        defer file.close();

        file.writeAll(llvm_ir) catch |err| {
            std.debug.print("Error writing to '{s}': {}\n", .{ out_path, err });
            return;
        };

        std.debug.print("Compiled {s} -> {s} ({d} bytes)\n", .{ path, out_path, llvm_ir.len });
        std.debug.print("To build: clang {s} -o program -lm\n", .{out_path});
    } else {
        // Write to stdout
        stdout.writeAll(llvm_ir) catch {};
    }
}

fn lexFile(allocator: std.mem.Allocator, path: []const u8) !void {
    const source = std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024) catch |err| {
        switch (err) {
            error.FileNotFound => {
                std.debug.print("Error: file not found: '{s}'\n", .{path});
                std.debug.print("Check that the path is correct and the file exists.\n", .{});
            },
            error.AccessDenied => {
                std.debug.print("Error: permission denied: '{s}'\n", .{path});
            },
            else => {
                std.debug.print("Error reading file '{s}': {}\n", .{ path, err });
            },
        }
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
        switch (err) {
            error.FileNotFound => {
                std.debug.print("Error: file not found: '{s}'\n", .{path});
                std.debug.print("Check that the path is correct and the file exists.\n", .{});
            },
            error.AccessDenied => {
                std.debug.print("Error: permission denied: '{s}'\n", .{path});
            },
            else => {
                std.debug.print("Error reading file '{s}': {}\n", .{ path, err });
            },
        }
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
            .tucker_decl => |t| {
                const msg = std.fmt.allocPrint(allocator, "Tucker: {s} ({d} ranks)\n", .{ t.name, t.core_ranks.len }) catch continue;
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
            .query => |q| {
                const msg = std.fmt.allocPrint(allocator, "Query: {s}?\n", .{q.tensor.name}) catch continue;
                defer allocator.free(msg);
                stdout.writeAll(msg) catch continue;
            },
            .save_stmt => |s| {
                const msg = std.fmt.allocPrint(allocator, "Save: {s} -> {s}\n", .{ s.tensor_name, s.path }) catch continue;
                defer allocator.free(msg);
                stdout.writeAll(msg) catch continue;
            },
            .load_stmt => |l| {
                const msg = std.fmt.allocPrint(allocator, "Load: {s} <- {s}\n", .{ l.tensor_name, l.path }) catch continue;
                defer allocator.free(msg);
                stdout.writeAll(msg) catch continue;
            },
            .backward_stmt => |b| {
                const msg = std.fmt.allocPrint(allocator, "Backward: {s} wrt {d} params\n", .{ b.loss, b.params.len }) catch continue;
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
        switch (err) {
            error.FileNotFound => {
                std.debug.print("Error: file not found: '{s}'\n", .{path});
                std.debug.print("Check that the path is correct and the file exists.\n", .{});
            },
            error.AccessDenied => {
                std.debug.print("Error: permission denied: '{s}'\n", .{path});
            },
            else => {
                std.debug.print("Error reading file '{s}': {}\n", .{ path, err });
            },
        }
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

    // Print any errors/warnings
    if (checker.env.errors.items.len > 0) {
        stdout.writeAll("\nDiagnostics:\n") catch return;

        // Use enhanced error formatting
        const formatted = checker.formatErrors(allocator) catch return;
        defer allocator.free(formatted);
        stdout.writeAll(formatted) catch return;

        // Print summary
        const counts = checker.countBySeverity();
        const summary = std.fmt.allocPrint(allocator, "\n{d} error(s), {d} warning(s)\n", .{
            counts.errors,
            counts.warnings,
        }) catch return;
        defer allocator.free(summary);
        stdout.writeAll(summary) catch return;

        // Exit with error if there are hard errors
        if (counts.errors > 0) {
            std.process.exit(1);
        }
    } else {
        stdout.writeAll("\nNo type errors.\n") catch return;
    }
}

test "main module" {
    // Import tests from submodules
    _ = @import("frontend/lexer.zig");
    _ = @import("frontend/tokens.zig");
    _ = @import("frontend/ast.zig");
    _ = @import("frontend/parser.zig");
}
