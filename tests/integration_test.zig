// Integration tests for Tensor Logic Compiler
// Compiles examples and verifies output

const std = @import("std");
const builtin = @import("builtin");

const TestMode = enum {
    build_and_run, // Full build + run test
    check_only, // Just parse and type check (expect success)
    check_expect_error, // Parse and type check, expect error exit code
};

const TestCase = struct {
    name: []const u8,
    file: []const u8,
    mode: TestMode = .build_and_run,
    expected_contains: []const []const u8 = &.{},
};

const test_cases = [_]TestCase{
    // Full build + run tests
    .{
        .name = "matmul",
        .file = "examples/matmul.tl",
    },
    .{
        .name = "training",
        .file = "examples/training.tl",
    },
    .{
        .name = "mlp",
        .file = "examples/mlp.tl",
    },
    .{
        .name = "ancestor",
        .file = "examples/ancestor.tl",
    },
    // Parse/check only tests (for examples with file I/O or complex features)
    .{
        .name = "attention",
        .file = "examples/attention.tl",
        .mode = .check_only,
    },
    .{
        .name = "gnn",
        .file = "examples/gnn.tl",
        .mode = .check_only,
    },
    .{
        .name = "temperature",
        .file = "examples/temperature.tl",
        .mode = .check_only,
    },
    .{
        .name = "backward_query",
        .file = "examples/backward_query.tl",
        .mode = .check_only,
    },
    .{
        .name = "backward_vs_forward",
        .file = "examples/backward_vs_forward.tl",
        .mode = .check_only,
    },
    .{
        .name = "fileio",
        .file = "examples/fileio.tl",
        .mode = .check_only,
    },
    // Semantic error tests (expect type checker to report errors)
    .{
        .name = "rank_mismatch",
        .file = "tests/semantic/rank_mismatch.tl",
        .mode = .check_expect_error,
    },
    .{
        .name = "div_by_zero",
        .file = "tests/semantic/div_by_zero.tl",
        .mode = .check_expect_error,
    },
    // Semantic success test
    .{
        .name = "rank_consistent",
        .file = "tests/semantic/rank_consistent.tl",
        .mode = .check_only,
    },
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var passed: usize = 0;
    var failed: usize = 0;

    for (test_cases) |tc| {
        const result = runTest(allocator, tc) catch |err| {
            std.debug.print("FAIL {s}: {}\n", .{ tc.name, err });
            failed += 1;
            continue;
        };

        if (result) {
            std.debug.print("PASS {s}\n", .{tc.name});
            passed += 1;
        } else {
            std.debug.print("FAIL {s}: output mismatch\n", .{tc.name});
            failed += 1;
        }
    }

    std.debug.print("\n{d} passed, {d} failed\n", .{ passed, failed });
    if (failed > 0) {
        std.process.exit(1);
    }
}

fn runTest(allocator: std.mem.Allocator, tc: TestCase) !bool {
    switch (tc.mode) {
        .check_only => {
            // Just parse and type check (expect success)
            const check_result = try std.process.Child.run(.{
                .allocator = allocator,
                .argv = &.{ "./zig-out/bin/tlc", "check", tc.file },
            });
            defer allocator.free(check_result.stdout);
            defer allocator.free(check_result.stderr);

            if (check_result.term.Exited != 0) {
                std.debug.print("Check failed: {s}\n", .{check_result.stderr});
                return error.CheckFailed;
            }
            return true;
        },
        .check_expect_error => {
            // Parse and type check, expect error exit code (semantic error tests)
            const check_result = try std.process.Child.run(.{
                .allocator = allocator,
                .argv = &.{ "./zig-out/bin/tlc", "check", tc.file },
            });
            defer allocator.free(check_result.stdout);
            defer allocator.free(check_result.stderr);

            if (check_result.term.Exited == 0) {
                std.debug.print("Expected error but check passed\n", .{});
                return error.ExpectedError;
            }
            // Non-zero exit code is expected
            return true;
        },
        .build_and_run => {
            var tmp_dir = try std.fs.cwd().makeOpenPath("zig-cache/tlc_integration", .{});
            defer tmp_dir.close();

            const out_file_name = if (builtin.os.tag == .windows)
                try std.fmt.allocPrint(allocator, "tlc_test_{s}.exe", .{tc.name})
            else
                try std.fmt.allocPrint(allocator, "tlc_test_{s}", .{tc.name});
            defer allocator.free(out_file_name);

            const out_path = try std.fs.path.join(allocator, &.{ "zig-cache/tlc_integration", out_file_name });
            defer allocator.free(out_path);

            // Build the program
            const build_result = try std.process.Child.run(.{
                .allocator = allocator,
                .argv = &.{ "./zig-out/bin/tlc", "build", tc.file, "-o", out_path },
            });
            defer allocator.free(build_result.stdout);
            defer allocator.free(build_result.stderr);

            if (build_result.term.Exited != 0) {
                std.debug.print("Build failed: {s}\n", .{build_result.stderr});
                return error.BuildFailed;
            }

            // Run the program
            const run_result = try std.process.Child.run(.{
                .allocator = allocator,
                .argv = &.{out_path},
            });
            defer allocator.free(run_result.stdout);
            defer allocator.free(run_result.stderr);

            if (run_result.term.Exited != 0) {
                std.debug.print("Run failed: {s}\n", .{run_result.stderr});
                return error.RunFailed;
            }

            // Check expected output
            for (tc.expected_contains) |expected| {
                if (std.mem.indexOf(u8, run_result.stdout, expected) == null) {
                    std.debug.print("Missing expected output: {s}\n", .{expected});
                    std.debug.print("Got: {s}\n", .{run_result.stdout});
                    return false;
                }
            }

            std.fs.cwd().deleteFile(out_path) catch {};

            return true;
        },
    }
}

test "integration tests" {
    // This allows running via `zig build test`
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    for (test_cases) |tc| {
        const result = runTest(allocator, tc) catch |err| {
            std.debug.print("Test {s} failed: {}\n", .{ tc.name, err });
            continue;
        };
        try std.testing.expect(result);
    }
}
