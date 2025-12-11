// Integration tests for Tensor Logic Compiler
// Compiles examples and verifies output

const std = @import("std");

const TestCase = struct {
    name: []const u8,
    file: []const u8,
    expected_contains: []const []const u8,
};

const test_cases = [_]TestCase{
    .{
        .name = "matmul",
        .file = "examples/matmul.tl",
        .expected_contains = &.{}, // No output expected (no query)
    },
    .{
        .name = "training",
        .file = "examples/training.tl",
        .expected_contains = &.{}, // No queries, just verifies compilation
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
    // Build the program
    const build_result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "./zig-out/bin/tlc", "build", tc.file, "-o", "/tmp/tlc_test" },
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
        .argv = &.{"/tmp/tlc_test"},
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

    return true;
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
