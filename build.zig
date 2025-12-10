const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main executable: tlc (Tensor Logic Compiler)
    const exe = b.addExecutable(.{
        .name = "tlc",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    // Create frontend module for reuse
    const frontend_module = b.createModule(.{
        .root_source_file = b.path("src/frontend/frontend.zig"),
        .target = target,
        .optimize = optimize,
    });

    // LSP Server executable: tlc-lsp
    const lsp_module = b.createModule(.{
        .root_source_file = b.path("src/lsp/server.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "frontend", .module = frontend_module },
        },
    });
    const lsp_exe = b.addExecutable(.{
        .name = "tlc-lsp",
        .root_module = lsp_module,
    });
    b.installArtifact(lsp_exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the Tensor Logic compiler");
    run_step.dependOn(&run_cmd.step);

    // Unit tests
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Lexer tests
    const lexer_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/frontend/lexer.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_lexer_tests = b.addRunArtifact(lexer_tests);
    test_step.dependOn(&run_lexer_tests.step);

    // Type checker tests
    const type_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/frontend/types.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_type_tests = b.addRunArtifact(type_tests);
    test_step.dependOn(&run_type_tests.step);

    // AST optimizer tests
    const ast_opt_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/frontend/optimize.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_ast_opt_tests = b.addRunArtifact(ast_opt_tests);
    test_step.dependOn(&run_ast_opt_tests.step);

    // Integration tests (requires tlc to be built first)
    const integration_exe = b.addExecutable(.{
        .name = "integration_test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/integration_test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_integration = b.addRunArtifact(integration_exe);
    run_integration.step.dependOn(b.getInstallStep()); // Depends on tlc being built
    const integration_step = b.step("integration", "Run integration tests");
    integration_step.dependOn(&run_integration.step);
}
