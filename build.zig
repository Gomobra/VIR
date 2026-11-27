const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "vir",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    // Library module for tests
    const lib_mod = b.addModule("vir", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // GPU/PTX integration (PRD 3.3): compile Zig kernels to PTX (sm_75).
    const nvptx_target = b.resolveTargetQuery(.{
        .cpu_arch = .nvptx64,
        .os_tag = .cuda,
        .abi = .none,
    });
    const elem_kernel = b.addObject(.{
        .name = "elementwise_kernel",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/kernels/elementwise.zig"),
            .target = nvptx_target,
            .optimize = .ReleaseFast,
        }),
    });
    const matmul_kernel = b.addObject(.{
        .name = "matmul_kernel",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/kernels/matmul.zig"),
            .target = nvptx_target,
            .optimize = .ReleaseFast,
        }),
    });
    const elem_ptx = elem_kernel.getEmittedAsm();
    const matmul_ptx = matmul_kernel.getEmittedAsm();

    // Embed PTX into host/test modules.
    exe.root_module.addAnonymousImport("elementwise_ptx", .{ .root_source_file = elem_ptx });
    exe.root_module.addAnonymousImport("matmul_ptx", .{ .root_source_file = matmul_ptx });
    lib_mod.addAnonymousImport("elementwise_ptx", .{ .root_source_file = elem_ptx });
    lib_mod.addAnonymousImport("matmul_ptx", .{ .root_source_file = matmul_ptx });

    // Unit tests
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    tests.root_module.addAnonymousImport("elementwise_ptx", .{ .root_source_file = elem_ptx });
    tests.root_module.addAnonymousImport("matmul_ptx", .{ .root_source_file = matmul_ptx });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // PTX build step
    const ptx_step = b.step("ptx", "Compile GPU kernels to PTX");
    ptx_step.dependOn(&elem_kernel.step);
    ptx_step.dependOn(&matmul_kernel.step);

}
