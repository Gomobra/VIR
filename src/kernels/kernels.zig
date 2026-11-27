// PTX kernels (PRD 3.3). Minimal placeholder; real kernels added per tasklist.

const std = @import("std");

// A tiny PTX module string (vector add) for reference.
pub const vector_add_ptx =
    \\.version 7.5
    \\.target sm_75
    \\.address_size 64
    \\
    \\.entry vecadd(.param .u64 a, .param .u64 b, .param .u64 c, .param .u32 n) {
    \\  .reg .pred p;
    \\  .reg .f32 x, y, z;
    \\  .reg .b64 pa, pb, pc;
    \\  .reg .u32 i, nreg;
    \\  ld.param.u64 pa, [a];
    \\  ld.param.u64 pb, [b];
    \\  ld.param.u64 pc, [c];
    \\  ld.param.u32 nreg, [n];
    \\  mov.u32 i, %tid.x;
    \\  // simplified: one thread per element (no loop)
    \\  ret;
    \\}
;

test "vector_add_ptx nonempty" {
    try std.testing.expect(vector_add_ptx.len > 0);
}

// Tiled matmul PTX kernel (reference, sm_75). Simplified 16x16 tile.
pub const matmul_tiled_ptx =
    \\.version 7.5
    \\.target sm_75
    \\.address_size 64
    \\
    \\.entry matmul_tiled(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 M, .param .u32 N, .param .u32 K) {
    \\  .reg .b32 %r<10>;
    \\  .reg .b64 %rd<10>;
    \\  .reg .f32 %f<4>;
    \\  // simplified: one thread computes one element
    \\  ld.param.u64 %rd0, [A];
    \\  ld.param.u64 %rd1, [B];
    \\  ld.param.u64 %rd2, [C];
    \\  ld.param.u32 %r0, [M];
    \\  ld.param.u32 %r1, [N];
    \\  ld.param.u32 %r2, [K];
    \\  mov.u32 %r3, %tid.x;
    \\  mov.u32 %r4, %ctaid.x;
    \\  // row = blockIdx.y * blockDim.y + threadIdx.y
    \\  // col = blockIdx.x * blockDim.x + threadIdx.x
    \\  ret;
    \\}
;

test "matmul_tiled_ptx nonempty" {
    try std.testing.expect(matmul_tiled_ptx.len > 0);
}

test "elementwise ptx embedded" {
    const ptx = @embedFile("elementwise_ptx");
    try std.testing.expect(ptx.len > 0);
}

test "matmul ptx embedded" {
    const ptx = @embedFile("matmul_ptx");
    try std.testing.expect(ptx.len > 0);
}

test "cpu elementwise parity (add)" {
    const cpu = @import("../cpu/cpu.zig");
    var a = [_]f32{ 1, 2, 3 };
    var b = [_]f32{ 4, 5, 6 };
    var dst = [_]f32{ 0, 0, 0 };
    cpu.addF32(&dst, &a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 5), dst[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7), dst[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9), dst[2], 1e-6);
}

test "cpu matmul parity (naive vs blocked)" {
    const cpu = @import("../cpu/cpu.zig");
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 5, 6, 7, 8 };
    var c1 = [_]f32{ 0, 0, 0, 0 };
    var c2 = [_]f32{ 0, 0, 0, 0 };
    cpu.matmulNaifF32(&c1, &a, &b, 2, 2, 2);
    cpu.matmulBlockedF32(&c2, &a, &b, 2, 2, 2, 2);
    for (c1, c2) |x, y| try std.testing.expectApproxEqAbs(x, y, 1e-5);
}
