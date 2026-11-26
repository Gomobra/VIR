// Kernel CPU: elementwise ops (PRD 3.1). Implementasi bertahap.

const std = @import("std");

pub fn addF32(dst: []f32, a: []const f32, b: []const f32) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*d, x, y| d.* = x + y;
}

pub fn mulF32(dst: []f32, a: []const f32, b: []const f32) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*d, x, y| d.* = x * y;
}

pub fn scaleF32(dst: []f32, a: []const f32, s: f32) void {
    std.debug.assert(dst.len == a.len);
    for (dst, a) |*d, x| d.* = x * s;
}

pub fn reluF32(dst: []f32, a: []const f32) void {
    std.debug.assert(dst.len == a.len);
    for (dst, a) |*d, x| d.* = if (x > 0) x else 0;
}

pub fn relu2F32(dst: []f32, a: []const f32) void {
    std.debug.assert(dst.len == a.len);
    for (dst, a) |*d, x| {
        const r = if (x > 0) x else 0;
        d.* = r * r;
    }
}

test "addF32" {
    var a = [_]f32{ 1, 2, 3 };
    var b = [_]f32{ 4, 5, 6 };
    var d = [_]f32{ 0, 0, 0 };
    addF32(&d, &a, &b);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 5, 7, 9 }, &d);
}

test "relu2F32" {
    var a = [_]f32{ -1, 2, 3 };
    var d = [_]f32{ 0, 0, 0 };
    relu2F32(&d, &a);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 0, 4, 9 }, &d);
}

// Matmul naif: C[m,n] = A[m,k] * B[k,n], row-major.
pub fn matmulNaifF32(C: []f32, A: []const f32, B: []const f32, m: usize, n: usize, k: usize) void {
    std.debug.assert(C.len == m * n);
    std.debug.assert(A.len == m * k);
    std.debug.assert(B.len == k * n);
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var acc: f32 = 0;
            var t: usize = 0;
            while (t < k) : (t += 1) {
                acc += A[i * k + t] * B[t * n + j];
            }
            C[i * n + j] = acc;
        }
    }
}

test "matmulNaifF32" {
    var A = [_]f32{ 1, 2, 3, 4 };
    var B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 0, 0, 0, 0 };
    matmulNaifF32(&C, &A, &B, 2, 2, 2);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 19, 22, 43, 50 }, &C);
}

// Softmax stabil per baris: in shape [rows, cols], out[i] = exp(x_i - max) / sum.
pub fn softmaxRowF32(out: []f32, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(out.len == x.len and x.len == rows * cols);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const base = r * cols;
        var mx: f32 = x[base];
        var j: usize = 1;
        while (j < cols) : (j += 1) if (x[base + j] > mx) { mx = x[base + j]; };
        var sum: f32 = 0;
        j = 0;
        while (j < cols) : (j += 1) {
            out[base + j] = @exp(x[base + j] - mx);
            sum += out[base + j];
        }
        j = 0;
        while (j < cols) : (j += 1) out[base + j] /= sum;
    }
}

test "softmaxRowF32" {
    var x = [_]f32{ 1, 2, 3, 4 };
    var o = [_]f32{ 0, 0, 0, 0 };
    softmaxRowF32(&o, &x, 1, 4);
    // sum should be ~1
    var s: f32 = 0;
    for (o) |v| s += v;
    try std.testing.expect(@abs(s - 1.0) < 1e-6);
    // monotonic
    try std.testing.expect(o[0] < o[1] and o[1] < o[2] and o[2] < o[3]);
}

// RMSNorm: out = x / sqrt(mean(x^2)+eps) * (no gain here, gain applied separately).
pub fn rmsnormF32(out: []f32, x: []const f32, eps: f32) void {
    std.debug.assert(out.len == x.len);
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const rms = @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    const inv = 1.0 / rms;
    for (out, x) |*o, v| o.* = v * inv;
}

test "rmsnormF32" {
    var x = [_]f32{ 1, 2, 3, 4 };
    var o = [_]f32{ 0, 0, 0, 0 };
    rmsnormF32(&o, &x, 1e-6);
    // RMS of output should be ~1
    var ss: f32 = 0;
    for (o) |v| ss += v * v;
    const rms = @sqrt(ss / 4.0);
    try std.testing.expect(@abs(rms - 1.0) < 1e-5);
}
