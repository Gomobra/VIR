// Kernel CPU: elementwise ops, matmul, softmax, rmsnorm, embedding (PRD 3.1).

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

pub fn softmaxRowF32(out: []f32, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(out.len == x.len and x.len == rows * cols);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const base = r * cols;
        var mx: f32 = x[base];
        var j: usize = 1;
        while (j < cols) : (j += 1) if (x[base + j] > mx) {
            mx = x[base + j];
        };
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

pub fn rmsnormF32(out: []f32, x: []const f32, eps: f32) void {
    std.debug.assert(out.len == x.len);
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const rms = @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    const inv = 1.0 / rms;
    for (out, x) |*o, v| o.* = v * inv;
}

pub fn embeddingGatherF32(out: []f32, table: []const f32, ids: []const u32, vocab: usize, dim: usize) void {
    std.debug.assert(out.len == ids.len * dim);
    std.debug.assert(table.len == vocab * dim);
    for (ids, 0..) |id, i| {
        std.debug.assert(id < vocab);
        const src = table[id * dim .. (id + 1) * dim];
        const dst = out[i * dim .. (i + 1) * dim];
        @memcpy(dst, src);
    }
}

// --- tests ---

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

test "matmulNaifF32" {
    var A = [_]f32{ 1, 2, 3, 4 };
    var B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 0, 0, 0, 0 };
    matmulNaifF32(&C, &A, &B, 2, 2, 2);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 19, 22, 43, 50 }, &C);
}

test "softmaxRowF32" {
    var x = [_]f32{ 1, 2, 3, 4 };
    var o = [_]f32{ 0, 0, 0, 0 };
    softmaxRowF32(&o, &x, 1, 4);
    var s: f32 = 0;
    for (o) |v| s += v;
    try std.testing.expect(@abs(s - 1.0) < 1e-6);
    try std.testing.expect(o[0] < o[1] and o[1] < o[2] and o[2] < o[3]);
}

test "rmsnormF32" {
    var x = [_]f32{ 1, 2, 3, 4 };
    var o = [_]f32{ 0, 0, 0, 0 };
    rmsnormF32(&o, &x, 1e-6);
    var ss: f32 = 0;
    for (o) |v| ss += v * v;
    const rms = @sqrt(ss / 4.0);
    try std.testing.expect(@abs(rms - 1.0) < 1e-5);
}

test "embeddingGatherF32" {
    var table = [_]f32{ 1, 2, 3, 4 };
    var ids = [_]u32{ 1, 0 };
    var out = [_]f32{ 0, 0, 0, 0 };
    embeddingGatherF32(&out, &table, &ids, 2, 2);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3, 4, 1, 2 }, &out);
}

// Blocked matmul: C[m,n] += A[m,k] * B[k,n], block size bs.
pub fn matmulBlockedF32(C: []f32, A: []const f32, B: []const f32, m: usize, n: usize, k: usize, bs: usize) void {
    std.debug.assert(C.len == m * n);
    std.debug.assert(A.len == m * k);
    std.debug.assert(B.len == k * n);
    // zero C first
    for (C) |*c| c.* = 0;
    var ii0: usize = 0;
    while (ii0 < m) : (ii0 += bs) {
        var jj0: usize = 0;
        while (jj0 < n) : (jj0 += bs) {
            var tt0: usize = 0;
            while (tt0 < k) : (tt0 += bs) {
                const iend = @min(ii0 + bs, m);
                const jend = @min(jj0 + bs, n);
                const tend = @min(tt0 + bs, k);
                var i = ii0;
                while (i < iend) : (i += 1) {
                    var t = tt0;
                    while (t < tend) : (t += 1) {
                        const aik = A[i * k + t];
                        var j = jj0;
                        while (j < jend) : (j += 1) {
                            C[i * n + j] += aik * B[t * n + j];
                        }
                    }
                }
            }
        }
    }
}

test "matmulBlockedF32 matches naif" {
    var A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    // A: 2x3
    var B = [_]f32{ 7, 8, 9, 10, 11, 12 };
    // B: 3x2
    var C1 = [_]f32{ 0, 0, 0, 0 };
    var C2 = [_]f32{ 0, 0, 0, 0 };
    matmulNaifF32(&C1, &A, &B, 2, 2, 3);
    matmulBlockedF32(&C2, &A, &B, 2, 2, 3, 2);
    for (C1, C2) |a, b| try std.testing.expect(@abs(a - b) < 1e-5);
}

// SIMD matmul (f32), vector width 4, naive with vectorized inner loop.
pub fn matmulSIMDF32(C: []f32, A: []const f32, B: []const f32, m: usize, n: usize, k: usize) void {
    std.debug.assert(C.len == m * n);
    std.debug.assert(A.len == m * k);
    std.debug.assert(B.len == k * n);
    for (C) |*c| c.* = 0;
    const Vec = @Vector(4, f32);
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j + 4 <= n) : (j += 4) {
            var acc: Vec = @splat(0);
            var t: usize = 0;
            while (t < k) : (t += 1) {
                const aik: f32 = A[i * k + t];
                const va: Vec = @splat(aik);
                const vb: Vec = B[t * n + j ..][0..4].*;
                acc += va * vb;
            }
            const res = acc;
            C[i * n + j + 0] = res[0];
            C[i * n + j + 1] = res[1];
            C[i * n + j + 2] = res[2];
            C[i * n + j + 3] = res[3];
        }
        // tail
        while (j < n) : (j += 1) {
            var acc: f32 = 0;
            var t: usize = 0;
            while (t < k) : (t += 1) acc += A[i * k + t] * B[t * n + j];
            C[i * n + j] = acc;
        }
    }
}

test "matmulSIMDF32 matches naif" {
    var A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var B = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var C1 = [_]f32{ 0, 0, 0, 0 };
    var C2 = [_]f32{ 0, 0, 0, 0 };
    matmulNaifF32(&C1, &A, &B, 2, 2, 3);
    matmulSIMDF32(&C2, &A, &B, 2, 2, 3);
    for (C1, C2) |a, b| try std.testing.expect(@abs(a - b) < 1e-5);
}
