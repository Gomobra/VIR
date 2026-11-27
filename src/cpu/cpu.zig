// Kernel CPU: elementwise ops, matmul, softmax, rmsnorm, embedding (PRD 3.1).
pub const rng = @import("rng.zig");

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

pub fn sigmoidF32(dst: []f32, a: []const f32) void {
    std.debug.assert(dst.len == a.len);
    for (dst, a) |*d, x| d.* = 1.0 / (1.0 + @exp(-x));
}

pub fn siluF32(dst: []f32, a: []const f32) void {
    std.debug.assert(dst.len == a.len);
    for (dst, a) |*d, x| d.* = x / (1.0 + @exp(-x));
}

pub fn softplusF32(dst: []f32, a: []const f32) void {
    std.debug.assert(dst.len == a.len);
    for (dst, a) |*d, x| d.* = @log(1.0 + @exp(x));
}

pub fn expF32(dst: []f32, a: []const f32) void {
    std.debug.assert(dst.len == a.len);
    for (dst, a) |*d, x| d.* = @exp(x);
}

pub fn logF32(dst: []f32, a: []const f32) void {
    std.debug.assert(dst.len == a.len);
    for (dst, a) |*d, x| d.* = @log(x);
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

// Block-diagonal attention mask for packed sequences (PRD 3.8, 3.6).
// mask[i*seq + j] = 1 if doc_ids[i]==doc_ids[j] and j<=i (causal within doc), else 0.
pub fn blockDiagonalMask(mask: []f32, doc_ids: []const u32, seq: usize) void {
    std.debug.assert(mask.len == seq * seq);
    std.debug.assert(doc_ids.len == seq);
    var i: usize = 0;
    while (i < seq) : (i += 1) {
        var j: usize = 0;
        while (j < seq) : (j += 1) {
            if (j <= i and doc_ids[i] == doc_ids[j]) {
                mask[i * seq + j] = 1.0;
            } else {
                mask[i * seq + j] = 0.0;
            }
        }
    }
}

test "blockDiagonalMask" {
    // two docs: [0,0,1,1], causal within doc
    const doc_ids = [_]u32{ 0, 0, 1, 1 };
    var mask = [_]f32{0} * *16;
    blockDiagonalMask(&mask, &doc_ids, 4);
    // position 2 (doc1) cannot attend to position 0 (doc0)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mask[2 * 4 + 0], 1e-6);
    // position 2 can attend to position 2
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mask[2 * 4 + 2], 1e-6);
    // position 1 can attend to position 0 (same doc)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mask[1 * 4 + 0], 1e-6);
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

// Causal scaled dot-product attention (single head): out = softmax(Q K^T / sqrt(d)) V, kausal.
pub fn attentionNaiveF32(out: []f32, q: []const f32, k: []const f32, v: []const f32, seq: usize, d: usize) void {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
    var i: usize = 0;
    while (i < seq) : (i += 1) {
        // scores[i, j] for j<=i
        var maxs: f32 = -1e9;
        var j: usize = 0;
        // compute max
        while (j <= i) : (j += 1) {
            var acc: f32 = 0;
            var t: usize = 0;
            while (t < d) : (t += 1) acc += q[i * d + t] * k[j * d + t];
            acc *= scale;
            if (acc > maxs) maxs = acc;
        }
        // softmax + weighted sum
        var sum: f32 = 0;
        // store scores in temp? reuse out row
        var jj: usize = 0;
        while (jj <= i) : (jj += 1) {
            var acc: f32 = 0;
            var t: usize = 0;
            while (t < d) : (t += 1) acc += q[i * d + t] * k[jj * d + t];
            acc *= scale;
            const e = @exp(acc - maxs);
            out[i * d + jj] = e; // store unnormalized weight temporarily in out? no, out is d-dim
            sum += e;
        }
        // compute output
        var t: usize = 0;
        while (t < d) : (t += 1) {
            var acc: f32 = 0;
            var jj2: usize = 0;
            while (jj2 <= i) : (jj2 += 1) {
                var sc: f32 = 0;
                var t2: usize = 0;
                while (t2 < d) : (t2 += 1) sc += q[i * d + t2] * k[jj2 * d + t2];
                sc *= scale;
                const w = @exp(sc - maxs) / sum;
                acc += w * v[jj2 * d + t];
            }
            out[i * d + t] = acc;
        }
    }
}

test "attentionNaiveF32 finite" {
    const seq = 2;
    const d = 2;
    var q = [_]f32{ 1, 0, 0, 1 };
    var k = [_]f32{ 1, 0, 0, 1 };
    var v = [_]f32{ 1, 2, 3, 4 };
    var out = [_]f32{ 0, 0, 0, 0 };
    attentionNaiveF32(&out, &q, &k, &v, seq, d);
    for (out) |x| try std.testing.expect(std.math.isFinite(x));
}

// Fused cross-entropy (per row): given logits [rows, vocab] and targets [rows],
// returns sum of -log softmax(target). Computes in f32.
pub fn crossEntropyF32(logits: []const f32, targets: []const u32, rows: usize, vocab: usize) f64 {
    std.debug.assert(logits.len == rows * vocab);
    std.debug.assert(targets.len == rows);
    var total: f64 = 0;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const base = r * vocab;
        // max
        var mx: f32 = logits[base];
        var j: usize = 1;
        while (j < vocab) : (j += 1) if (logits[base + j] > mx) {
            mx = logits[base + j];
        };
        // logsumexp
        var sum: f32 = 0;
        j = 0;
        while (j < vocab) : (j += 1) sum += @exp(logits[base + j] - mx);
        const lse = mx + @log(sum);
        const tgt = targets[r];
        total += @as(f64, lse - logits[base + tgt]);
    }
    return total;
}

test "crossEntropyF32" {
    // logits: row0 = [0,0], target 0 -> -log(0.5)=0.693
    var logits = [_]f32{ 0, 0 };
    var targets = [_]u32{0};
    const ce = crossEntropyF32(&logits, &targets, 1, 2);
    try std.testing.expect(@abs(ce - 0.6931) < 1e-3);
}

// Causal 1D convolution: x [Cin, L], w [Cout, Cin, K], out [Cout, L].
// Padding: K-1 zeros on the left (causal).
pub fn conv1dF32(out: []f32, x: []const f32, w: []const f32, cin: usize, cout: usize, kernel: usize, len: usize) void {
    std.debug.assert(out.len == cout * len);
    std.debug.assert(x.len == cin * len);
    std.debug.assert(w.len == cout * cin * kernel);
    var co: usize = 0;
    while (co < cout) : (co += 1) {
        var t: usize = 0;
        while (t < len) : (t += 1) {
            var acc: f32 = 0;
            var ci: usize = 0;
            while (ci < cin) : (ci += 1) {
                var k: usize = 0;
                while (k < kernel) : (k += 1) {
                    const ti = @as(isize, @intCast(t)) - @as(isize, @intCast(kernel - 1)) + @as(isize, @intCast(k));
                    if (ti >= 0) {
                        const xidx = ci * len + @as(usize, @intCast(ti));
                        acc += x[xidx] * w[(co * cin + ci) * kernel + k];
                    }
                }
            }
            out[co * len + t] = acc;
        }
    }
}

// Backward for causal conv1d. Accumulates dX and dW.
pub fn conv1dBackwardF32(dX: []f32, dW: []f32, x: []const f32, w: []const f32, dOut: []const f32, cin: usize, cout: usize, kernel: usize, len: usize) void {
    std.debug.assert(dX.len == cin * len);
    std.debug.assert(dW.len == cout * cin * kernel);
    std.debug.assert(dOut.len == cout * len);
    // dX
    var ci: usize = 0;
    while (ci < cin) : (ci += 1) {
        var t: usize = 0;
        while (t < len) : (t += 1) {
            var acc: f32 = 0;
            var co: usize = 0;
            while (co < cout) : (co += 1) {
                var k: usize = 0;
                while (k < kernel) : (k += 1) {
                    const to = @as(isize, @intCast(t)) + @as(isize, @intCast(kernel - 1)) - @as(isize, @intCast(k));
                    if (to >= 0 and to < @as(isize, @intCast(len))) {
                        acc += dOut[co * len + @as(usize, @intCast(to))] * w[(co * cin + ci) * kernel + k];
                    }
                }
            }
            dX[ci * len + t] += acc;
        }
    }
    // dW
    var co: usize = 0;
    while (co < cout) : (co += 1) {
        var ci2: usize = 0;
        while (ci2 < cin) : (ci2 += 1) {
            var k: usize = 0;
            while (k < kernel) : (k += 1) {
                var acc: f32 = 0;
                var t: usize = 0;
                while (t < len) : (t += 1) {
                    const ti = @as(isize, @intCast(t)) - @as(isize, @intCast(kernel - 1)) + @as(isize, @intCast(k));
                    if (ti >= 0) {
                        acc += x[ci2 * len + @as(usize, @intCast(ti))] * dOut[co * len + t];
                    }
                }
                dW[(co * cin + ci2) * kernel + k] += acc;
            }
        }
    }
}

test "conv1d causal finite" {
    const cin = 1;
    const cout = 1;
    const kernel = 2;
    const len = 4;
    var x = [_]f32{ 1, 2, 3, 4 };
    var w = [_]f32{ 1, -1 };
    var out = [_]f32{ 0, 0, 0, 0 };
    conv1dF32(&out, &x, &w, cin, cout, kernel, len);
    // y[t] = x[t] - x[t-1] (causal), y[0]=x[0]
    try std.testing.expectApproxEqAbs(@as(f32, 1), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), out[3], 1e-6);
}

test "conv1d backward finite" {
    const cin = 1;
    const cout = 1;
    const kernel = 2;
    const len = 4;
    var x = [_]f32{ 1, 2, 3, 4 };
    var w = [_]f32{ 1, -1 };
    var dOut = [_]f32{ 1, 1, 1, 1 };
    var dX = [_]f32{ 0, 0, 0, 0 };
    var dW = [_]f32{ 0, 0 };
    conv1dBackwardF32(&dX, &dW, &x, &w, &dOut, cin, cout, kernel, len);
    for (dX) |v| try std.testing.expect(std.math.isFinite(v));
    for (dW) |v| try std.testing.expect(std.math.isFinite(v));
}

// 2D convolution (stride 1, 'same' padding for odd kernels): x [Cin,H,W], w [Cout,Cin,Kh,Kw], out [Cout,H,W].
pub fn conv2dF32(out: []f32, x: []const f32, w: []const f32, cin: usize, cout: usize, kh: usize, kw: usize, h: usize, wd: usize) void {
    std.debug.assert(out.len == cout * h * wd);
    std.debug.assert(x.len == cin * h * wd);
    std.debug.assert(w.len == cout * cin * kh * kw);
    const pad_h = (kh - 1) / 2;
    const pad_w = (kw - 1) / 2;
    var co: usize = 0;
    while (co < cout) : (co += 1) {
        var y: usize = 0;
        while (y < h) : (y += 1) {
            var xidx: usize = 0;
            while (xidx < wd) : (xidx += 1) {
                var acc: f32 = 0;
                var ci: usize = 0;
                while (ci < cin) : (ci += 1) {
                    var ky: usize = 0;
                    while (ky < kh) : (ky += 1) {
                        var kx: usize = 0;
                        while (kx < kw) : (kx += 1) {
                            const iy = @as(isize, @intCast(y)) + @as(isize, @intCast(ky)) - @as(isize, @intCast(pad_h));
                            const ix = @as(isize, @intCast(xidx)) + @as(isize, @intCast(kx)) - @as(isize, @intCast(pad_w));
                            if (iy >= 0 and iy < @as(isize, @intCast(h)) and ix >= 0 and ix < @as(isize, @intCast(wd))) {
                                const xi = ci * h * wd + @as(usize, @intCast(iy)) * wd + @as(usize, @intCast(ix));
                                const wi = (co * cin + ci) * kh * kw + ky * kw + kx;
                                acc += x[xi] * w[wi];
                            }
                        }
                    }
                }
                out[co * h * wd + y * wd + xidx] = acc;
            }
        }
    }
}

pub fn conv2dBackwardF32(dX: []f32, dW: []f32, x: []const f32, w: []const f32, dOut: []const f32, cin: usize, cout: usize, kh: usize, kw: usize, h: usize, wd: usize) void {
    std.debug.assert(dX.len == cin * h * wd);
    std.debug.assert(dW.len == cout * cin * kh * kw);
    std.debug.assert(dOut.len == cout * h * wd);
    const pad_h = (kh - 1) / 2;
    const pad_w = (kw - 1) / 2;
    // dX
    var ci: usize = 0;
    while (ci < cin) : (ci += 1) {
        var y: usize = 0;
        while (y < h) : (y += 1) {
            var xidx: usize = 0;
            while (xidx < wd) : (xidx += 1) {
                var acc: f32 = 0;
                var co: usize = 0;
                while (co < cout) : (co += 1) {
                    var ky: usize = 0;
                    while (ky < kh) : (ky += 1) {
                        var kx: usize = 0;
                        while (kx < kw) : (kx += 1) {
                            const oy = @as(isize, @intCast(y)) - @as(isize, @intCast(ky)) + @as(isize, @intCast(pad_h));
                            const ox = @as(isize, @intCast(xidx)) - @as(isize, @intCast(kx)) + @as(isize, @intCast(pad_w));
                            if (oy >= 0 and oy < @as(isize, @intCast(h)) and ox >= 0 and ox < @as(isize, @intCast(wd))) {
                                const wi = (co * cin + ci) * kh * kw + ky * kw + kx;
                                acc += dOut[co * h * wd + @as(usize, @intCast(oy)) * wd + @as(usize, @intCast(ox))] * w[wi];
                            }
                        }
                    }
                }
                dX[ci * h * wd + y * wd + xidx] += acc;
            }
        }
    }
    // dW
    var co: usize = 0;
    while (co < cout) : (co += 1) {
        var ci2: usize = 0;
        while (ci2 < cin) : (ci2 += 1) {
            var ky: usize = 0;
            while (ky < kh) : (ky += 1) {
                var kx: usize = 0;
                while (kx < kw) : (kx += 1) {
                    var acc: f32 = 0;
                    var y: usize = 0;
                    while (y < h) : (y += 1) {
                        var xidx: usize = 0;
                        while (xidx < wd) : (xidx += 1) {
                            const iy = @as(isize, @intCast(y)) + @as(isize, @intCast(ky)) - @as(isize, @intCast(pad_h));
                            const ix = @as(isize, @intCast(xidx)) + @as(isize, @intCast(kx)) - @as(isize, @intCast(pad_w));
                            if (iy >= 0 and iy < @as(isize, @intCast(h)) and ix >= 0 and ix < @as(isize, @intCast(wd))) {
                                const xi = ci2 * h * wd + @as(usize, @intCast(iy)) * wd + @as(usize, @intCast(ix));
                                acc += x[xi] * dOut[co * h * wd + y * wd + xidx];
                            }
                        }
                    }
                    dW[(co * cin + ci2) * kh * kw + ky * kw + kx] += acc;
                }
            }
        }
    }
}

test "conv2d finite" {
    const cin = 1;
    const cout = 1;
    const kh = 3;
    const kw = 3;
    const h = 4;
    const wdim = 4;
    var x = [_]f32{0} * *(cin * h * wdim);
    for (&x) |*v| v.* = 1;
    var w = [_]f32{0} * *(cout * cin * kh * kw);
    for (&w) |*v| v.* = 0.1;
    var out = [_]f32{0} * *(cout * h * wdim);
    conv2dF32(&out, &x, &w, cin, cout, kh, kw, h, wdim);
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}

test "conv2d backward finite" {
    const cin = 1;
    const cout = 1;
    const kh = 3;
    const kw = 3;
    const h = 4;
    const wdim = 4;
    var x = [_]f32{0} * *(cin * h * wdim);
    for (&x) |*v| v.* = 1;
    var w = [_]f32{0} * *(cout * cin * kh * kw);
    for (&w) |*v| v.* = 0.1;
    var dOut = [_]f32{0} * *(cout * h * wdim);
    for (&dOut) |*v| v.* = 1;
    var dX = [_]f32{0} * *(cin * h * wdim);
    var dW = [_]f32{0} * *(cout * cin * kh * kw);
    conv2dBackwardF32(&dX, &dW, &x, &w, &dOut, cin, cout, kh, kw, h, wdim);
    for (dX) |v| try std.testing.expect(std.math.isFinite(v));
    for (dW) |v| try std.testing.expect(std.math.isFinite(v));
}

test "elementwise new ops finite" {
    var a = [_]f32{ -1, 0, 1, 2 };
    var d = [_]f32{ 0, 0, 0, 0 };
    sigmoidF32(&d, &a);
    for (d) |v| try std.testing.expect(v >= 0 and v <= 1);
    siluF32(&d, &a);
    for (d) |v| try std.testing.expect(std.math.isFinite(v));
    softplusF32(&d, &a);
    for (d) |v| try std.testing.expect(std.math.isFinite(v));
    var p = [_]f32{ 1, 2, 3, 4 };
    expF32(&d, &p);
    logF32(&d, &d);
    for (d, p) |v, x| try std.testing.expectApproxEqAbs(x, v, 1e-5);
}

// Transposed causal 1D convolution (upsampling): x [Cin, Lin], w [Cout, Cin, K], out [Cout, (Lin-1)*stride + K].
pub fn convTranspose1dF32(out: []f32, x: []const f32, w: []const f32, cin: usize, cout: usize, kernel: usize, stride: usize, in_len: usize) void {
    const out_len = (in_len - 1) * stride + kernel;
    std.debug.assert(out.len == cout * out_len);
    std.debug.assert(x.len == cin * in_len);
    std.debug.assert(w.len == cout * cin * kernel);
    @memset(out, 0);
    var co: usize = 0;
    while (co < cout) : (co += 1) {
        var ci: usize = 0;
        while (ci < cin) : (ci += 1) {
            var i: usize = 0;
            while (i < in_len) : (i += 1) {
                var k: usize = 0;
                while (k < kernel) : (k += 1) {
                    const t = i * stride + k;
                    out[co * out_len + t] += x[ci * in_len + i] * w[(co * cin + ci) * kernel + k];
                }
            }
        }
    }
}

pub fn convTranspose1dBackwardF32(dX: []f32, dW: []f32, x: []const f32, w: []const f32, dOut: []const f32, cin: usize, cout: usize, kernel: usize, stride: usize, in_len: usize) void {
    const out_len = (in_len - 1) * stride + kernel;
    std.debug.assert(dX.len == cin * in_len);
    std.debug.assert(dW.len == cout * cin * kernel);
    std.debug.assert(dOut.len == cout * out_len);
    var co: usize = 0;
    while (co < cout) : (co += 1) {
        var ci: usize = 0;
        while (ci < cin) : (ci += 1) {
            var i: usize = 0;
            while (i < in_len) : (i += 1) {
                var acc: f32 = 0;
                var k: usize = 0;
                while (k < kernel) : (k += 1) {
                    const t = i * stride + k;
                    acc += dOut[co * out_len + t] * w[(co * cin + ci) * kernel + k];
                }
                dX[ci * in_len + i] += acc;
            }
            var k: usize = 0;
            while (k < kernel) : (k += 1) {
                var acc: f32 = 0;
                var ii: usize = 0;
                while (ii < in_len) : (ii += 1) {
                    const t = ii * stride + k;
                    acc += x[ci * in_len + ii] * dOut[co * out_len + t];
                }
                dW[(co * cin + ci) * kernel + k] += acc;
            }
        }
    }
}

test "convTranspose1d shape" {
    const cin = 1;
    const cout = 1;
    const kernel = 4;
    const stride = 2;
    const in_len = 3;
    const out_len = (in_len - 1) * stride + kernel;
    var x = [_]f32{ 1, 2, 3 };
    var w = [_]f32{ 1, 0, 0, 0 };
    var out = [_]f32{0} * *out_len;
    convTranspose1dF32(&out, &x, &w, cin, cout, kernel, stride, in_len);
    try std.testing.expectApproxEqAbs(@as(f32, 1), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), out[4], 1e-6);
}

test "convTranspose1d backward finite" {
    const cin = 1;
    const cout = 1;
    const kernel = 3;
    const stride = 2;
    const in_len = 2;
    const out_len = (in_len - 1) * stride + kernel;
    var x = [_]f32{ 1, 2 };
    var w = [_]f32{ 0.5, -0.25, 0.1 };
    var dOut = [_]f32{0} * *out_len;
    for (&dOut) |*v| v.* = 1;
    var dX = [_]f32{ 0, 0 };
    var dW = [_]f32{ 0, 0, 0 };
    convTranspose1dBackwardF32(&dX, &dW, &x, &w, &dOut, cin, cout, kernel, stride, in_len);
    for (dX) |v| try std.testing.expect(std.math.isFinite(v));
    for (dW) |v| try std.testing.expect(std.math.isFinite(v));
}

// 3D convolution (stride 1, 'same' padding): x [Cin,T,H,W], w [Cout,Cin,Kt,Kh,Kw], out [Cout,T,H,W].
pub fn conv3dF32(out: []f32, x: []const f32, w: []const f32, cin: usize, cout: usize, kt: usize, kh: usize, kw: usize, t: usize, h: usize, wd: usize) void {
    std.debug.assert(out.len == cout * t * h * wd);
    std.debug.assert(x.len == cin * t * h * wd);
    std.debug.assert(w.len == cout * cin * kt * kh * kw);
    const pad_t = (kt - 1) / 2;
    const pad_h = (kh - 1) / 2;
    const pad_w = (kw - 1) / 2;
    var co: usize = 0;
    while (co < cout) : (co += 1) {
        var tt: usize = 0;
        while (tt < t) : (tt += 1) {
            var y: usize = 0;
            while (y < h) : (y += 1) {
                var xidx: usize = 0;
                while (xidx < wd) : (xidx += 1) {
                    var acc: f32 = 0;
                    var ci: usize = 0;
                    while (ci < cin) : (ci += 1) {
                        var kti: usize = 0;
                        while (kti < kt) : (kti += 1) {
                            var khi: usize = 0;
                            while (khi < kh) : (khi += 1) {
                                var kwi: usize = 0;
                                while (kwi < kw) : (kwi += 1) {
                                    const it = @as(isize, @intCast(tt)) + @as(isize, @intCast(kti)) - @as(isize, @intCast(pad_t));
                                    const iy = @as(isize, @intCast(y)) + @as(isize, @intCast(khi)) - @as(isize, @intCast(pad_h));
                                    const ix = @as(isize, @intCast(xidx)) + @as(isize, @intCast(kwi)) - @as(isize, @intCast(pad_w));
                                    if (it >= 0 and it < @as(isize, @intCast(t)) and iy >= 0 and iy < @as(isize, @intCast(h)) and ix >= 0 and ix < @as(isize, @intCast(wd))) {
                                        const xi = ci * t * h * wd + @as(usize, @intCast(it)) * h * wd + @as(usize, @intCast(iy)) * wd + @as(usize, @intCast(ix));
                                        const wi = (co * cin + ci) * kt * kh * kw + kti * kh * kw + khi * kw + kwi;
                                        acc += x[xi] * w[wi];
                                    }
                                }
                            }
                        }
                    }
                    out[co * t * h * wd + tt * h * wd + y * wd + xidx] = acc;
                }
            }
        }
    }
}

pub fn conv3dBackwardF32(dX: []f32, dW: []f32, x: []const f32, w: []const f32, dOut: []const f32, cin: usize, cout: usize, kt: usize, kh: usize, kw: usize, t: usize, h: usize, wd: usize) void {
    std.debug.assert(dX.len == cin * t * h * wd);
    std.debug.assert(dW.len == cout * cin * kt * kh * kw);
    std.debug.assert(dOut.len == cout * t * h * wd);
    const pad_t = (kt - 1) / 2;
    const pad_h = (kh - 1) / 2;
    const pad_w = (kw - 1) / 2;
    // dX
    var ci: usize = 0;
    while (ci < cin) : (ci += 1) {
        var tt: usize = 0;
        while (tt < t) : (tt += 1) {
            var y: usize = 0;
            while (y < h) : (y += 1) {
                var xidx: usize = 0;
                while (xidx < wd) : (xidx += 1) {
                    var acc: f32 = 0;
                    var co: usize = 0;
                    while (co < cout) : (co += 1) {
                        var kti: usize = 0;
                        while (kti < kt) : (kti += 1) {
                            var khi: usize = 0;
                            while (khi < kh) : (khi += 1) {
                                var kwi: usize = 0;
                                while (kwi < kw) : (kwi += 1) {
                                    const ot = @as(isize, @intCast(tt)) - @as(isize, @intCast(kti)) + @as(isize, @intCast(pad_t));
                                    const oy = @as(isize, @intCast(y)) - @as(isize, @intCast(khi)) + @as(isize, @intCast(pad_h));
                                    const ox = @as(isize, @intCast(xidx)) - @as(isize, @intCast(kwi)) + @as(isize, @intCast(pad_w));
                                    if (ot >= 0 and ot < @as(isize, @intCast(t)) and oy >= 0 and oy < @as(isize, @intCast(h)) and ox >= 0 and ox < @as(isize, @intCast(wd))) {
                                        const wi = (co * cin + ci) * kt * kh * kw + kti * kh * kw + khi * kw + kwi;
                                        acc += dOut[co * t * h * wd + @as(usize, @intCast(ot)) * h * wd + @as(usize, @intCast(oy)) * wd + @as(usize, @intCast(ox))] * w[wi];
                                    }
                                }
                            }
                        }
                    }
                    dX[ci * t * h * wd + tt * h * wd + y * wd + xidx] += acc;
                }
            }
        }
    }
    // dW
    var co2: usize = 0;
    while (co2 < cout) : (co2 += 1) {
        var ci2: usize = 0;
        while (ci2 < cin) : (ci2 += 1) {
            var kti: usize = 0;
            while (kti < kt) : (kti += 1) {
                var khi: usize = 0;
                while (khi < kh) : (khi += 1) {
                    var kwi: usize = 0;
                    while (kwi < kw) : (kwi += 1) {
                        var acc: f32 = 0;
                        var tt: usize = 0;
                        while (tt < t) : (tt += 1) {
                            var y: usize = 0;
                            while (y < h) : (y += 1) {
                                var xidx: usize = 0;
                                while (xidx < wd) : (xidx += 1) {
                                    const it = @as(isize, @intCast(tt)) + @as(isize, @intCast(kti)) - @as(isize, @intCast(pad_t));
                                    const iy = @as(isize, @intCast(y)) + @as(isize, @intCast(khi)) - @as(isize, @intCast(pad_h));
                                    const ix = @as(isize, @intCast(xidx)) + @as(isize, @intCast(kwi)) - @as(isize, @intCast(pad_w));
                                    if (it >= 0 and it < @as(isize, @intCast(t)) and iy >= 0 and iy < @as(isize, @intCast(h)) and ix >= 0 and ix < @as(isize, @intCast(wd))) {
                                        const xi = ci2 * t * h * wd + @as(usize, @intCast(it)) * h * wd + @as(usize, @intCast(iy)) * wd + @as(usize, @intCast(ix));
                                        acc += x[xi] * dOut[co2 * t * h * wd + tt * h * wd + y * wd + xidx];
                                    }
                                }
                            }
                        }
                        dW[(co2 * cin + ci2) * kt * kh * kw + kti * kh * kw + khi * kw + kwi] += acc;
                    }
                }
            }
        }
    }
}

test "conv3d finite" {
    const cin = 1;
    const cout = 1;
    const kt = 3;
    const kh = 3;
    const kw = 3;
    const t = 2;
    const h = 3;
    const wd = 3;
    var x = [_]f32{0} * *(cin * t * h * wd);
    for (&x) |*v| v.* = 1;
    var w = [_]f32{0} * *(cout * cin * kt * kh * kw);
    for (&w) |*v| v.* = 0.1;
    var out = [_]f32{0} * *(cout * t * h * wd);
    conv3dF32(&out, &x, &w, cin, cout, kt, kh, kw, t, h, wd);
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}

test "conv3d backward finite" {
    const cin = 1;
    const cout = 1;
    const kt = 3;
    const kh = 3;
    const kw = 3;
    const t = 2;
    const h = 3;
    const wd = 3;
    var x = [_]f32{0} * *(cin * t * h * wd);
    for (&x) |*v| v.* = 1;
    var w = [_]f32{0} * *(cout * cin * kt * kh * kw);
    for (&w) |*v| v.* = 0.1;
    var dOut = [_]f32{0} * *(cout * t * h * wd);
    for (&dOut) |*v| v.* = 1;
    var dX = [_]f32{0} * *(cin * t * h * wd);
    var dW = [_]f32{0} * *(cout * cin * kt * kh * kw);
    conv3dBackwardF32(&dX, &dW, &x, &w, &dOut, cin, cout, kt, kh, kw, t, h, wd);
    for (dX) |v| try std.testing.expect(std.math.isFinite(v));
    for (dW) |v| try std.testing.expect(std.math.isFinite(v));
}

// GroupNorm: x/out [C, H, W], groups divides C.
pub fn groupNormF32(out: []f32, x: []const f32, groups: usize, channels: usize, h: usize, w: usize, eps: f32) void {
    std.debug.assert(out.len == channels * h * w);
    std.debug.assert(x.len == channels * h * w);
    std.debug.assert(groups > 0 and channels % groups == 0);
    const cg = channels / groups;
    const n = cg * h * w;
    var g: usize = 0;
    while (g < groups) : (g += 1) {
        var mean: f32 = 0;
        var ci: usize = g * cg;
        while (ci < (g + 1) * cg) : (ci += 1) {
            var y: usize = 0;
            while (y < h) : (y += 1) {
                var xi: usize = 0;
                while (xi < w) : (xi += 1) {
                    mean += x[ci * h * w + y * w + xi];
                }
            }
        }
        mean /= @as(f32, @floatFromInt(n));
        var varv: f32 = 0;
        ci = g * cg;
        while (ci < (g + 1) * cg) : (ci += 1) {
            var y: usize = 0;
            while (y < h) : (y += 1) {
                var xi: usize = 0;
                while (xi < w) : (xi += 1) {
                    const diff = x[ci * h * w + y * w + xi] - mean;
                    varv += diff * diff;
                }
            }
        }
        varv /= @as(f32, @floatFromInt(n));
        const inv_std = 1.0 / @sqrt(varv + eps);
        ci = g * cg;
        while (ci < (g + 1) * cg) : (ci += 1) {
            var y: usize = 0;
            while (y < h) : (y += 1) {
                var xi: usize = 0;
                while (xi < w) : (xi += 1) {
                    const idx = ci * h * w + y * w + xi;
                    out[idx] = (x[idx] - mean) * inv_std;
                }
            }
        }
    }
}

pub fn groupNormBackwardF32(dX: []f32, x: []const f32, dOut: []const f32, groups: usize, channels: usize, h: usize, w: usize, eps: f32) void {
    std.debug.assert(dX.len == channels * h * w);
    std.debug.assert(x.len == channels * h * w);
    std.debug.assert(dOut.len == channels * h * w);
    std.debug.assert(groups > 0 and channels % groups == 0);
    const cg = channels / groups;
    const n = cg * h * w;
    var g: usize = 0;
    while (g < groups) : (g += 1) {
        var mean_x: f32 = 0;
        var mean_do: f32 = 0;
        var ci: usize = g * cg;
        while (ci < (g + 1) * cg) : (ci += 1) {
            var y: usize = 0;
            while (y < h) : (y += 1) {
                var xi: usize = 0;
                while (xi < w) : (xi += 1) {
                    const idx = ci * h * w + y * w + xi;
                    mean_x += x[idx];
                    mean_do += dOut[idx];
                }
            }
        }
        mean_x /= @as(f32, @floatFromInt(n));
        mean_do /= @as(f32, @floatFromInt(n));
        var varv: f32 = 0;
        ci = g * cg;
        while (ci < (g + 1) * cg) : (ci += 1) {
            var y: usize = 0;
            while (y < h) : (y += 1) {
                var xi: usize = 0;
                while (xi < w) : (xi += 1) {
                    const idx = ci * h * w + y * w + xi;
                    const diff = x[idx] - mean_x;
                    varv += diff * diff;
                }
            }
        }
        varv /= @as(f32, @floatFromInt(n));
        const inv_std = 1.0 / @sqrt(varv + eps);
        var sum_do_x: f32 = 0;
        ci = g * cg;
        while (ci < (g + 1) * cg) : (ci += 1) {
            var y: usize = 0;
            while (y < h) : (y += 1) {
                var xi: usize = 0;
                while (xi < w) : (xi += 1) {
                    const idx = ci * h * w + y * w + xi;
                    sum_do_x += dOut[idx] * (x[idx] - mean_x);
                }
            }
        }
        const scale = sum_do_x / (varv + eps) / @as(f32, @floatFromInt(n));
        ci = g * cg;
        while (ci < (g + 1) * cg) : (ci += 1) {
            var y: usize = 0;
            while (y < h) : (y += 1) {
                var xi: usize = 0;
                while (xi < w) : (xi += 1) {
                    const idx = ci * h * w + y * w + xi;
                    dX[idx] += inv_std * (dOut[idx] - mean_do - (x[idx] - mean_x) * scale);
                }
            }
        }
    }
}

test "groupNorm finite" {
    const c = 2;
    const h = 2;
    const w = 2;
    var x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var out = [_]f32{0} * *(c * h * w);
    groupNormF32(&out, &x, 1, c, h, w, 1e-6);
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}

test "groupNorm backward finite" {
    const c = 2;
    const h = 2;
    const w = 2;
    var x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var dOut = [_]f32{ 1, 0, 0, 1, 1, 1, 0, 0 };
    var dX = [_]f32{0} * *(c * h * w);
    groupNormBackwardF32(&dX, &x, &dOut, 1, c, h, w, 1e-6);
    for (dX) |v| try std.testing.expect(std.math.isFinite(v));
}
