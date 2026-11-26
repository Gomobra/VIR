// Autograd reverse-mode (PRD 3.2). Tape + backward helpers.

const std = @import("std");

pub const OpTag = enum { add, mul, scale, relu };

pub const Node = struct {
    tag: OpTag,
    n_inputs: u8,
    n_outputs: u8,
};

pub const Tape = struct {
    nodes: std.ArrayList(Node),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Tape {
        return .{ .nodes = std.ArrayList(Node).init(allocator), .allocator = allocator };
    }

    pub fn deinit(self: *Tape) void {
        self.nodes.deinit();
    }

    pub fn record(self: *Tape, node: Node) !void {
        try self.nodes.append(node);
    }

    pub fn len(self: *const Tape) usize {
        return self.nodes.items.len;
    }

    pub fn clear(self: *Tape) void {
        self.nodes.clearRetainingCapacity();
    }
};

// Backward helpers (slice-based, contiguous).

// dA += dOut (for z = a + b), dB += dOut.
pub fn backwardAdd(dA: []f32, dB: []f32, dOut: []const f32) void {
    std.debug.assert(dA.len == dOut.len and dB.len == dOut.len);
    for (dA, dB, dOut) |*a, *b, g| {
        a.* += g;
        b.* += g;
    }
}

// For z = a * b: dA += dOut * b, dB += dOut * a.
pub fn backwardMul(dA: []f32, dB: []f32, a: []const f32, b: []const f32, dOut: []const f32) void {
    std.debug.assert(dA.len == dOut.len and dB.len == dOut.len);
    std.debug.assert(a.len == dOut.len and b.len == dOut.len);
    for (dA, dB, a, b, dOut) |*da, *db, av, bv, g| {
        da.* += g * bv;
        db.* += g * av;
    }
}

// For z = s * a: dA += dOut * s.
pub fn backwardScale(dA: []f32, dOut: []const f32, s: f32) void {
    std.debug.assert(dA.len == dOut.len);
    for (dA, dOut) |*a, g| a.* += g * s;
}

// For z = relu(a): dA += dOut if a > 0.
pub fn backwardRelu(dA: []f32, a: []const f32, dOut: []const f32) void {
    std.debug.assert(dA.len == dOut.len and a.len == dOut.len);
    for (dA, a, dOut) |*da, av, g| {
        if (av > 0) da.* += g;
    }
}

// Backward matmul: C=A@B (m x n), A (m x k), B (k x n).
// dA[m,k] += dC[m,n] @ B^T ; dB[k,n] += A^T @ dC[m,n].
pub fn backwardMatmul(dA: []f32, dB: []f32, A: []const f32, B: []const f32, dC: []const f32, m: usize, n: usize, k: usize) void {
    std.debug.assert(dA.len == m * k);
    std.debug.assert(dB.len == k * n);
    std.debug.assert(dC.len == m * n);
    // dA
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var t: usize = 0;
        while (t < k) : (t += 1) {
            var acc: f32 = 0;
            var j: usize = 0;
            while (j < n) : (j += 1) {
                acc += dC[i * n + j] * B[t * n + j];
            }
            dA[i * k + t] += acc;
        }
    }
    // dB
    var t2: usize = 0;
    while (t2 < k) : (t2 += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var acc: f32 = 0;
            var iidx: usize = 0;
            while (iidx < m) : (iidx += 1) {
                acc += A[iidx * k + t2] * dC[iidx * n + j];
            }
            dB[t2 * n + j] += acc;
        }
    }
}

// Backward softmax: given softmax output s (rows x cols) and dOut, dX = s*(dOut - sum(dOut*s)).
pub fn backwardSoftmax(dX: []f32, s: []const f32, dOut: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(dX.len == s.len and s.len == dOut.len and s.len == rows * cols);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const base = r * cols;
        // sum(dOut * s)
        var sum: f32 = 0;
        var j: usize = 0;
        while (j < cols) : (j += 1) sum += dOut[base + j] * s[base + j];
        j = 0;
        while (j < cols) : (j += 1) {
            dX[base + j] += s[base + j] * (dOut[base + j] - sum);
        }
    }
}

// Backward RMSNorm (no gain): dX = dOut/rms - x * (sum(dOut*x))/(rms^3 * n) ... simplified: dX_i = (dOut_i - (sum dOut*x)/ss) / rms
// where ss = sum x^2, rms = sqrt(ss/n + eps).
pub fn backwardRMSNorm(dX: []f32, x: []const f32, dOut: []const f32, eps: f32) void {
    std.debug.assert(dX.len == x.len and x.len == dOut.len);
    const n = x.len;
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const rms = @sqrt(ss / @as(f32, @floatFromInt(n)) + eps);
    var dot: f32 = 0;
    for (x, dOut) |xi, gi| dot += xi * gi;
    const scale = dot / (ss + eps * @as(f32, @floatFromInt(n)));
    for (dX, x, dOut) |*dx, xi, gi| {
        dx.* += (gi - xi * scale) / rms;
    }
}

// Gradient check: finite difference for f(x)=sum? Here check scalar f.
pub fn finiteDiff(f: *const fn (x: f64) f64, x: f64, eps: f64) f64 {
    return (f(x + eps) - f(x - eps)) / (2.0 * eps);
}

test "Tape record/clear" {
    const allocator = std.testing.allocator;
    var t = Tape.init(allocator);
    defer t.deinit();
    try t.record(.{ .tag = .add, .n_inputs = 2, .n_outputs = 1 });
    try std.testing.expectEqual(@as(usize, 1), t.len());
    t.clear();
    try std.testing.expectEqual(@as(usize, 0), t.len());
}

test "backwardAdd" {
    var dA = [_]f32{ 0, 0 };
    var dB = [_]f32{ 0, 0 };
    const g = [_]f32{ 1, 2 };
    backwardAdd(&dA, &dB, &g);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2 }, &dA);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2 }, &dB);
}

test "backwardMul" {
    var dA = [_]f32{ 0, 0 };
    var dB = [_]f32{ 0, 0 };
    const a = [_]f32{ 2, 3 };
    const b = [_]f32{ 4, 5 };
    const g = [_]f32{ 1, 1 };
    backwardMul(&dA, &dB, &a, &b, &g);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 4, 5 }, &dA);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 2, 3 }, &dB);
}

test "backwardRelu" {
    var dA = [_]f32{ 0, 0, 0 };
    const a = [_]f32{ -1, 2, 0 };
    const g = [_]f32{ 1, 1, 1 };
    backwardRelu(&dA, &a, &g);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 0, 1, 0 }, &dA);
}

test "backwardMatmul shapes" {
    // A 2x2, B 2x2, C 2x2
    var dA = [_]f32{ 0, 0, 0, 0 };
    var dB = [_]f32{ 0, 0, 0, 0 };
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    const dC = [_]f32{ 1, 0, 0, 1 };
    backwardMatmul(&dA, &dB, &A, &B, &dC, 2, 2, 2);
    // dA[0,0] = dC[0,0]*B[0,0]+dC[0,1]*B[0,1] = 5+0=5
    try std.testing.expectApproxEqAbs(@as(f32, 5), dA[0], 1e-5);
}

test "backwardSoftmax" {
    // s uniform 0.25, dOut all 1 -> dX should be 0
    var s = [_]f32{ 0.25, 0.25, 0.25, 0.25 };
    var dOut = [_]f32{ 1, 1, 1, 1 };
    var dX = [_]f32{ 0, 0, 0, 0 };
    backwardSoftmax(&dX, &s, &dOut, 1, 4);
    for (dX) |v| try std.testing.expect(@abs(v) < 1e-5);
}

test "backwardRMSNorm" {
    var x = [_]f32{ 1, 2, 3, 4 };
    var dOut = [_]f32{ 1, 1, 1, 1 };
    var dX = [_]f32{ 0, 0, 0, 0 };
    backwardRMSNorm(&dX, &x, &dOut, 1e-6);
    // sum of dX*x should relate; just check finite
    for (dX) |v| try std.testing.expect(std.math.isFinite(v));
}

// Backward cross-entropy: dLogits = softmax(logits) - one_hot(target).
pub fn backwardCrossEntropy(dLogits: []f32, logits: []const f32, targets: []const u32, rows: usize, vocab: usize) void {
    std.debug.assert(dLogits.len == logits.len and logits.len == rows * vocab);
    std.debug.assert(targets.len == rows);
    // softmax per row
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const base = r * vocab;
        var mx: f32 = logits[base];
        var j: usize = 1;
        while (j < vocab) : (j += 1) if (logits[base + j] > mx) {
            mx = logits[base + j];
        };
        var sum: f32 = 0;
        j = 0;
        while (j < vocab) : (j += 1) {
            dLogits[base + j] = @exp(logits[base + j] - mx);
            sum += dLogits[base + j];
        }
        j = 0;
        while (j < vocab) : (j += 1) {
            dLogits[base + j] /= sum;
            if (j == targets[r]) dLogits[base + j] -= 1.0;
        }
    }
}

// Backward embedding: scatter-add dOut into dTable at ids.
pub fn backwardEmbedding(dTable: []f32, dOut: []const f32, ids: []const u32, vocab: usize, dim: usize) void {
    std.debug.assert(dTable.len == vocab * dim);
    std.debug.assert(dOut.len == ids.len * dim);
    for (ids, 0..) |id, i| {
        std.debug.assert(id < vocab);
        var d: usize = 0;
        while (d < dim) : (d += 1) {
            dTable[id * dim + d] += dOut[i * dim + d];
        }
    }
}

// RoPE forward: rotate pairs (x[2i], x[2i+1]) by angle = pos * theta_i.
// theta_i = 1 / (base^(2i/d)).
pub fn ropeForward(out: []f32, x: []const f32, pos: usize, d: usize, base: f32) void {
    std.debug.assert(out.len == x.len and x.len % 2 == 0);
    const half = x.len / 2;
    var i: usize = 0;
    while (i < half) : (i += 1) {
        const freq = 1.0 / std.math.pow(f32, base, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(d)));
        const angle = @as(f32, @floatFromInt(pos)) * freq;
        const cos = @cos(angle);
        const sin = @sin(angle);
        const a = x[2 * i];
        const b = x[2 * i + 1];
        out[2 * i] = a * cos - b * sin;
        out[2 * i + 1] = a * sin + b * cos;
    }
}

// RoPE backward (inverse rotation).
pub fn ropeBackward(dX: []f32, dOut: []const f32, pos: usize, d: usize, base: f32) void {
    std.debug.assert(dX.len == dOut.len and dX.len % 2 == 0);
    const half = dX.len / 2;
    var i: usize = 0;
    while (i < half) : (i += 1) {
        const freq = 1.0 / std.math.pow(f32, base, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(d)));
        const angle = @as(f32, @floatFromInt(pos)) * freq;
        const cos = @cos(angle);
        const sin = @sin(angle);
        const a = dOut[2 * i];
        const b = dOut[2 * i + 1];
        // inverse: rotate by -angle
        dX[2 * i] += a * cos + b * sin;
        dX[2 * i + 1] += -a * sin + b * cos;
    }
}

// Gradient check harness: finite difference for scalar function f at x, compare to analytic g.
pub fn gradCheckScalar(f: *const fn (x: f64) f64, analytic: f64, x: f64, eps: f64, tol: f64) bool {
    const fd = (f(x + eps) - f(x - eps)) / (2.0 * eps);
    const denom = if (@abs(analytic) > 1e-12) @abs(analytic) else 1.0;
    const rel = @abs(fd - analytic) / denom;
    return rel < tol;
}

test "backwardCrossEntropy" {
    // logits [0,0], target 0 -> softmax [0.5,0.5], dLogit = [0.5-1, 0.5] = [-0.5, 0.5]
    var logits = [_]f32{ 0, 0 };
    var targets = [_]u32{0};
    var dL = [_]f32{ 0, 0 };
    backwardCrossEntropy(&dL, &logits, &targets, 1, 2);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), dL[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), dL[1], 1e-5);
}

test "backwardEmbedding" {
    var table = [_]f32{ 0, 0, 0, 0 };
    var dOut = [_]f32{ 1, 2, 3, 4 };
    var ids = [_]u32{ 1, 0 };
    backwardEmbedding(&table, &dOut, &ids, 2, 2);
    // id0 gets dOut[2..4] = [3,4]; id1 gets dOut[0..2]=[1,2]
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3, 4, 1, 2 }, &table);
}

test "ropeForward/Backward roundtrip" {
    var x = [_]f32{ 1, 0, 0, 1 };
    var out = [_]f32{ 0, 0, 0, 0 };
    ropeForward(&out, &x, 3, 4, 10000.0);
    var dx = [_]f32{ 0, 0, 0, 0 };
    ropeBackward(&dx, &out, 3, 4, 10000.0);
    for (dx, x) |a, b| try std.testing.expectApproxEqAbs(b, a, 1e-4);
}

test "gradCheckScalar" {
    const f = struct {
        fn sq(x: f64) f64 {
            return x * x;
        }
    }.sq;
    // derivative of x^2 at x=2 is 4
    try std.testing.expect(gradCheckScalar(f, 4.0, 2.0, 1e-5, 1e-6));
}
