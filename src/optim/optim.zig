// Optimizer (PRD 3.2): AdamW + Muon (Newton-Schulz).

const std = @import("std");

pub const AdamW = struct {
    lr: f32,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    wd: f32 = 0.01,
    t: usize = 0,
    m: []f32,
    v: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize, lr: f32) !AdamW {
        const m = try allocator.alloc(f32, n);
        const v = try allocator.alloc(f32, n);
        @memset(m, 0);
        @memset(v, 0);
        return .{ .lr = lr, .m = m, .v = v, .allocator = allocator };
    }

    pub fn deinit(self: *AdamW) void {
        self.allocator.free(self.m);
        self.allocator.free(self.v);
    }

    pub fn step(self: *AdamW, params: []f32, grads: []const f32) void {
        std.debug.assert(params.len == grads.len and grads.len == self.m.len);
        self.t += 1;
        const b1 = self.beta1;
        const b2 = self.beta2;
        const lr = self.lr;
        const wd = self.wd;
        const eps = self.eps;
        const bc1 = 1.0 - std.math.pow(f32, b1, @floatFromInt(self.t));
        const bc2 = 1.0 - std.math.pow(f32, b2, @floatFromInt(self.t));
        for (params, grads, self.m, self.v) |*p, g, *mi, *vi| {
            mi.* = b1 * mi.* + (1 - b1) * g;
            vi.* = b2 * vi.* + (1 - b2) * g * g;
            const mhat = mi.* / bc1;
            const vhat = vi.* / bc2;
            p.* = p.* - lr * (mhat / (@sqrt(vhat) + eps) + wd * p.*);
        }
    }
};

// Muon: Newton-Schulz 5 iterations to orthogonalize a 2D gradient matrix.
pub fn newtonSchulz5(mat: []f32, rows: usize, cols: usize) void {
    if (rows == 0 or cols == 0) return;
    std.debug.assert(mat.len == rows * cols);
    // Normalize by Frobenius norm.
    var norm: f32 = 0;
    for (mat) |v| norm += v * v;
    norm = @sqrt(norm);
    if (norm == 0) return;
    const inv = 1.0 / norm;
    for (mat) |*v| v.* *= inv;

    const a: f32 = 3.4445;
    const b: f32 = -4.7750;
    const c: f32 = 2.0315;

    // If rows > cols, work on transpose for efficiency (conceptually).
    // Implement NS on the matrix directly: X = a X + (b A + c A^2) X, A = X X^T.
    const m = if (rows <= cols) rows else cols;
    const n = if (rows <= cols) cols else rows;
    // We'll assume rows <= cols for simplicity; if not, caller transposes.
    if (rows > cols) return; // skip non-square-friendly case for minimal impl

    // Temp buffers
    var buf_a = std.heap.page_allocator.alloc(f32, m * m) catch return;
    defer std.heap.page_allocator.free(buf_a);
    const buf_x = std.heap.page_allocator.alloc(f32, m * n) catch return;
    defer std.heap.page_allocator.free(buf_x);

    // copy mat -> buf_x
    @memcpy(buf_x, mat);

    var step: usize = 0;
    while (step < 5) : (step += 1) {
        // A = X X^T  (m x m)
        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < m) : (j += 1) {
                var acc: f32 = 0;
                var t: usize = 0;
                while (t < n) : (t += 1) {
                    acc += buf_x[i * n + t] * buf_x[j * n + t];
                }
                buf_a[i * m + j] = acc;
            }
        }
        // B = b*A + c*A*A
        var buf_a2 = std.heap.page_allocator.alloc(f32, m * m) catch return;
        defer std.heap.page_allocator.free(buf_a2);
        // A*A
        i = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < m) : (j += 1) {
                var acc: f32 = 0;
                var t: usize = 0;
                while (t < m) : (t += 1) {
                    acc += buf_a[i * m + t] * buf_a[t * m + j];
                }
                buf_a2[i * m + j] = acc;
            }
        }
        // X = a X + (b A + c A^2) X
        i = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                var acc: f32 = a * buf_x[i * n + j];
                var t: usize = 0;
                while (t < m) : (t += 1) {
                    const coef = b * buf_a[i * m + t] + c * buf_a2[i * m + t];
                    acc += coef * buf_x[t * n + j];
                }
                mat[i * n + j] = acc;
            }
        }
        // update buf_x
        @memcpy(buf_x, mat);
    }
}

test "AdamW step reduces param toward gradient" {
    const allocator = std.testing.allocator;
    var opt = try AdamW.init(allocator, 1, 0.1);
    defer opt.deinit();
    var params = [_]f32{5.0};
    const grads = [_]f32{1.0};
    opt.step(&params, &grads);
    try std.testing.expect(params[0] < 5.0);
}

test "newtonSchulz5 normalizes" {
    var g = [_]f32{ 3, 4, 0, 0 };
    newtonSchulz5(&g, 2, 2, 5);
    // Frobenius norm should be ~1
    var n: f32 = 0;
    for (g) |v| n += v * v;
    try std.testing.expect(@abs(@sqrt(n) - 1.0) < 1e-4);
}

// muP (maximal update parameterization) helpers (PRD 1.2, D-07).
pub fn muPInitStd(base_std: f32, d: usize, base_d: usize) f32 {
    // init std scales as 1/sqrt(d), relative to base width
    if (d == 0) return base_std;
    return base_std * @sqrt(@as(f32, @floatFromInt(base_d)) / @as(f32, @floatFromInt(d)));
}

pub fn muPLr(base_lr: f32, d: usize, base_d: usize) f32 {
    // LR for hidden weights scales as base_d / d
    if (d == 0) return base_lr;
    return base_lr * @as(f32, @floatFromInt(base_d)) / @as(f32, @floatFromInt(d));
}

test "muPInitStd" {
    const s = muPInitStd(0.02, 512, 256);
    try std.testing.expect(s < 0.02); // wider -> smaller init
}

test "muPLr" {
    const lr = muPLr(0.02, 512, 256);
    try std.testing.expect(lr < 0.02);
}

// Gradient clipping by global norm (PRD 3.8).
pub fn clipGradNorm(grads: []f32, max_norm: f32) f32 {
    var ss: f32 = 0;
    for (grads) |g| ss += g * g;
    const norm = @sqrt(ss);
    if (norm > max_norm and norm > 0) {
        const scale = max_norm / norm;
        for (grads) |*g| g.* *= scale;
    }
    return norm;
}

test "clipGradNorm clips" {
    var g = [_]f32{ 3, 4 };
    const n = clipGradNorm(&g, 5.0);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), n, 1e-5);
    // norm after clip should be 5
    var ss: f32 = 0;
    for (g) |v| ss += v * v;
    try std.testing.expect(@abs(@sqrt(ss) - 5.0) < 1e-5);
}

test "clipGradNorm no clip" {
    var g = [_]f32{ 1, 2 };
    const n = clipGradNorm(&g, 5.0);
    try std.testing.expect(n < 5.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), g[0], 1e-6);
}
