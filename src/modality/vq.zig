// VQ-VAE codebook (PRD 1.5, 3.15): EMA update + dead-code restart.

const std = @import("std");

pub const VQ = struct {
    n_codes: usize,
    dim: usize,
    codebook: []f32,
    ema_count: []f32,
    ema_sum: []f32,
    decay: f32 = 0.99,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n_codes: usize, dim: usize) !VQ {
        const cb = try allocator.alloc(f32, n_codes * dim);
        for (cb, 0..) |*x, i| x.* = 0.01 * @as(f32, @floatFromInt(i % 13)) - 0.06;
        const cnt = try allocator.alloc(f32, n_codes);
        const sm = try allocator.alloc(f32, n_codes * dim);
        @memset(cnt, 0);
        @memset(sm, 0);
        return .{ .n_codes = n_codes, .dim = dim, .codebook = cb, .ema_count = cnt, .ema_sum = sm, .allocator = allocator };
    }

    pub fn deinit(self: *VQ) void {
        self.allocator.free(self.codebook);
        self.allocator.free(self.ema_count);
        self.allocator.free(self.ema_sum);
    }

    pub fn quantize(self: *const VQ, x: []const f32) usize {
        std.debug.assert(x.len == self.dim);
        var best: usize = 0;
        var best_dist: f32 = std.math.inf(f32);
        var e: usize = 0;
        while (e < self.n_codes) : (e += 1) {
            var dist: f32 = 0;
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                const diff = x[d] - self.codebook[e * self.dim + d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best = e;
            }
        }
        return best;
    }

    pub fn emaUpdate(self: *VQ, xs: []const f32, codes: []const usize, n: usize) void {
        std.debug.assert(xs.len == n * self.dim);
        std.debug.assert(codes.len == n);
        for (codes, 0..) |c, i| {
            self.ema_count[c] += 1;
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                self.ema_sum[c * self.dim + d] += xs[i * self.dim + d];
            }
        }
        var e: usize = 0;
        while (e < self.n_codes) : (e += 1) {
            if (self.ema_count[e] > 0) {
                var d: usize = 0;
                while (d < self.dim) : (d += 1) {
                    self.codebook[e * self.dim + d] = self.ema_sum[e * self.dim + d] / self.ema_count[e];
                }
            }
        }
    }

    pub fn restartDead(self: *VQ, data: []const f32, n_data: usize) void {
        var e: usize = 0;
        while (e < self.n_codes) : (e += 1) {
            if (self.ema_count[e] == 0 and n_data > 0) {
                const src = e % n_data;
                var d: usize = 0;
                while (d < self.dim) : (d += 1) {
                    self.codebook[e * self.dim + d] = data[src * self.dim + d];
                }
                self.ema_count[e] = 1;
            }
        }
    }
};

test "VQ quantize" {
    const allocator = std.testing.allocator;
    var vq = try VQ.init(allocator, 4, 3);
    defer vq.deinit();
    const x = [_]f32{ 0.1, 0.2, 0.3 };
    const idx = vq.quantize(&x);
    try std.testing.expect(idx < 4);
}

test "VQ emaUpdate" {
    const allocator = std.testing.allocator;
    var vq = try VQ.init(allocator, 2, 2);
    defer vq.deinit();
    const xs = [_]f32{ 1, 2, 3, 4 };
    const codes = [_]usize{ 0, 1 };
    vq.emaUpdate(&xs, &codes, 2);
    try std.testing.expect(vq.codebook[0] > 0);
}
