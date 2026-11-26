// FSQ (finite scalar quantization, PRD 6.0, arXiv 2309.15505).

const std = @import("std");

pub const FSQ = struct {
    levels: usize, // per dimension levels (e.g. 8)
    // quantize: round to nearest level in [-levels/2, levels/2]
    pub fn quantize(self: FSQ, x: f32) f32 {
        const half = @as(f32, @floatFromInt(self.levels)) / 2.0;
        var v = x * half;
        v = @round(v);
        v = @max(-half, @min(half, v));
        return v / half;
    }

    pub fn dequantize(self: FSQ, q: f32) f32 {
        return q; // FSQ is identity-ish after quantize
    }
};

pub const RVQ = struct {
    // residual vector quantization: multiple codebooks
    n_codebooks: usize,
    codebook_size: usize,
    // codebooks: [n_codebooks][codebook_size][dim] flattened
    dim: usize,
    codebooks: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n_codebooks: usize, codebook_size: usize, dim: usize) !RVQ {
        const total = n_codebooks * codebook_size * dim;
        const cb = try allocator.alloc(f32, total);
        for (cb, 0..) |*x, i| x.* = 0.01 * @as(f32, @floatFromInt(i % 13)) - 0.06;
        return .{ .n_codebooks = n_codebooks, .codebook_size = codebook_size, .dim = dim, .codebooks = cb, .allocator = allocator };
    }

    pub fn deinit(self: *RVQ) void {
        self.allocator.free(self.codebooks);
    }

    // quantize a vector: returns codebook indices per codebook
    pub fn quantize(self: *const RVQ, x: []const f32, out_indices: []usize) void {
        std.debug.assert(x.len == self.dim);
        std.debug.assert(out_indices.len == self.n_codebooks);
        var residual = self.allocator.alloc(f32, self.dim) catch return;
        defer self.allocator.free(residual);
        @memcpy(residual, x);
        var cb: usize = 0;
        while (cb < self.n_codebooks) : (cb += 1) {
            // find nearest codebook entry
            var best: usize = 0;
            var best_dist: f32 = std.math.inf(f32);
            var e: usize = 0;
            while (e < self.codebook_size) : (e += 1) {
                var dist: f32 = 0;
                var d: usize = 0;
                while (d < self.dim) : (d += 1) {
                    const diff = residual[d] - self.codebooks[(cb * self.codebook_size + e) * self.dim + d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best = e;
                }
            }
            out_indices[cb] = best;
            // subtract
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                residual[d] -= self.codebooks[(cb * self.codebook_size + best) * self.dim + d];
            }
        }
    }
};

test "FSQ quantize range" {
    const fsq = FSQ{ .levels = 8 };
    const q = fsq.quantize(0.5);
    try std.testing.expect(q >= -1.0 and q <= 1.0);
    const q2 = fsq.quantize(10.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), q2, 1e-6);
}

test "RVQ quantize" {
    const allocator = std.testing.allocator;
    var rvq = try RVQ.init(allocator, 2, 4, 3);
    defer rvq.deinit();
    const x = [_]f32{ 0.1, 0.2, 0.3 };
    var idx = [_]usize{ 0, 0 };
    rvq.quantize(&x, &idx);
    try std.testing.expect(idx[0] < 4);
    try std.testing.expect(idx[1] < 4);
}
