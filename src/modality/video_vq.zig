// VQ-VAE video tokenizer (PRD 3.19): patch-based encode/decode with conv3d features.

const std = @import("std");
const vq_mod = @import("vq.zig");
const cpu = @import("../cpu/cpu.zig");

pub const VideoVQ = struct {
    t: usize,
    h: usize,
    w: usize,
    patch_t: usize,
    patch_h: usize,
    patch_w: usize,
    n_codes: usize,
    dim: usize,
    conv_w: []f32,
    vq: vq_mod.VQ,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, t: usize, h: usize, w: usize, patch_t: usize, patch_h: usize, patch_w: usize, n_codes: usize) !VideoVQ {
        std.debug.assert(t % patch_t == 0 and h % patch_h == 0 and w % patch_w == 0);
        const dim = patch_t * patch_h * patch_w;
        const vq = try vq_mod.VQ.init(allocator, n_codes, dim);
        const cw = try allocator.alloc(f32, 1 * 1 * 3 * 3 * 3);
        for (cw) |*v| v.* = 0.1;
        return .{
            .t = t,
            .h = h,
            .w = w,
            .patch_t = patch_t,
            .patch_h = patch_h,
            .patch_w = patch_w,
            .n_codes = n_codes,
            .dim = dim,
            .conv_w = cw,
            .vq = vq,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *VideoVQ) void {
        self.allocator.free(self.conv_w);
        self.vq.deinit();
    }

    pub fn nPatches(self: *const VideoVQ) usize {
        return (self.t / self.patch_t) * (self.h / self.patch_h) * (self.w / self.patch_w);
    }

    pub fn encode(self: *const VideoVQ, video: []const f32, out_codes: []usize) usize {
        std.debug.assert(video.len == self.t * self.h * self.w);
        const conv_video = self.allocator.alloc(f32, self.t * self.h * self.w) catch return 0;
        defer self.allocator.free(conv_video);
        cpu.conv3dF32(conv_video, video, self.conv_w, 1, 1, 3, 3, 3, self.t, self.h, self.w);

        const npt = self.t / self.patch_t;
        const nph = self.h / self.patch_h;
        const npw = self.w / self.patch_w;
        const np = npt * nph * npw;
        std.debug.assert(out_codes.len >= np);
        var pi: usize = 0;
        var pt: usize = 0;
        while (pt < npt) : (pt += 1) {
            var py: usize = 0;
            while (py < nph) : (py += 1) {
                var px: usize = 0;
                while (px < npw) : (px += 1) {
                    const patch = self.allocator.alloc(f32, self.dim) catch return pi;
                    defer self.allocator.free(patch);
                    var i: usize = 0;
                    var z: usize = 0;
                    while (z < self.patch_t) : (z += 1) {
                        var y: usize = 0;
                        while (y < self.patch_h) : (y += 1) {
                            var x: usize = 0;
                            while (x < self.patch_w) : (x += 1) {
                                const it = pt * self.patch_t + z;
                                const iy = py * self.patch_h + y;
                                const ix = px * self.patch_w + x;
                                patch[i] = conv_video[it * self.h * self.w + iy * self.w + ix];
                                i += 1;
                            }
                        }
                    }
                    out_codes[pi] = self.vq.quantize(patch);
                    pi += 1;
                }
            }
        }
        return np;
    }

    pub fn decode(self: *const VideoVQ, codes: []const usize, out_video: []f32) void {
        std.debug.assert(out_video.len == self.t * self.h * self.w);
        @memset(out_video, 0);
        const npt = self.t / self.patch_t;
        const nph = self.h / self.patch_h;
        const npw = self.w / self.patch_w;
        var pi: usize = 0;
        var pt: usize = 0;
        while (pt < npt) : (pt += 1) {
            var py: usize = 0;
            while (py < nph) : (py += 1) {
                var px: usize = 0;
                while (px < npw) : (px += 1) {
                    const code = codes[pi];
                    std.debug.assert(code < self.n_codes);
                    var i: usize = 0;
                    var z: usize = 0;
                    while (z < self.patch_t) : (z += 1) {
                        var y: usize = 0;
                        while (y < self.patch_h) : (y += 1) {
                            var x: usize = 0;
                            while (x < self.patch_w) : (x += 1) {
                                const it = pt * self.patch_t + z;
                                const iy = py * self.patch_h + y;
                                const ix = px * self.patch_w + x;
                                out_video[it * self.h * self.w + iy * self.w + ix] = self.vq.codebook[code * self.dim + i];
                                i += 1;
                            }
                        }
                    }
                    pi += 1;
                }
            }
        }
        // smoothing conv block
        const conv_video = self.allocator.alloc(f32, self.t * self.h * self.w) catch return;
        defer self.allocator.free(conv_video);
        cpu.conv3dF32(conv_video, out_video, self.conv_w, 1, 1, 3, 3, 3, self.t, self.h, self.w);
        @memcpy(out_video, conv_video);
    }
};

test "VideoVQ encode shape" {
    const allocator = std.testing.allocator;
    var vvq = try VideoVQ.init(allocator, 2, 2, 2, 1, 1, 1, 4);
    defer vvq.deinit();
    var vid = [_]f32{0} * *8;
    for (&vid, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1;
    var codes = [_]usize{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const np = vvq.encode(&vid, &codes);
    try std.testing.expectEqual(@as(usize, 8), np);
    for (codes) |c| try std.testing.expect(c < 4);
}

test "VideoVQ decode finite" {
    const allocator = std.testing.allocator;
    var vvq = try VideoVQ.init(allocator, 2, 2, 2, 1, 1, 1, 4);
    defer vvq.deinit();
    const codes = [_]usize{ 0, 1, 2, 3, 0, 1, 2, 3 };
    var out = [_]f32{0} * *8;
    vvq.decode(&codes, &out);
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}

test "VideoVQ deterministic parity" {
    const allocator = std.testing.allocator;
    var vvq = try VideoVQ.init(allocator, 2, 2, 2, 1, 1, 1, 4);
    defer vvq.deinit();
    var vid = [_]f32{0} * *8;
    for (&vid, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.05;
    var c1 = [_]usize{0} * *8;
    var c2 = [_]usize{0} * *8;
    _ = vvq.encode(&vid, &c1);
    _ = vvq.encode(&vid, &c2);
    try std.testing.expectEqualSlices(usize, &c1, &c2);
}
