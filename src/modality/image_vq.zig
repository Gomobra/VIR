// VQ-VAE image tokenizer (PRD 3.15): patch-based encode/decode with VQ codebook.

const std = @import("std");
const vq_mod = @import("vq.zig");
const cpu = @import("../cpu/cpu.zig");

pub const ImageVQ = struct {
    height: usize,
    width: usize,
    patch_h: usize,
    patch_w: usize,
    n_codes: usize,
    dim: usize,
    conv_w: []f32,
    conv_b: []f32,
    vq: vq_mod.VQ,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, height: usize, width: usize, patch_h: usize, patch_w: usize, n_codes: usize) !ImageVQ {
        std.debug.assert(height % patch_h == 0 and width % patch_w == 0);
        const dim = patch_h * patch_w;
        const vq = try vq_mod.VQ.init(allocator, n_codes, dim);
        const cw = try allocator.alloc(f32, 1 * 1 * 3 * 3);
        for (cw) |*v| v.* = 0.1;
        const cb = try allocator.alloc(f32, 1);
        cb[0] = 0.0;
        return .{
            .height = height,
            .width = width,
            .patch_h = patch_h,
            .patch_w = patch_w,
            .n_codes = n_codes,
            .dim = dim,
            .conv_w = cw,
            .conv_b = cb,
            .vq = vq,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ImageVQ) void {
        self.allocator.free(self.conv_w);
        self.allocator.free(self.conv_b);
        self.vq.deinit();
    }

    pub fn nPatches(self: *const ImageVQ) usize {
        return (self.height / self.patch_h) * (self.width / self.patch_w);
    }

    // Encode grayscale image [H*W] into patch codes.
    pub fn encode(self: *const ImageVQ, image: []const f32, out_codes: []usize) usize {
        std.debug.assert(image.len == self.height * self.width);
        const conv_img = self.allocator.alloc(f32, self.height * self.width) catch return 0;
        defer self.allocator.free(conv_img);
        cpu.conv2dF32(conv_img, image, self.conv_w, 1, 1, 3, 3, self.height, self.width);
        const nph = self.height / self.patch_h;
        const npw = self.width / self.patch_w;
        const np = nph * npw;
        std.debug.assert(out_codes.len >= np);
        var pi: usize = 0;
        var py: usize = 0;
        while (py < nph) : (py += 1) {
            var px: usize = 0;
            while (px < npw) : (px += 1) {
                const patch = self.allocator.alloc(f32, self.dim) catch return pi;
                defer self.allocator.free(patch);
                var i: usize = 0;
                var y: usize = 0;
                while (y < self.patch_h) : (y += 1) {
                    var x: usize = 0;
                    while (x < self.patch_w) : (x += 1) {
                        const iy = py * self.patch_h + y;
                        const ix = px * self.patch_w + x;
                        patch[i] = conv_img[iy * self.width + ix];
                        i += 1;
                    }
                }
                out_codes[pi] = self.vq.quantize(patch);
                pi += 1;
            }
        }
        return np;
    }

    // Decode patch codes into grayscale image [H*W].
    pub fn decode(self: *const ImageVQ, codes: []const usize, out_image: []f32) void {
        std.debug.assert(out_image.len == self.height * self.width);
        @memset(out_image, 0);
        const nph = self.height / self.patch_h;
        const npw = self.width / self.patch_w;
        var pi: usize = 0;
        var py: usize = 0;
        while (py < nph) : (py += 1) {
            var px: usize = 0;
            while (px < npw) : (px += 1) {
                const code = codes[pi];
                std.debug.assert(code < self.n_codes);
                var i: usize = 0;
                var y: usize = 0;
                while (y < self.patch_h) : (y += 1) {
                    var x: usize = 0;
                    while (x < self.patch_w) : (x += 1) {
                        const iy = py * self.patch_h + y;
                        const ix = px * self.patch_w + x;
                        out_image[iy * self.width + ix] = self.vq.codebook[code * self.dim + i];
                        i += 1;
                    }
                }
                pi += 1;
            }
        }
        // Apply a conv block for reconstruction smoothing.
        const conv_img = self.allocator.alloc(f32, self.height * self.width) catch return;
        defer self.allocator.free(conv_img);
        cpu.conv2dF32(conv_img, out_image, self.conv_w, 1, 1, 3, 3, self.height, self.width);
        @memcpy(out_image, conv_img);
    }
};

test "ImageVQ encode shape" {
    const allocator = std.testing.allocator;
    var ivq = try ImageVQ.init(allocator, 4, 4, 2, 2, 4);
    defer ivq.deinit();
    var img = [_]f32{0} * *16;
    for (&img, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1;
    var codes = [_]usize{ 0, 0, 0, 0 };
    const np = ivq.encode(&img, &codes);
    try std.testing.expectEqual(@as(usize, 4), np);
    for (codes) |c| try std.testing.expect(c < 4);
}

test "ImageVQ decode finite" {
    const allocator = std.testing.allocator;
    var ivq = try ImageVQ.init(allocator, 4, 4, 2, 2, 4);
    defer ivq.deinit();
    const codes = [_]usize{ 0, 1, 2, 3 };
    var out = [_]f32{0} * *16;
    ivq.decode(&codes, &out);
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}

test "ImageVQ deterministic parity" {
    const allocator = std.testing.allocator;
    var ivq = try ImageVQ.init(allocator, 4, 4, 2, 2, 4);
    defer ivq.deinit();
    var img = [_]f32{0} * *16;
    for (&img, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.05;
    var c1 = [_]usize{ 0, 0, 0, 0 };
    var c2 = [_]usize{ 0, 0, 0, 0 };
    _ = ivq.encode(&img, &c1);
    _ = ivq.encode(&img, &c2);
    try std.testing.expectEqualSlices(usize, &c1, &c2);
}
