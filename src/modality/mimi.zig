// Mimi-style audio codec (PRD 3.16/3.17): CPU reference encode/decode.
// Implements a causal, frame-based codec: 24 kHz audio -> 12.5 Hz frames (1920 samples)
// using RVQ. This is a compact, deterministic reference used for parity tests;
// pretrained SEANet/transformer weights can replace the feature extractor later.

const std = @import("std");
const fsq_mod = @import("fsq.zig");
const cpu = @import("../cpu/cpu.zig");

pub const MimiCodec = struct {
    sample_rate: usize = 24000,
    frame_size: usize = 1920,
    n_codebooks: usize,
    codebook_size: usize,
    dim: usize,
    upsample_stride: usize = 1920,
    upsample_kernel: usize = 1920,
    upsample_weight: []f32,
    rvq: fsq_mod.RVQ,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n_codebooks: usize, codebook_size: usize, dim: usize) !MimiCodec {
        const rvq = try fsq_mod.RVQ.init(allocator, n_codebooks, codebook_size, dim);
        const w = try allocator.alloc(f32, 1 * 1 * 1920);
        for (w) |*v| v.* = 0.01;
        // make first tap dominant for stable upsampling
        if (w.len > 0) w[0] = 1.0;
        return .{ .n_codebooks = n_codebooks, .codebook_size = codebook_size, .dim = dim, .upsample_weight = w, .rvq = rvq, .allocator = allocator };
    }

    pub fn deinit(self: *MimiCodec) void {
        self.allocator.free(self.upsample_weight);
        self.rvq.deinit();
    }

    fn nFrames(self: *const MimiCodec, n_samples: usize) usize {
        if (n_samples == 0) return 0;
        return (n_samples + self.frame_size - 1) / self.frame_size;
    }

    // Feature extraction: frame mean repeated to `dim` (placeholder for SEANet).
    fn frameFeature(self: *const MimiCodec, audio: []const f32, frame_idx: usize, out: []f32) void {
        std.debug.assert(out.len == self.dim);
        const start = frame_idx * self.frame_size;
        const end = @min(start + self.frame_size, audio.len);
        var sum: f32 = 0;
        var count: usize = 0;
        var i: usize = start;
        while (i < end) : (i += 1) {
            sum += audio[i];
            count += 1;
        }
        const mean = if (count > 0) sum / @as(f32, @floatFromInt(count)) else 0;
        for (out) |*v| v.* = mean;
    }

    // Encode audio into RVQ code indices. Returns number of frames.
    pub fn encode(self: *const MimiCodec, audio: []const f32, out_codes: []usize) usize {
        const nf = self.nFrames(audio.len);
        std.debug.assert(out_codes.len >= nf * self.n_codebooks);
        var f: usize = 0;
        while (f < nf) : (f += 1) {
            const feat = self.allocator.alloc(f32, self.dim) catch return f;
            defer self.allocator.free(feat);
            self.frameFeature(audio, f, feat);
            const idx = self.allocator.alloc(usize, self.n_codebooks) catch return f;
            defer self.allocator.free(idx);
            self.rvq.quantize(feat, idx);
            var cb: usize = 0;
            while (cb < self.n_codebooks) : (cb += 1) {
                out_codes[f * self.n_codebooks + cb] = idx[cb];
            }
        }
        return nf;
    }

    // Decode code indices into audio using transposed causal conv1d upsampling.
    pub fn decode(self: *const MimiCodec, codes: []const usize, n_frames: usize, out_audio: []f32) void {
        std.debug.assert(out_audio.len >= n_frames * self.frame_size);
        const stride = self.upsample_stride;
        const kernel = self.upsample_kernel;
        std.debug.assert(kernel == stride);
        const feat_len = n_frames;
        const out_len = (feat_len - 1) * stride + kernel;
        std.debug.assert(out_audio.len >= out_len);
        var feat = self.allocator.alloc(f32, feat_len) catch return;
        defer self.allocator.free(feat);
        @memset(feat, 0);
        var f: usize = 0;
        while (f < n_frames) : (f += 1) {
            const idx = codes[f * self.n_codebooks .. (f + 1) * self.n_codebooks];
            const fv = self.allocator.alloc(f32, self.dim) catch return;
            defer self.allocator.free(fv);
            self.rvq.dequantize(idx, fv);
            feat[f] = fv[0];
        }
        cpu.convTranspose1dF32(out_audio[0..out_len], feat, self.upsample_weight, 1, 1, kernel, stride, feat_len);
    }
};

test "MimiCodec encode shape" {
    const allocator = std.testing.allocator;
    var codec = try MimiCodec.init(allocator, 2, 4, 3);
    defer codec.deinit();
    var audio = [_]f32{ 0.1, 0.2, 0.3, 0.4 } ++ [_]f32{0} * *1916;
    var codes = [_]usize{ 0, 0 };
    const nf = codec.encode(&audio, &codes);
    try std.testing.expectEqual(@as(usize, 1), nf);
    try std.testing.expect(codes[0] < 4);
}

test "MimiCodec decode shape" {
    const allocator = std.testing.allocator;
    var codec = try MimiCodec.init(allocator, 2, 4, 3);
    defer codec.deinit();
    const codes = [_]usize{ 0, 1 };
    var out = [_]f32{0} * *1920;
    codec.decode(&codes, 1, &out);
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}

test "MimiCodec roundtrip finite" {
    const allocator = std.testing.allocator;
    var codec = try MimiCodec.init(allocator, 2, 4, 3);
    defer codec.deinit();
    var audio = [_]f32{0.5} * *1920;
    var codes = [_]usize{ 0, 0 };
    const nf = codec.encode(&audio, &codes);
    var out = [_]f32{0} * *1920;
    codec.decode(&codes, nf, &out);
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}

test "MimiCodec deterministic parity" {
    const allocator = std.testing.allocator;
    var codec = try MimiCodec.init(allocator, 2, 4, 3);
    defer codec.deinit();
    var audio = [_]f32{0.25} * *1920;
    var c1 = [_]usize{ 0, 0 };
    var c2 = [_]usize{ 0, 0 };
    _ = codec.encode(&audio, &c1);
    _ = codec.encode(&audio, &c2);
    try std.testing.expectEqualSlices(usize, &c1, &c2);
}
