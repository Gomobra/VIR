// Audio understanding harness (PRD 3.16): build audio+text sequences and masked loss.

const std = @import("std");
const layout_mod = @import("modality.zig");
const mimi_mod = @import("mimi.zig");

pub const Sequence = struct {
    tokens: []u32,
    loss_mask: []bool,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Sequence) void {
        self.allocator.free(self.tokens);
        self.allocator.free(self.loss_mask);
    }
};

pub fn audioTokensFromSamples(allocator: std.mem.Allocator, audio: []const f32, codec: *const mimi_mod.MimiCodec, layout: layout_mod.VocabLayout) ![]u32 {
    const nf = codec.nFrames(audio.len);
    const n_codes = nf * codec.n_codebooks;
    const raw = try allocator.alloc(usize, n_codes);
    defer allocator.free(raw);
    _ = codec.encode(audio, raw);
    const out = try allocator.alloc(u32, n_codes);
    for (raw, 0..) |c, i| out[i] = @as(u32, @intCast(layout.audio_offset + c));
    return out;
}

pub fn buildAudioUnderstandingSequence(allocator: std.mem.Allocator, audio_tokens: []const u32, question: []const u32, answer: []const u32) !Sequence {
    const total = audio_tokens.len + question.len + answer.len;
    const tokens = try allocator.alloc(u32, total);
    const mask = try allocator.alloc(bool, total);
    @memset(mask, false);
    var pos: usize = 0;
    for (audio_tokens) |t| {
        tokens[pos] = t;
        pos += 1;
    }
    for (question) |t| {
        tokens[pos] = t;
        pos += 1;
    }
    for (answer) |t| {
        tokens[pos] = t;
        mask[pos] = true;
        pos += 1;
    }
    return .{ .tokens = tokens, .loss_mask = mask, .allocator = allocator };
}

pub fn maskedCrossEntropy(logits: []const f32, targets: []const u32, mask: []const bool, seq: usize, vocab: usize) f32 {
    std.debug.assert(logits.len == seq * vocab);
    std.debug.assert(targets.len == seq);
    std.debug.assert(mask.len == seq);
    var total: f32 = 0;
    var count: usize = 0;
    var i: usize = 0;
    while (i < seq) : (i += 1) {
        if (!mask[i]) continue;
        var mx: f32 = logits[i * vocab];
        var j: usize = 1;
        while (j < vocab) : (j += 1) {
            if (logits[i * vocab + j] > mx) mx = logits[i * vocab + j];
        }
        var sum: f32 = 0;
        j = 0;
        while (j < vocab) : (j += 1) sum += @exp(logits[i * vocab + j] - mx);
        const p = @exp(logits[i * vocab + targets[i]] - mx) / sum;
        total += -@log(p + 1e-12);
        count += 1;
    }
    if (count == 0) return 0;
    return total / @as(f32, @floatFromInt(count));
}

pub fn maskedAccuracy(logits: []const f32, targets: []const u32, mask: []const bool, seq: usize, vocab: usize) f32 {
    std.debug.assert(logits.len == seq * vocab);
    std.debug.assert(targets.len == seq);
    std.debug.assert(mask.len == seq);
    var correct: usize = 0;
    var count: usize = 0;
    var i: usize = 0;
    while (i < seq) : (i += 1) {
        if (!mask[i]) continue;
        var best: usize = 0;
        var bestv: f32 = logits[i * vocab];
        var j: usize = 1;
        while (j < vocab) : (j += 1) {
            if (logits[i * vocab + j] > bestv) {
                bestv = logits[i * vocab + j];
                best = j;
            }
        }
        if (best == targets[i]) correct += 1;
        count += 1;
    }
    if (count == 0) return 0;
    return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(count));
}

test "buildAudioUnderstandingSequence" {
    const allocator = std.testing.allocator;
    const aud = [_]u32{ 10, 11 };
    const q = [_]u32{ 1, 2 };
    const a = [_]u32{ 3, 4 };
    var seq = try buildAudioUnderstandingSequence(allocator, &aud, &q, &a);
    defer seq.deinit();
    try std.testing.expectEqual(@as(usize, 6), seq.tokens.len);
    var masked: usize = 0;
    for (seq.loss_mask) |m| if (m) {
        masked += 1;
    };
    try std.testing.expectEqual(@as(usize, 2), masked);
}

test "audio maskedCrossEntropy finite" {
    const seq = 3;
    const vocab = 4;
    var logits = [_]f32{0} * *(seq * vocab);
    for (&logits) |*v| v.* = 0.1;
    const targets = [_]u32{ 0, 1, 2 };
    const mask = [_]bool{ false, true, true };
    const loss = maskedCrossEntropy(&logits, &targets, &mask, seq, vocab);
    try std.testing.expect(std.math.isFinite(loss));
}

test "audioTokensFromSamples shape" {
    const allocator = std.testing.allocator;
    var codec = try mimi_mod.MimiCodec.init(allocator, 2, 4, 3);
    defer codec.deinit();
    const layout = layout_mod.defaultLayout();
    var audio = [_]f32{0.1} * *1920;
    const toks = try audioTokensFromSamples(allocator, &audio, &codec, layout);
    defer allocator.free(toks);
    try std.testing.expect(toks.len > 0);
    for (toks) |t| try std.testing.expect(t >= layout.audio_offset);
}
