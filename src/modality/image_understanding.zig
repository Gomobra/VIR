// Image understanding harness (PRD 3.14): build image+text sequences and masked loss.

const std = @import("std");
const layout_mod = @import("modality.zig");
const image_vq_mod = @import("image_vq.zig");
const cpu = @import("../cpu/cpu.zig");

pub const Sequence = struct {
    tokens: []u32,
    loss_mask: []bool,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Sequence) void {
        self.allocator.free(self.tokens);
        self.allocator.free(self.loss_mask);
    }
};

// Convert grayscale image pixels into image token ids using VocabLayout offsets.
pub fn imageTokensFromPixels(allocator: std.mem.Allocator, image: []const f32, ivq: *const image_vq_mod.ImageVQ, layout: layout_mod.VocabLayout) ![]u32 {
    std.debug.assert(image.len == ivq.height * ivq.width);
    const np = ivq.nPatches();
    const raw = try allocator.alloc(usize, np);
    defer allocator.free(raw);
    _ = ivq.encode(image, raw);
    const out = try allocator.alloc(u32, np);
    for (raw, 0..) |c, i| out[i] = @as(u32, @intCast(layout.image_offset + c));
    return out;
}

// Build a multimodal sequence: image tokens, then question tokens, then answer tokens.
// Loss is enabled only on answer tokens.
pub fn buildUnderstandingSequence(allocator: std.mem.Allocator, image_tokens: []const u32, question: []const u32, answer: []const u32) !Sequence {
    const total = image_tokens.len + question.len + answer.len;
    const tokens = try allocator.alloc(u32, total);
    const mask = try allocator.alloc(bool, total);
    @memset(mask, false);
    var pos: usize = 0;
    for (image_tokens) |t| {
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

// Masked cross-entropy over positions where loss_mask is true.
// logits: [seq * vocab], targets: [seq] (target token at same position).
pub fn maskedCrossEntropy(logits: []const f32, targets: []const u32, mask: []const bool, seq: usize, vocab: usize) f32 {
    std.debug.assert(logits.len == seq * vocab);
    std.debug.assert(targets.len == seq);
    std.debug.assert(mask.len == seq);
    var total: f32 = 0;
    var count: usize = 0;
    var i: usize = 0;
    while (i < seq) : (i += 1) {
        if (!mask[i]) continue;
        // softmax row i
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

// Simple accuracy over masked positions using argmax.
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

test "buildUnderstandingSequence" {
    const allocator = std.testing.allocator;
    const img = [_]u32{ 10, 11 };
    const q = [_]u32{ 1, 2 };
    const a = [_]u32{ 3, 4 };
    var seq = try buildUnderstandingSequence(allocator, &img, &q, &a);
    defer seq.deinit();
    try std.testing.expectEqual(@as(usize, 6), seq.tokens.len);
    try std.testing.expectEqual(@as(usize, 2), seq.loss_mask.len);
    // only answer positions masked
    var masked: usize = 0;
    for (seq.loss_mask) |m| if (m) {
        masked += 1;
    };
    try std.testing.expectEqual(@as(usize, 2), masked);
}

test "maskedCrossEntropy finite" {
    const allocator = std.testing.allocator;
    const seq = 3;
    const vocab = 4;
    var logits = [_]f32{0} * *(seq * vocab);
    for (&logits) |*v| v.* = 0.1;
    const targets = [_]u32{ 0, 1, 2 };
    const mask = [_]bool{ false, true, true };
    const loss = maskedCrossEntropy(&logits, &targets, &mask, seq, vocab);
    try std.testing.expect(std.math.isFinite(loss));
    _ = allocator;
}

test "imageTokensFromPixels shape" {
    const allocator = std.testing.allocator;
    var ivq = try image_vq_mod.ImageVQ.init(allocator, 2, 2, 1, 1, 4);
    defer ivq.deinit();
    const layout = layout_mod.defaultLayout();
    var img = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const toks = try imageTokensFromPixels(allocator, &img, &ivq, layout);
    defer allocator.free(toks);
    try std.testing.expectEqual(@as(usize, 4), toks.len);
    for (toks) |t| try std.testing.expect(t >= layout.image_offset);
}
