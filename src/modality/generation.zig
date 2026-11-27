// Multimodal generation/eval harness (PRD 3.12, 3.13): sampling with per-segment logit masking.

const std = @import("std");
const layout_mod = @import("modality.zig");
const infer = @import("../infer/infer.zig");

pub const Segment = enum { text, image, audio, video, special };

// Restrict logits to the vocabulary slice for the active modality segment.
pub fn applySegmentMask(logits: []f32, segment: Segment, layout: layout_mod.VocabLayout) void {
    const total = layout.total();
    var i: usize = 0;
    while (i < logits.len and i < total) : (i += 1) {
        const allowed = switch (segment) {
            .text => i < layout.text_size or (i >= layout.special_offset and i < total),
            .image => i >= layout.image_offset and i < layout.image_offset + layout.image_size,
            .audio => i >= layout.audio_offset and i < layout.audio_offset + layout.audio_size,
            .video => i >= layout.video_offset and i < layout.video_offset + layout.video_size,
            .special => i >= layout.special_offset and i < layout.special_offset + layout.special_size,
        };
        if (!allowed) logits[i] = -std.math.inf(f32);
    }
}

// Sample one token from masked logits.
pub fn sampleNext(allocator: std.mem.Allocator, logits: []const f32, temperature: f32, top_k: usize, top_p: f32, u: f32) usize {
    const work = allocator.alloc(f32, logits.len) catch return 0;
    defer allocator.free(work);
    @memcpy(work, logits);
    infer.temperatureScale(work, temperature);
    if (top_k > 0 and top_k < work.len) infer.topK(work, top_k);
    infer.topP(work, top_p);
    infer.softmax(work);
    return infer.sampleCategorical(work, u);
}

// Greedy/multinomial generation where every new token uses the same segment mask.
pub fn generate(allocator: std.mem.Allocator, model: anytype, prompt: []const u32, max_new: usize, segment: Segment, layout: layout_mod.VocabLayout, temperature: f32, top_k: usize, top_p: f32, u: f32) ![]u32 {
    const v = layout.total();
    var seq = std.ArrayList(u32).init(allocator);
    defer seq.deinit();
    try seq.appendSlice(prompt);

    var step: usize = 0;
    while (step < max_new) : (step += 1) {
        const logits = try allocator.alloc(f32, seq.items.len * v);
        defer allocator.free(logits);
        model.forward(logits, seq.items);
        const last = logits[(seq.items.len - 1) * v .. seq.items.len * v];
        applySegmentMask(last, segment, layout);
        const next = sampleNext(allocator, last, temperature, top_k, top_p, u);
        try seq.append(next);
    }
    return seq.toOwnedSlice();
}

// Token accuracy between generated and reference (prefix length).
pub fn tokenAccuracy(generated: []const u32, reference: []const u32) f32 {
    if (reference.len == 0) return 0;
    const n = @min(generated.len, reference.len);
    var correct: usize = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (generated[i] == reference[i]) correct += 1;
    }
    return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(reference.len));
}

test "applySegmentMask text" {
    const layout = layout_mod.defaultLayout();
    var logits = [_]f32{ 1, 2, 3, 4 };
    // shrink layout? Use indices 0..3 all text
    applySegmentMask(&logits, .text, layout);
    for (logits) |v| try std.testing.expect(std.math.isFinite(v));
}

test "applySegmentMask image" {
    const layout = layout_mod.defaultLayout();
    var logits = [_]f32{0} * *73984;
    for (&logits) |*v| v.* = 1;
    applySegmentMask(&logits, .image, layout);
    // image slice finite, text slice -inf
    try std.testing.expect(std.math.isFinite(logits[layout.image_offset]));
    try std.testing.expect(std.math.isInf(logits[0]));
}

test "sampleNext finite" {
    const logits = [_]f32{ 1, 2, 3, 4 };
    const allocator = std.testing.allocator;
    const idx = sampleNext(allocator, &logits, 1.0, 4, 1.0, 0.5);
    try std.testing.expect(idx < 4);
}

test "tokenAccuracy" {
    const gen = [_]u32{ 1, 2, 3 };
    const ref = [_]u32{ 1, 9, 3, 4 };
    const acc = tokenAccuracy(&gen, &ref);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), acc, 1e-6);
}
