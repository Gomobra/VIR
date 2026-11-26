// Evaluation (PRD 3.11). BPB minimal.

const std = @import("std");

pub fn bitsPerByte(total_neg_log2_prob: f64, n_bytes: usize) f64 {
    if (n_bytes == 0) return 0;
    return total_neg_log2_prob / @as(f64, @floatFromInt(n_bytes));
}

test "bitsPerByte" {
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), bitsPerByte(8.0, 8), 1e-9);
    try std.testing.expectEqual(@as(f64, 0.0), bitsPerByte(0.0, 0));
}

// Multiple-choice accuracy: predictions vs labels.
pub fn mcAccuracy(predictions: []const usize, labels: []const usize) f32 {
    if (predictions.len == 0 or predictions.len != labels.len) return 0;
    var correct: usize = 0;
    for (predictions, labels) |p, l| if (p == l) {
        correct += 1;
    };
    return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(predictions.len));
}

// Needle-in-a-haystack generator: build a haystack of `haystack_len` filler tokens,
// insert needle at needle_pos. Returns allocated string.
pub fn needleHaystack(allocator: std.mem.Allocator, haystack_len: usize, needle: []const u8, needle_pos: usize) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    defer out.deinit();
    const pos = @min(needle_pos, haystack_len);
    // filler before needle
    var i: usize = 0;
    while (i < pos) : (i += 1) try out.appendSlice("filler ");
    try out.appendSlice(needle);
    try out.appendSlice(" ");
    i = pos + needle.len;
    while (i < haystack_len) : (i += 1) try out.appendSlice("filler ");
    return out.toOwnedSlice();
}

// ROUGE-L (simplified LCS-based) score between hypothesis and reference.
pub fn rougeL(hyp: []const u8, ref: []const u8) f32 {
    // LCS length over bytes (simplified)
    if (hyp.len == 0 or ref.len == 0) return 0;
    // dynamic programming LCS
    const allocator = std.heap.page_allocator;
    const dp = allocator.alloc(usize, (hyp.len + 1) * (ref.len + 1)) catch return 0;
    defer allocator.free(dp);
    var i: usize = 0;
    while (i <= hyp.len) : (i += 1) dp[i * (ref.len + 1) + 0] = 0;
    var j: usize = 0;
    while (j <= ref.len) : (j += 1) dp[0 * (ref.len + 1) + j] = 0;
    i = 1;
    while (i <= hyp.len) : (i += 1) {
        j = 1;
        while (j <= ref.len) : (j += 1) {
            if (hyp[i - 1] == ref[j - 1]) {
                dp[i * (ref.len + 1) + j] = dp[(i - 1) * (ref.len + 1) + (j - 1)] + 1;
            } else {
                const a = dp[(i - 1) * (ref.len + 1) + j];
                const b = dp[i * (ref.len + 1) + (j - 1)];
                dp[i * (ref.len + 1) + j] = if (a > b) a else b;
            }
        }
    }
    const lcs = dp[hyp.len * (ref.len + 1) + ref.len];
    const prec = @as(f32, @floatFromInt(lcs)) / @as(f32, @floatFromInt(hyp.len));
    const rec = @as(f32, @floatFromInt(lcs)) / @as(f32, @floatFromInt(ref.len));
    if (prec + rec == 0) return 0;
    return 2 * prec * rec / (prec + rec);
}

test "mcAccuracy" {
    const preds = [_]usize{ 0, 1, 2, 0 };
    const labels = [_]usize{ 0, 1, 1, 0 };
    const acc = mcAccuracy(&preds, &labels);
    try std.testing.expect(@abs(acc - 0.75) < 1e-6);
}

test "needleHaystack contains needle" {
    const allocator = std.testing.allocator;
    const out = try needleHaystack(allocator, 10, "NEEDLE", 3);
    defer allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "NEEDLE") != null);
}

test "rougeL identical" {
    const s = "hello world";
    const r = rougeL(s, s);
    try std.testing.expect(@abs(r - 1.0) < 1e-5);
}

// Perplexity from average negative log-likelihood.
pub fn perplexity(avg_nll: f64) f64 {
    return @exp(avg_nll);
}

// F1 score (token-level) between prediction and reference sets.
pub fn f1Score(pred: []const []const u8, ref: []const []const u8) f32 {
    if (pred.len == 0 and ref.len == 0) return 1.0;
    if (pred.len == 0 or ref.len == 0) return 0.0;
    // count intersection
    var inter: usize = 0;
    for (pred) |p| for (ref) |r| if (std.mem.eql(u8, p, r)) {
        inter += 1;
    };
    const prec = @as(f32, @floatFromInt(inter)) / @as(f32, @floatFromInt(pred.len));
    const rec = @as(f32, @floatFromInt(inter)) / @as(f32, @floatFromInt(ref.len));
    if (prec + rec == 0) return 0;
    return 2 * prec * rec / (prec + rec);
}

// Exact match (string equality).
pub fn exactMatch(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

test "perplexity" {
    try std.testing.expectApproxEqAbs(@as(f64, 2.71828), perplexity(1.0), 1e-4);
}

test "f1Score identical" {
    const a = [_][]const u8{ "a", "b" };
    const b = [_][]const u8{ "a", "b" };
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), f1Score(&a, &b), 1e-6);
}

test "exactMatch" {
    try std.testing.expect(exactMatch("abc", "abc"));
    try std.testing.expect(!exactMatch("abc", "def"));
}

// pass@k unbiased estimator (PRD 4.1): 1 - C(n-c, k) / C(n, k).
pub fn passAtK(n: usize, c: usize, k: usize) f64 {
    if (n == 0) return 0;
    if (k > n) return 0;
    if (c >= n) return 1.0;
    // compute C(n-c, k) / C(n, k) via product
    // = prod_{i=0}^{k-1} (n - c - i) / (n - i)
    var ratio: f64 = 1.0;
    var i: usize = 0;
    while (i < k) : (i += 1) {
        const num = @as(f64, @floatFromInt(n - c - i));
        const den = @as(f64, @floatFromInt(n - i));
        if (den == 0) return 0;
        ratio *= num / den;
    }
    return 1.0 - ratio;
}

test "passAtK" {
    // n=10, c=1, k=1 -> 0.1
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), passAtK(10, 1, 1), 1e-9);
    // n=5, c=5, k=1 -> 1.0
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), passAtK(5, 5, 1), 1e-9);
    // n=10, c=0, k=1 -> 0
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), passAtK(10, 0, 1), 1e-9);
}
