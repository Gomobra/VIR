// Inference (PRD 3.12). Forward-only sampling helpers.

const std = @import("std");

pub fn argmax(logits: []const f32) usize {
    if (logits.len == 0) return 0;
    var best: usize = 0;
    var bestv: f32 = logits[0];
    for (logits, 0..) |v, i| {
        if (v > bestv) {
            bestv = v;
            best = i;
        }
    }
    return best;
}

pub fn temperatureScale(logits: []f32, t: f32) void {
    if (t <= 0) return;
    for (logits) |*v| v.* /= t;
}

test "argmax" {
    const l = [_]f32{ 0.1, 0.5, 0.2 };
    try std.testing.expectEqual(@as(usize, 1), argmax(&l));
}

test "temperatureScale" {
    var l = [_]f32{ 2.0, 4.0 };
    temperatureScale(&l, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), l[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), l[1], 1e-6);
}
