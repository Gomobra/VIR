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
