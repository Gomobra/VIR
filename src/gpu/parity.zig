// GPU parity harness (PRD 3.3, 3.21): compare CPU reference vs GPU outputs.

const std = @import("std");

pub const ParityReport = struct {
    max_abs: f32,
    max_rel: f32,
    ok: bool,
};

pub fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var m: f32 = 0;
    for (a, b) |x, y| {
        const d = @abs(x - y);
        if (d > m) m = d;
    }
    return m;
}

pub fn maxRelDiff(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var m: f32 = 0;
    for (a, b) |x, y| {
        const denom = if (@abs(x) > 1e-12) @abs(x) else 1.0;
        const d = @abs(x - y) / denom;
        if (d > m) m = d;
    }
    return m;
}

pub fn compare(a: []const f32, b: []const f32, rel_tol: f32, abs_tol: f32) ParityReport {
    const mad = maxAbsDiff(a, b);
    const mrd = maxRelDiff(a, b);
    const ok = mad <= abs_tol and mrd <= rel_tol;
    return .{ .max_abs = mad, .max_rel = mrd, .ok = ok };
}

pub fn ptxHasEntry(ptx: []const u8, name: []const u8) bool {
    // Looks for ".entry <name>" in PTX text.
    var i: usize = 0;
    while (i + name.len + 7 <= ptx.len) : (i += 1) {
        if (std.mem.startsWith(u8, ptx[i..], ".entry")) {
            // skip whitespace
            var j = i + 6;
            while (j < ptx.len and (ptx[j] == ' ' or ptx[j] == '\t')) : (j += 1) {}
            if (j + name.len <= ptx.len and std.mem.eql(u8, ptx[j .. j + name.len], name)) {
                return true;
            }
        }
    }
    return false;
}

test "parity compare" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 1.001, 2.0, 2.999 };
    const r = compare(&a, &b, 1e-2, 1e-2);
    try std.testing.expect(r.ok);
}

test "ptxHasEntry" {
    const ptx = ".entry vecadd(\n";
    try std.testing.expect(ptxHasEntry(ptx, "vecadd"));
    try std.testing.expect(!ptxHasEntry(ptx, "matmul"));
}
