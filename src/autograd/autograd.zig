// Autograd reverse-mode (PRD 3.2). Tape + backward helpers.

const std = @import("std");

pub const OpTag = enum { add, mul, scale, relu };

pub const Node = struct {
    tag: OpTag,
    n_inputs: u8,
    n_outputs: u8,
};

pub const Tape = struct {
    nodes: std.ArrayList(Node),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Tape {
        return .{ .nodes = std.ArrayList(Node).init(allocator), .allocator = allocator };
    }

    pub fn deinit(self: *Tape) void {
        self.nodes.deinit();
    }

    pub fn record(self: *Tape, node: Node) !void {
        try self.nodes.append(node);
    }

    pub fn len(self: *const Tape) usize {
        return self.nodes.items.len;
    }

    pub fn clear(self: *Tape) void {
        self.nodes.clearRetainingCapacity();
    }
};

// Backward helpers (slice-based, contiguous).

// dA += dOut (for z = a + b), dB += dOut.
pub fn backwardAdd(dA: []f32, dB: []f32, dOut: []const f32) void {
    std.debug.assert(dA.len == dOut.len and dB.len == dOut.len);
    for (dA, dB, dOut) |*a, *b, g| {
        a.* += g;
        b.* += g;
    }
}

// For z = a * b: dA += dOut * b, dB += dOut * a.
pub fn backwardMul(dA: []f32, dB: []f32, a: []const f32, b: []const f32, dOut: []const f32) void {
    std.debug.assert(dA.len == dOut.len and dB.len == dOut.len);
    std.debug.assert(a.len == dOut.len and b.len == dOut.len);
    for (dA, dB, a, b, dOut) |*da, *db, av, bv, g| {
        da.* += g * bv;
        db.* += g * av;
    }
}

// For z = s * a: dA += dOut * s.
pub fn backwardScale(dA: []f32, dOut: []const f32, s: f32) void {
    std.debug.assert(dA.len == dOut.len);
    for (dA, dOut) |*a, g| a.* += g * s;
}

// For z = relu(a): dA += dOut if a > 0.
pub fn backwardRelu(dA: []f32, a: []const f32, dOut: []const f32) void {
    std.debug.assert(dA.len == dOut.len and a.len == dOut.len);
    for (dA, a, dOut) |*da, av, g| {
        if (av > 0) da.* += g;
    }
}

// Gradient check: finite difference for f(x)=sum? Here check scalar f.
pub fn finiteDiff(f: *const fn (x: f64) f64, x: f64, eps: f64) f64 {
    return (f(x + eps) - f(x - eps)) / (2.0 * eps);
}

test "Tape record/clear" {
    const allocator = std.testing.allocator;
    var t = Tape.init(allocator);
    defer t.deinit();
    try t.record(.{ .tag = .add, .n_inputs = 2, .n_outputs = 1 });
    try std.testing.expectEqual(@as(usize, 1), t.len());
    t.clear();
    try std.testing.expectEqual(@as(usize, 0), t.len());
}

test "backwardAdd" {
    var dA = [_]f32{ 0, 0 };
    var dB = [_]f32{ 0, 0 };
    const g = [_]f32{ 1, 2 };
    backwardAdd(&dA, &dB, &g);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2 }, &dA);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2 }, &dB);
}

test "backwardMul" {
    var dA = [_]f32{ 0, 0 };
    var dB = [_]f32{ 0, 0 };
    const a = [_]f32{ 2, 3 };
    const b = [_]f32{ 4, 5 };
    const g = [_]f32{ 1, 1 };
    backwardMul(&dA, &dB, &a, &b, &g);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 4, 5 }, &dA);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 2, 3 }, &dB);
}

test "backwardRelu" {
    var dA = [_]f32{ 0, 0, 0 };
    const a = [_]f32{ -1, 2, 0 };
    const g = [_]f32{ 1, 1, 1 };
    backwardRelu(&dA, &a, &g);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 0, 1, 0 }, &dA);
}
EOF
echo done