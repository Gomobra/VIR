// Autograd reverse-mode (PRD 3.2). Tape sederhana.

const std = @import("std");

pub const OpTag = enum { add, mul, scale, relu };

pub const Node = struct {
    tag: OpTag,
    // indeks input/output dalam arena (disimpan sebagai usize placeholder)
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

test "Tape record/clear" {
    const allocator = std.testing.allocator;
    var t = Tape.init(allocator);
    defer t.deinit();
    try t.record(.{ .tag = .add, .n_inputs = 2, .n_outputs = 1 });
    try std.testing.expectEqual(@as(usize, 1), t.len());
    t.clear();
    try std.testing.expectEqual(@as(usize, 0), t.len());
}
