// Tokenizer BPE baseline (PRD 3.5). Minimal byte-level BPE.

const std = @import("std");

pub const Vocab = struct {
    // id -> bytes
    tokens: std.ArrayList([]u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Vocab {
        return .{ .tokens = std.ArrayList([]u8).init(allocator), .allocator = allocator };
    }

    pub fn deinit(self: *Vocab) void {
        for (self.tokens.items) |t| self.allocator.free(t);
        self.tokens.deinit();
    }

    pub fn addToken(self: *Vocab, bytes: []const u8) !u32 {
        const copy = try self.allocator.dupe(u8, bytes);
        try self.tokens.append(copy);
        return @intCast(self.tokens.items.len - 1);
    }

    pub fn size(self: *const Vocab) usize {
        return self.tokens.items.len;
    }
};

// Byte-level: each byte becomes a token id (0..255) initially; here we just
// expose a helper to encode a string as bytes.
pub fn encodeBytes(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    const out = try allocator.dupe(u8, s);
    return out;
}

test "Vocab addToken" {
    const allocator = std.testing.allocator;
    var v = Vocab.init(allocator);
    defer v.deinit();
    const id0 = try v.addToken("a");
    const id1 = try v.addToken("b");
    try std.testing.expectEqual(@as(u32, 0), id0);
    try std.testing.expectEqual(@as(u32, 1), id1);
    try std.testing.expectEqual(@as(usize, 2), v.size());
}

test "encodeBytes" {
    const allocator = std.testing.allocator;
    const out = try encodeBytes(allocator, "hi");
    defer allocator.free(out);
    try std.testing.expectEqualSlices(u8, "hi", out);
}
