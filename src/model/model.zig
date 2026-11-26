// Model (PRD 3.7). Tiny transformer block skeleton.

const std = @import("std");
const nn = @import("../nn/nn.zig");

pub const Block = struct {
    attn: nn.Linear,
    mlp: nn.Linear,
    norm1: nn.RMSNorm,
    norm2: nn.RMSNorm,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, d: usize) !Block {
        return .{
            .attn = try nn.Linear.init(allocator, d, d),
            .mlp = try nn.Linear.init(allocator, d, d * 2),
            .norm1 = .{},
            .norm2 = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Block) void {
        self.attn.deinit();
        self.mlp.deinit();
    }

    // forward: x (batch, d) -> x + attn(norm(x)) + mlp(norm(x))
    pub fn forward(self: *const Block, out: []f32, x: []const f32, batch: usize, d: usize) void {
        std.debug.assert(x.len == batch * d and out.len == batch * d);
        const tmp = self.allocator.alloc(f32, batch * d) catch return;
        defer self.allocator.free(tmp);
        self.norm1.forward(tmp, x);
        self.attn.forward(out, tmp, batch);
        // residual
        for (out, x) |*o, v| o.* += v;
        self.norm2.forward(tmp, out);
        // mlp (simplified: linear only)
        const tmp2 = self.allocator.alloc(f32, batch * (d * 2)) catch return;
        defer self.allocator.free(tmp2);
        self.mlp.forward(tmp2, tmp, batch);
        // add first d back
        for (0..batch) |b| {
            for (0..d) |i| {
                out[b * d + i] += tmp2[b * (d * 2) + i];
            }
        }
    }
};

test "Block forward finite" {
    const allocator = std.testing.allocator;
    var blk = try Block.init(allocator, 4);
    defer blk.deinit();
    var x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var out = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    blk.forward(&out, &x, 2, 4);
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}
