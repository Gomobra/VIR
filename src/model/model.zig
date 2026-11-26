// Model (PRD 3.7). Full model: embedding + N blocks + final norm + output head.

const std = @import("std");
const nn = @import("../nn/nn.zig");
const config = @import("../config/config.zig");

pub const Model = struct {
    cfg: config.ModelConfig,
    embed: []f32, // vocab * d
    blocks: []Block,
    final_norm: nn.RMSNorm,
    head: nn.Linear, // d -> vocab (separate; could tie)
    allocator: std.mem.Allocator,

    const Block = struct {
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

        pub fn forward(self: *const Block, out: []f32, x: []const f32, batch: usize, d: usize) void {
            std.debug.assert(x.len == batch * d and out.len == batch * d);
            const tmp = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(tmp);
            self.norm1.forward(tmp, x);
            self.attn.forward(out, tmp, batch);
            for (out, x) |*o, v| o.* += v;
            const tmp2 = self.allocator.alloc(f32, batch * (d * 2)) catch return;
            defer self.allocator.free(tmp2);
            self.norm2.forward(tmp, out);
            self.mlp.forward(tmp2, tmp, batch);
            for (0..batch) |b| {
                for (0..d) |i| {
                    out[b * d + i] += tmp2[b * (d * 2) + i];
                }
            }
        }
    };

    pub fn init(allocator: std.mem.Allocator, cfg: config.ModelConfig) !Model {
        const d = cfg.d_model;
        const v = cfg.vocab_size;
        const n_blocks = cfg.n_groups * 4; // 3 GDN + 1 attn per group (simplified: all Block)
        const embed = try allocator.alloc(f32, v * d);
        for (embed, 0..) |*x, i| x.* = 0.02 * @as(f32, @floatFromInt(i % 13)) - 0.12;
        const blocks = try allocator.alloc(Block, n_blocks);
        for (blocks) |*bl| bl.* = try Block.init(allocator, d);
        const head = try nn.Linear.init(allocator, d, v);
        return .{
            .cfg = cfg,
            .embed = embed,
            .blocks = blocks,
            .final_norm = .{},
            .head = head,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Model) void {
        self.allocator.free(self.embed);
        for (self.blocks) |*bl| bl.deinit();
        self.allocator.free(self.blocks);
        self.head.deinit();
    }

    // forward: token ids [seq] -> logits [seq, vocab]
    pub fn forward(self: *const Model, logits: []f32, tokens: []const u32) void {
        const d = self.cfg.d_model;
        const v = self.cfg.vocab_size;
        const seq = tokens.len;
        std.debug.assert(logits.len == seq * v);

        // embedding gather
        const hidden = self.allocator.alloc(f32, seq * d) catch return;
        defer self.allocator.free(hidden);
        cpuEmbeddingGather(hidden, self.embed, tokens, v, d);

        // blocks
        var cur = hidden;
        var next = self.allocator.alloc(f32, seq * d) catch return;
        defer self.allocator.free(next);
        for (self.blocks) |*bl| {
            bl.forward(next, cur, seq, d);
            // swap
            const tmp = cur;
            cur = next;
            next = tmp;
        }
        // final norm
        var normed = self.allocator.alloc(f32, seq * d) catch return;
        defer self.allocator.free(normed);
        var i: usize = 0;
        while (i < seq) : (i += 1) {
            self.final_norm.forward(normed[i * d .. (i + 1) * d], cur[i * d .. (i + 1) * d]);
        }

        // head: logits = normed @ W^T  (d -> vocab)
        self.head.forward(logits, normed, seq);
    }

    pub fn nBlocks(self: *const Model) usize {
        return self.blocks.len;
    }
};

fn cpuEmbeddingGather(out: []f32, table: []const f32, ids: []const u32, vocab: usize, dim: usize) void {
    const cpu = @import("../cpu/cpu.zig");
    cpu.embeddingGatherF32(out, table, ids, vocab, dim);
}

test "Model forward finite" {
    const allocator = std.testing.allocator;
    const cfg = config.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 8, .seq_len = 4 };
    var m = try Model.init(allocator, cfg);
    defer m.deinit();
    const tokens = [_]u32{ 1, 2, 3, 4 };
    var logits = [_]f32{0} * *(4 * 8);
    m.forward(&logits, &tokens);
    for (logits) |v| try std.testing.expect(std.math.isFinite(v));
    try std.testing.expectEqual(@as(usize, 4), m.nBlocks());
}
