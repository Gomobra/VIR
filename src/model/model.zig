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

        pub fn backward(self: *const Block, dX: []f32, dW_attn: []f32, dW_mlp: []f32, x: []const f32, dOut: []const f32, batch: usize, d: usize) void {
            std.debug.assert(dX.len == batch * d and dOut.len == batch * d);
            std.debug.assert(dW_attn.len == d * d);
            std.debug.assert(dW_mlp.len == (d * 2) * d);
            const autograd = @import("../autograd/autograd.zig");
            const cpu = @import("../cpu/cpu.zig");
            // recompute intermediates
            const norm1 = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(norm1);
            self.norm1.forward(norm1, x);
            const attn_out = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(attn_out);
            self.attn.forward(attn_out, norm1, batch);
            // residual1 = x + attn_out
            const r1 = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(r1);
            for (r1, x, attn_out) |*o, xv, av| o.* = xv + av;
            const norm2 = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(norm2);
            self.norm2.forward(norm2, r1);
            const mlp_out = self.allocator.alloc(f32, batch * (d * 2)) catch return;
            defer self.allocator.free(mlp_out);
            self.mlp.forward(mlp_out, norm2, batch);

            // dOut -> dR1 (residual) + dMlp
            // dMlpOutput (first d) = dOut; rest 0
            const dMlp = self.allocator.alloc(f32, batch * (d * 2)) catch return;
            defer self.allocator.free(dMlp);
            @memset(dMlp, 0);
            for (0..batch) |b| for (0..d) |i| dMlp[b * (d * 2) + i] = dOut[b * d + i];
            // dNorm2 = dMlp @ W_mlp^T (first d rows) -> [batch, d]
            // dW_mlp += norm2^T @ dMlp (full 2d x d)
            const dNorm2 = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(dNorm2);
            @memset(dNorm2, 0);
            for (0..batch) |b| {
                for (0..d) |i| {
                    var acc: f32 = 0;
                    for (0..(d * 2)) |j| acc += dMlp[b * (d * 2) + j] * self.mlp.w[j * d + i];
                    dNorm2[b * d + i] = acc;
                }
                // dW_mlp
                for (0..(d * 2)) |j| for (0..d) |i| {
                    dW_mlp[j * d + i] += norm2[b * d + i] * dMlp[b * (d * 2) + j];
                };
            }
            // backward through norm2: dR1_mlp = backwardRMSNorm(r1, dNorm2)
            const dR1_mlp = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(dR1_mlp);
            @memset(dR1_mlp, 0);
            for (0..batch) |b| {
                const slice_x = r1[b * d .. (b + 1) * d];
                const slice_do = dNorm2[b * d .. (b + 1) * d];
                const slice_dx = dR1_mlp[b * d .. (b + 1) * d];
                autograd.backwardRMSNorm(slice_dx, slice_x, slice_do, 1e-6);
            }
            // dR1 = dOut (residual) + dR1_mlp
            const dR1 = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(dR1);
            for (dR1, dOut, dR1_mlp) |*o, a, bb| o.* = a + bb;
            // dAttnOut = dR1 (from attn residual)
            // dNorm1 = dR1 @ W_attn^T
            const dNorm1 = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(dNorm1);
            @memset(dNorm1, 0);
            for (0..batch) |b| {
                for (0..d) |i| {
                    var acc: f32 = 0;
                    for (0..d) |j| acc += dR1[b * d + j] * self.attn.w[j * d + i];
                    dNorm1[b * d + i] = acc;
                }
                for (0..d) |j| for (0..d) |i| {
                    dW_attn[j * d + i] += norm1[b * d + i] * dR1[b * d + j];
                };
            }
            // backward through norm1: dX_attn = backwardRMSNorm(x, dNorm1)
            const dX_attn = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(dX_attn);
            @memset(dX_attn, 0);
            for (0..batch) |b| {
                const slice_x = x[b * d .. (b + 1) * d];
                const slice_do = dNorm1[b * d .. (b + 1) * d];
                const slice_dx = dX_attn[b * d .. (b + 1) * d];
                autograd.backwardRMSNorm(slice_dx, slice_x, slice_do, 1e-6);
            }
            // dX = dR1 (residual from first) + dX_attn
            for (dX, dR1, dX_attn) |*o, a, bb| o.* = a + bb;
            _ = cpu;
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

test "Block backward gradcheck" {
    const allocator = std.testing.allocator;
    var blk = try Block.init(allocator, 4);
    defer blk.deinit();
    var x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var out = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    blk.forward(&out, &x, 2, 4);
    // dOut = ones
    var dOut = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    var dX = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var dW_attn = try allocator.alloc(f32, 4 * 4);
    defer allocator.free(dW_attn);
    @memset(dW_attn, 0);
    var dW_mlp = try allocator.alloc(f32, 8 * 4);
    defer allocator.free(dW_mlp);
    @memset(dW_mlp, 0);
    blk.backward(&dX, dW_attn, dW_mlp, &x, &dOut, 2, 4);
    // finite diff for x[0]
    const eps: f32 = 1e-3;
    var xp = x;
    xp[0] += eps;
    var xm = x;
    xm[0] -= eps;
    var op = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var om = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    blk.forward(&op, &xp, 2, 4);
    blk.forward(&om, &xm, 2, 4);
    var sp: f32 = 0;
    var sm: f32 = 0;
    for (op) |v| sp += v;
    for (om) |v| sm += v;
    const fd = (sp - sm) / (2 * eps);
    try std.testing.expect(@abs(fd - dX[0]) < 1e-1);
}
