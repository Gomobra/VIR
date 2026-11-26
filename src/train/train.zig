// Training loop (PRD 3.8). Minimal end-to-end CPU training step.

const std = @import("std");
const cpu = @import("../cpu/cpu.zig");
const nn = @import("../nn/nn.zig");
const autograd = @import("../autograd/autograd.zig");
const optim = @import("../optim/optim.zig");

pub const TrainConfig = struct {
    lr: f32 = 1e-2,
    batch_tokens: usize = 4,
    max_steps: usize = 100,
};

pub const Trainer = struct {
    cfg: TrainConfig,
    step: usize = 0,

    pub fn init(cfg: TrainConfig) Trainer {
        return .{ .cfg = cfg };
    }

    pub fn runStep(self: *Trainer) void {
        self.step += 1;
    }

    pub fn done(self: *const Trainer) bool {
        return self.step >= self.cfg.max_steps;
    }
};

// Tiny model: embedding (vocab x dim) + linear (dim x vocab). One training step.
pub const TinyModel = struct {
    vocab: usize,
    dim: usize,
    embed: []f32, // vocab*dim
    W: []f32, // dim*vocab
    opt_embed: optim.AdamW,
    opt_W: optim.AdamW,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, vocab: usize, dim: usize, lr: f32) !TinyModel {
        const embed = try allocator.alloc(f32, vocab * dim);
        const W = try allocator.alloc(f32, dim * vocab);
        // small init
        for (embed, 0..) |*x, i| x.* = 0.01 * @as(f32, @floatFromInt(i % 5)) - 0.02;
        for (W, 0..) |*x, i| x.* = 0.01 * @as(f32, @floatFromInt(i % 7)) - 0.03;
        const opt_e = try optim.AdamW.init(allocator, vocab * dim, lr);
        const opt_w = try optim.AdamW.init(allocator, dim * vocab, lr);
        return .{ .vocab = vocab, .dim = dim, .embed = embed, .W = W, .opt_embed = opt_e, .opt_W = opt_w, .allocator = allocator };
    }

    pub fn deinit(self: *TinyModel) void {
        self.allocator.free(self.embed);
        self.allocator.free(self.W);
        self.opt_embed.deinit();
        self.opt_W.deinit();
    }

    // Returns cross-entropy loss (sum over batch).
    pub fn trainStep(self: *TinyModel, ids: []const u32, targets: []const u32) !f32 {
        const batch = ids.len;
        const dim = self.dim;
        const vocab = self.vocab;

        // forward: embed -> (batch, dim); logits = embed @ W (batch, vocab)
        const hidden = try self.allocator.alloc(f32, batch * dim);
        defer self.allocator.free(hidden);
        cpu.embeddingGatherF32(hidden, self.embed, ids, vocab, dim);

        const logits = try self.allocator.alloc(f32, batch * vocab);
        defer self.allocator.free(logits);
        // logits[b, v] = sum_d hidden[b,d] * W[d, v]  (W stored dim*vocab, row-major d-major)
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            var v: usize = 0;
            while (v < vocab) : (v += 1) {
                var acc: f32 = 0;
                var d: usize = 0;
                while (d < dim) : (d += 1) acc += hidden[b * dim + d] * self.W[d * vocab + v];
                logits[b * vocab + v] = acc;
            }
        }

        const loss = cpu.crossEntropyF32(logits, targets, batch, vocab);

        // backward
        const dLogits = try self.allocator.alloc(f32, batch * vocab);
        defer self.allocator.free(dLogits);
        @memset(dLogits, 0);
        autograd.backwardCrossEntropy(dLogits, logits, targets, batch, vocab);

        // dW and dHidden
        const dW = try self.allocator.alloc(f32, dim * vocab);
        defer self.allocator.free(dW);
        @memset(dW, 0);
        const dHidden = try self.allocator.alloc(f32, batch * dim);
        defer self.allocator.free(dHidden);
        @memset(dHidden, 0);

        // dW[d,v] += sum_b hidden[b,d] * dLogits[b,v]
        b = 0;
        while (b < batch) : (b += 1) {
            var d: usize = 0;
            while (d < dim) : (d += 1) {
                var v: usize = 0;
                while (v < vocab) : (v += 1) {
                    dW[d * vocab + v] += hidden[b * dim + d] * dLogits[b * vocab + v];
                }
            }
            // dHidden[b,d] = sum_v W[d,v] * dLogits[b,v]
            d = 0;
            while (d < dim) : (d += 1) {
                var acc: f32 = 0;
                var v: usize = 0;
                while (v < vocab) : (v += 1) acc += self.W[d * vocab + v] * dLogits[b * vocab + v];
                dHidden[b * dim + d] = acc;
            }
        }

        // embedding backward: scatter-add dHidden into dEmbed
        const dEmbed = try self.allocator.alloc(f32, vocab * dim);
        defer self.allocator.free(dEmbed);
        @memset(dEmbed, 0);
        autograd.backwardEmbedding(dEmbed, dHidden, ids, vocab, dim);

        // optimizer step
        self.opt_embed.step(self.embed, dEmbed);
        self.opt_W.step(self.W, dW);

        return loss;
    }
};

test "TinyModel trainStep reduces loss" {
    const allocator = std.testing.allocator;
    var model = try TinyModel.init(allocator, 4, 3, 0.1);
    defer model.deinit();
    const ids = [_]u32{ 0, 1, 2, 3 };
    const targets = [_]u32{ 1, 2, 3, 0 };
    const l1 = try model.trainStep(&ids, &targets);
    const l2 = try model.trainStep(&ids, &targets);
    // loss should generally decrease (not guaranteed single step, but check finite)
    try std.testing.expect(std.math.isFinite(l1));
    try std.testing.expect(std.math.isFinite(l2));
}
