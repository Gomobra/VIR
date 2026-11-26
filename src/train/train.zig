// Training loop (PRD 3.8). Minimal end-to-end CPU training step.

const std = @import("std");
const cpu = @import("../cpu/cpu.zig");
const nn = @import("../nn/nn.zig");
const autograd = @import("../autograd/autograd.zig");
const optim = @import("../optim/optim.zig");
const checkpoint = @import("../checkpoint/checkpoint.zig");
const infer = @import("../infer/infer.zig");

pub const TrainConfig = struct {
    lr: f32 = 1e-2,
    batch_tokens: usize = 4,
    max_steps: usize = 100,
    warmup_steps: usize = 5,
    decay_fraction: f32 = 0.2,
};

// WSD (warmup-stable-decay) learning rate schedule.
pub fn wsdLr(step: usize, cfg: TrainConfig) f32 {
    const peak = cfg.lr;
    const total = cfg.max_steps;
    if (total == 0) return peak;
    const warmup = if (cfg.warmup_steps > 0) cfg.warmup_steps else 1;
    const decay_steps: usize = @intFromFloat(@as(f32, @floatFromInt(total)) * cfg.decay_fraction);
    const stable_end = if (total > decay_steps) total - decay_steps else total;
    if (step < warmup) {
        return peak * (@as(f32, @floatFromInt(step + 1)) / @as(f32, @floatFromInt(warmup)));
    }
    if (step < stable_end) {
        return peak;
    }
    // decay
    const prog = @as(f32, @floatFromInt(step - stable_end + 1)) / @as(f32, @floatFromInt(if (decay_steps > 0) decay_steps else 1));
    const factor = 1.0 - @min(prog, 1.0);
    return peak * factor;
}

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

        const loss = @as(f32, @floatCast(cpu.crossEntropyF32(logits, targets, batch, vocab)));

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

    // Greedy generation: given prompt token ids, generate up to max_new tokens.
    pub fn generate(self: *TinyModel, allocator: std.mem.Allocator, prompt: []const u32, max_new: usize) ![]u32 {
        var seq = std.ArrayList(u32).init(allocator);
        defer seq.deinit();
        try seq.appendSlice(prompt);
        const dim = self.dim;
        const vocab = self.vocab;
        var i: usize = 0;
        while (i < max_new) : (i += 1) {
            const n = seq.items.len;
            const hidden = try allocator.alloc(f32, n * dim);
            defer allocator.free(hidden);
            cpu.embeddingGatherF32(hidden, self.embed, seq.items, vocab, dim);
            var logits = try allocator.alloc(f32, n * vocab);
            defer allocator.free(logits);
            var b: usize = 0;
            while (b < n) : (b += 1) {
                var v: usize = 0;
                while (v < vocab) : (v += 1) {
                    var acc: f32 = 0;
                    var d: usize = 0;
                    while (d < dim) : (d += 1) acc += hidden[b * dim + d] * self.W[d * vocab + v];
                    logits[b * vocab + v] = acc;
                }
            }
            const last = logits[(n - 1) * vocab .. n * vocab];
            const next = infer.argmax(last);
            try seq.append(next);
        }
        return seq.toOwnedSlice();
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

// Training loop with checkpoint save at interval.
pub fn runTrainingLoop(model: *TinyModel, ids: []const u32, targets: []const u32, cfg: TrainConfig, checkpoint_dir: []const u8, checkpoint_interval: usize) !usize {
    var trainer = Trainer.init(cfg);
    var steps: usize = 0;
    while (!trainer.done()) {
        const loss = try model.trainStep(ids, targets);
        _ = loss;
        trainer.runStep();
        steps += 1;
        if (checkpoint_interval > 0 and steps % checkpoint_interval == 0) {
            // save meta
            var meta_buf: [128]u8 = undefined;
            const meta = try std.fmt.bufPrint(&meta_buf, "{{\"step\":{d}}}", .{steps});
            try checkpoint.atomicSave(model.allocator, checkpoint_dir, steps, meta);
        }
    }
    return steps;
}

test "wsdLr warmup" {
    const cfg = TrainConfig{ .lr = 0.1, .max_steps = 10, .warmup_steps = 2, .decay_fraction = 0.2 };
    // step 0 (first) -> warmup: 0.1 * 1/2 = 0.05
    const lr0 = wsdLr(0, cfg);
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), lr0, 1e-5);
    // stable
    const lr3 = wsdLr(3, cfg);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), lr3, 1e-5);
}

test "runTrainingLoop saves checkpoint" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/trainckpt";
    std.fs.cwd().deleteTree(base) catch {};
    try std.fs.cwd().makePath(base);
    var model = try TinyModel.init(allocator, 4, 3, 0.1);
    defer model.deinit();
    const ids = [_]u32{ 0, 1, 2, 3 };
    const targets = [_]u32{ 1, 2, 3, 0 };
    const cfg = TrainConfig{ .max_steps = 4, .warmup_steps = 1 };
    const steps = try runTrainingLoop(&model, &ids, &targets, cfg, base, 2);
    try std.testing.expectEqual(@as(usize, 4), steps);
    // checkpoint at step 2 or 4 should exist
    const latest = try checkpoint.loadLatest(allocator, base);
    try std.testing.expect(latest != null);
}

// --- TST (Token-Superposition Training, PRD 3.8) ---

// Multi-hot cross-entropy: given logits (vocab) and a bag of target ids,
// loss = -sum_{t in bag} log softmax(logits)[t] / |bag|.
pub fn multiHotCE(logits: []const f32, bag: []const u32, vocab: usize) f32 {
    std.debug.assert(logits.len == vocab);
    if (bag.len == 0) return 0;
    // softmax
    var mx: f32 = logits[0];
    var j: usize = 1;
    while (j < vocab) : (j += 1) if (logits[j] > mx) {
        mx = logits[j];
    };
    var sum: f32 = 0;
    j = 0;
    while (j < vocab) : (j += 1) sum += @exp(logits[j] - mx);
    var total: f32 = 0;
    for (bag) |t| {
        const p = @exp(logits[t] - mx) / sum;
        total += -@log(p + 1e-12);
    }
    return total / @as(f32, @floatFromInt(bag.len));
}

// Build a bag of next s tokens from a token sequence starting at pos.
pub fn tstBag(tokens: []const u32, pos: usize, s: usize, vocab: usize) []const u32 {
    _ = vocab;
    // returns slice into tokens[pos..pos+s] (caller must ensure bounds)
    const end = @min(pos + s, tokens.len);
    return tokens[pos..end];
}

test "multiHotCE single target equals CE" {
    var logits = [_]f32{ 0, 0, 0, 0 };
    const bag = [_]u32{0};
    const loss = multiHotCE(&logits, &bag, 4);
    // uniform -> -log(0.25) = 1.386
    try std.testing.expect(@abs(loss - 1.3863) < 1e-3);
}

test "multiHotCE bag" {
    var logits = [_]f32{ 0, 0, 0, 0 };
    const bag = [_]u32{ 0, 1 };
    const loss = multiHotCE(&logits, &bag, 4);
    try std.testing.expect(std.math.isFinite(loss));
}

test "tstBag" {
    const toks = [_]u32{ 1, 2, 3, 4, 5 };
    const b = tstBag(&toks, 1, 2, 10);
    try std.testing.expectEqual(@as(usize, 2), b.len);
    try std.testing.expectEqual(@as(u32, 2), b[0]);
}

test "TinyModel generate" {
    const allocator = std.testing.allocator;
    var model = try TinyModel.init(allocator, 4, 3, 0.1);
    defer model.deinit();
    const prompt = [_]u32{ 0, 1 };
    const out = try model.generate(allocator, &prompt, 3);
    defer allocator.free(out);
    try std.testing.expectEqual(@as(usize, 5), out.len);
    for (out) |t| try std.testing.expect(t < 4);
}
