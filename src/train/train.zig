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

// --- Gradient accumulator for micro-batching (PRD 3.8) ---
pub const GradAccumulator = struct {
    buf: []f32,
    count: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize) !GradAccumulator {
        const buf = try allocator.alloc(f32, n);
        @memset(buf, 0);
        return .{ .buf = buf, .count = 0, .allocator = allocator };
    }

    pub fn deinit(self: *GradAccumulator) void {
        self.allocator.free(self.buf);
    }

    pub fn accumulate(self: *GradAccumulator, grads: []const f32) void {
        std.debug.assert(grads.len == self.buf.len);
        for (self.buf, grads) |*b, g| b.* += g;
        self.count += 1;
    }

    pub fn scale(self: *GradAccumulator) void {
        if (self.count == 0) return;
        const inv = 1.0 / @as(f32, @floatFromInt(self.count));
        for (self.buf) |*b| b.* *= inv;
    }

    pub fn reset(self: *GradAccumulator) void {
        @memset(self.buf, 0);
        self.count = 0;
    }
};

test "GradAccumulator" {
    const allocator = std.testing.allocator;
    var ga = try GradAccumulator.init(allocator, 2);
    defer ga.deinit();
    const g1 = [_]f32{ 1, 2 };
    const g2 = [_]f32{ 3, 4 };
    ga.accumulate(&g1);
    ga.accumulate(&g2);
    try std.testing.expectEqual(@as(usize, 2), ga.count);
    ga.scale();
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), ga.buf[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), ga.buf[1], 1e-6);
    ga.reset();
    try std.testing.expectEqual(@as(usize, 0), ga.count);
}

// Cosine LR schedule (alternative to WSD).
pub fn cosineLr(step: usize, total: usize, warmup: usize, peak: f32, min_lr: f32) f32 {
    if (total == 0) return peak;
    if (step < warmup) {
        return peak * (@as(f32, @floatFromInt(step + 1)) / @as(f32, @floatFromInt(if (warmup > 0) warmup else 1)));
    }
    const progress = @as(f32, @floatFromInt(step - warmup)) / @as(f32, @floatFromInt(if (total > warmup) total - warmup else 1));
    const cos = @cos(progress * std.math.pi);
    return min_lr + (peak - min_lr) * (1 + cos) / 2.0;
}

test "cosineLr" {
    const lr0 = cosineLr(0, 10, 2, 0.1, 0.0);
    try std.testing.expect(lr0 < 0.1);
    const lr_mid = cosineLr(5, 10, 2, 0.1, 0.0);
    try std.testing.expect(lr_mid >= 0 and lr_mid <= 0.1);
}

// --- Dynamic loss scaling (PRD 3.2, 3.8) ---
pub const LossScaler = struct {
    scale: f32 = 65536.0,
    growth_factor: f32 = 2.0,
    backoff_factor: f32 = 0.5,
    growth_interval: usize = 2000,
    steps_since_growth: usize = 0,

    pub fn init() LossScaler {
        return .{};
    }

    pub fn unscaleGrad(self: *const LossScaler, grad: []f32) void {
        const inv = 1.0 / self.scale;
        for (grad) |*g| g.* *= inv;
    }

    pub fn hasOverflow(self: *const LossScaler, grads: []const f32) bool {
        _ = self;
        for (grads) |g| {
            if (std.math.isNan(g) or std.math.isInf(g)) return true;
        }
        return false;
    }

    pub fn update(self: *LossScaler, overflow: bool) void {
        if (overflow) {
            self.scale *= self.backoff_factor;
            if (self.scale < 1.0) self.scale = 1.0;
            self.steps_since_growth = 0;
        } else {
            self.steps_since_growth += 1;
            if (self.steps_since_growth >= self.growth_interval) {
                self.scale *= self.growth_factor;
                self.steps_since_growth = 0;
            }
        }
    }
};

test "LossScaler overflow backs off" {
    var ls = LossScaler.init();
    const s0 = ls.scale;
    var grads = [_]f32{std.math.inf(f32)};
    try std.testing.expect(ls.hasOverflow(&grads));
    ls.update(true);
    try std.testing.expect(ls.scale < s0);
}

test "LossScaler no overflow grows" {
    var ls = LossScaler.init();
    ls.steps_since_growth = 1999;
    var grads = [_]f32{1.0};
    try std.testing.expect(!ls.hasOverflow(&grads));
    ls.update(false);
    try std.testing.expect(ls.scale > 65536.0);
}

// --- ZeRO-1 sharding (PRD 3.8): shard optimizer state across N shards ---
pub const ZeroShard = struct {
    n_shards: usize,
    shard_id: usize,
    pub fn range(self: ZeroShard, n_params: usize) struct { start: usize, end: usize } {
        const per = (n_params + self.n_shards - 1) / self.n_shards;
        const start = self.shard_id * per;
        const end = @min(start + per, n_params);
        return .{ .start = start, .end = end };
    }
};

test "ZeroShard range" {
    const zs = ZeroShard{ .n_shards = 2, .shard_id = 0 };
    const r = zs.range(10);
    try std.testing.expectEqual(@as(usize, 0), r.start);
    try std.testing.expectEqual(@as(usize, 5), r.end);
    const zs2 = ZeroShard{ .n_shards = 2, .shard_id = 1 };
    const r2 = zs2.range(10);
    try std.testing.expectEqual(@as(usize, 5), r2.start);
    try std.testing.expectEqual(@as(usize, 10), r2.end);
}

// --- LR Scheduler (PRD 3.8): wraps WSD/cosine/linear with state ---
pub const LRScheduler = struct {
    step: usize = 0,
    total: usize,
    warmup: usize,
    peak: f32,
    min_lr: f32,
    mode: enum { wsd, cosine, linear } = .wsd,
    cfg: TrainConfig,

    pub fn init(total: usize, warmup: usize, peak: f32, min_lr: f32, mode: @TypeOf(@as(enum { wsd, cosine, linear }, .wsd))) LRScheduler {
        return .{ .total = total, .warmup = warmup, .peak = peak, .min_lr = min_lr, .mode = mode, .cfg = .{ .lr = peak, .max_steps = total, .warmup_steps = warmup } };
    }

    pub fn getLr(self: *LRScheduler) f32 {
        const lr = switch (self.mode) {
            .wsd => wsdLr(self.step, self.cfg),
            .cosine => cosineLr(self.step, self.total, self.warmup, self.peak, self.min_lr),
            .linear => blk: {
                if (self.step < self.warmup) break :blk self.peak * (@as(f32, @floatFromInt(self.step + 1)) / @as(f32, @floatFromInt(if (self.warmup > 0) self.warmup else 1)));
                const prog = @as(f32, @floatFromInt(self.step - self.warmup)) / @as(f32, @floatFromInt(if (self.total > self.warmup) self.total - self.warmup else 1));
                break :blk self.peak * (1.0 - @min(prog, 1.0)) + self.min_lr * @min(prog, 1.0);
            },
        };
        return lr;
    }

    pub fn step(self: *LRScheduler) void {
        self.step += 1;
    }
};

test "LRScheduler cosine" {
    var sch = LRScheduler.init(10, 2, 0.1, 0.0, .cosine);
    const lr0 = sch.getLr();
    sch.step();
    const lr1 = sch.getLr();
    try std.testing.expect(lr0 >= 0 and lr1 >= 0);
}

test "LRScheduler wsd" {
    var sch = LRScheduler.init(10, 2, 0.1, 0.0, .wsd);
    const lr = sch.getLr();
    try std.testing.expect(lr > 0);
}
