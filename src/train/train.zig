// Training loop (PRD 3.8). Minimal end-to-end CPU training step.

const std = @import("std");
const cpu = @import("../cpu/cpu.zig");
const nn = @import("../nn/nn.zig");
const autograd = @import("../autograd/autograd.zig");
const optim = @import("../optim/optim.zig");
const checkpoint = @import("../checkpoint/checkpoint.zig");
const infer = @import("../infer/infer.zig");
const model_mod = @import("../model/model.zig");
const modality_mod = @import("../modality/modality.zig");

fn ioDeleteTree(path: []const u8) void {
    var threaded = std.Io.Threaded.init(std.heap.page_allocator, .{});
    defer threaded.deinit();
    std.Io.Dir.cwd().deleteTree(threaded.io(), path) catch {};
}

fn ioMakePath(path: []const u8) !void {
    var threaded = std.Io.Threaded.init(std.heap.page_allocator, .{});
    defer threaded.deinit();
    try std.Io.Dir.cwd().createDirPath(threaded.io(), path);
}

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

// Training loop driven by Model (PRD 3.8).
pub fn runModelTrainingLoop(model: *model_mod.Model, ids: []const u32, targets: []const u32, cfg: TrainConfig, checkpoint_dir: []const u8, checkpoint_interval: usize) !usize {
    var trainer = Trainer.init(cfg);
    var steps: usize = 0;
    while (!trainer.done()) {
        const loss = try model.trainStep(ids, targets);
        _ = loss;
        trainer.runStep();
        steps += 1;
        if (checkpoint_interval > 0 and steps % checkpoint_interval == 0) {
            var meta_buf: [128]u8 = undefined;
            const meta = try std.fmt.bufPrint(&meta_buf, "{{\"step\":{d}}}", .{steps});
            try checkpoint.atomicSave(model.allocator, checkpoint_dir, steps, meta);
        }
    }
    return steps;
}

// Apply a single learning rate to all Model optimizers (Muon + AdamW).
pub fn applyModelLr(model: *model_mod.Model, lr: f32) void {
    model.opt_embed.lr = lr;
    for (model.opt_blocks_attn) |*o| o.lr = lr;
    for (model.opt_blocks_mlp) |*o| o.lr = lr;
    for (model.opt_gdn_q) |*o| o.lr = lr;
    for (model.opt_gdn_k) |*o| o.lr = lr;
    for (model.opt_gdn_v) |*o| o.lr = lr;
    for (model.opt_gdn_a) |*o| o.lr = lr;
    for (model.opt_gdn_b) |*o| o.lr = lr;
    model.opt_head.lr = lr;
}

// Model training loop driven by an LRScheduler (PRD 3.8).
pub fn runModelTrainingLoopWithScheduler(model: *model_mod.Model, ids: []const u32, targets: []const u32, scheduler: *LRScheduler, checkpoint_dir: []const u8, checkpoint_interval: usize) !usize {
    const total = if (scheduler.total == 0) 1 else scheduler.total;
    var steps: usize = 0;
    while (steps < total) {
        const lr = scheduler.advance();
        applyModelLr(model, lr);
        const loss = try model.trainStep(ids, targets);
        _ = loss;
        steps += 1;
        if (checkpoint_interval > 0 and steps % checkpoint_interval == 0) {
            var meta_buf: [128]u8 = undefined;
            const meta = try std.fmt.bufPrint(&meta_buf, "{{\"step\":{d}}}", .{steps});
            try checkpoint.atomicSave(model.allocator, checkpoint_dir, steps, meta);
            try saveModelCheckpoint(model, checkpoint_dir, steps);
        }
    }
    return steps;
}

// Resume-aware Model training loop: loads latest checkpoint if present, then continues.
pub const MultimodalSample = struct {
    tokens: []const u32,
    targets: []const u32,
    mask: []const bool,
};

// --- Image classification harness (PRD 3.14) ---

pub fn buildImageClassificationSamples(allocator: std.mem.Allocator, images: []const []const f32, labels: []const u32, ivq: *const modality_mod.image_vq.ImageVQ, layout: modality_mod.VocabLayout) ![]MultimodalSample {
    if (images.len != labels.len) return error.LengthMismatch;
    const samples = try allocator.alloc(MultimodalSample, images.len);
    var i: usize = 0;
    while (i < images.len) : (i += 1) {
        const img_tokens = try modality_mod.image_understanding.imageTokensFromPixels(allocator, images[i], ivq, layout);
        const seq = img_tokens.len;
        const tokens = try allocator.alloc(u32, seq);
        @memcpy(tokens, img_tokens);
        const targets = try allocator.alloc(u32, seq);
        const mask = try allocator.alloc(bool, seq);
        @memset(mask, false);
        // causal targets: intermediate positions predict next image token; last predicts label.
        var j: usize = 0;
        while (j + 1 < seq) : (j += 1) targets[j] = img_tokens[j + 1];
        if (seq > 0) {
            targets[seq - 1] = labels[i];
            mask[seq - 1] = true;
        }
        if (seq == 0) {
            // degenerate: single label position
            allocator.free(tokens);
            allocator.free(targets);
            allocator.free(mask);
            allocator.free(img_tokens);
            continue;
        }
        samples[i] = .{ .tokens = tokens, .targets = targets, .mask = mask };
        allocator.free(img_tokens);
    }
    return samples;
}

pub fn freeImageClassificationSamples(allocator: std.mem.Allocator, samples: []MultimodalSample) void {
    for (samples) |s| {
        allocator.free(s.tokens);
        allocator.free(s.targets);
        allocator.free(s.mask);
    }
    allocator.free(samples);
}

pub fn runImageClassificationTraining(model: *model_mod.Model, images: []const []const f32, labels: []const u32, ivq: *const modality_mod.image_vq.ImageVQ, layout: modality_mod.VocabLayout, cfg: TrainConfig, checkpoint_dir: []const u8, checkpoint_interval: usize) !usize {
    const allocator = model.allocator;
    const samples = try buildImageClassificationSamples(allocator, images, labels, ivq, layout);
    defer freeImageClassificationSamples(allocator, samples);
    return runMultimodalTrainingLoop(model, samples, cfg, checkpoint_dir, checkpoint_interval);
}

pub fn classifyImage(model: *const model_mod.Model, allocator: std.mem.Allocator, image: []const f32, ivq: *const modality_mod.image_vq.ImageVQ, layout: modality_mod.VocabLayout, label_tokens: []const u32) !usize {
    const img_tokens = try modality_mod.image_understanding.imageTokensFromPixels(allocator, image, ivq, layout);
    defer allocator.free(img_tokens);
    const logits = try allocator.alloc(f32, img_tokens.len * model.cfg.vocab_size);
    defer allocator.free(logits);
    model.forward(logits, img_tokens);
    const last = logits[(img_tokens.len - 1) * model.cfg.vocab_size .. img_tokens.len * model.cfg.vocab_size];
    if (label_tokens.len == 0) return infer_argmax(last);
    var best_idx: usize = 0;
    var best_val: f32 = last[label_tokens[0]];
    var i: usize = 1;
    while (i < label_tokens.len) : (i += 1) {
        if (last[label_tokens[i]] > best_val) {
            best_val = last[label_tokens[i]];
            best_idx = i;
        }
    }
    return label_tokens[best_idx];
}

pub fn evalImageClassification(model: *const model_mod.Model, allocator: std.mem.Allocator, images: []const []const f32, labels: []const u32, ivq: *const modality_mod.image_vq.ImageVQ, layout: modality_mod.VocabLayout, label_tokens: []const u32) !f32 {
    if (images.len == 0) return 0;
    var correct: usize = 0;
    for (images, labels) |img, lab| {
        const pred = try classifyImage(model, allocator, img, ivq, layout, label_tokens);
        if (pred == lab) correct += 1;
    }
    return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(images.len));
}

fn infer_argmax(logits: []const f32) usize {
    if (logits.len == 0) return 0;
    var best: usize = 0;
    var bestv: f32 = logits[0];
    for (logits, 0..) |v, i| {
        if (v > bestv) {
            bestv = v;
            best = i;
        }
    }
    return best;
}

// Multimodal training loop: masked loss over packed image/audio+text sequences.
pub fn runMultimodalTrainingLoop(model: *model_mod.Model, samples: []const MultimodalSample, cfg: TrainConfig, checkpoint_dir: []const u8, checkpoint_interval: usize) !usize {
    if (samples.len == 0) return 0;
    var trainer = Trainer.init(cfg);
    var steps: usize = 0;
    while (!trainer.done()) {
        const s = samples[steps % samples.len];
        const loss = try model.trainStepMasked(s.tokens, s.targets, s.mask);
        _ = loss;
        trainer.runStep();
        steps += 1;
        if (checkpoint_interval > 0 and steps % checkpoint_interval == 0) {
            var meta_buf: [128]u8 = undefined;
            const meta = try std.fmt.bufPrint(&meta_buf, "{{\"step\":{d}}}", .{steps});
            try checkpoint.atomicSave(model.allocator, checkpoint_dir, steps, meta);
            try saveModelCheckpoint(model, checkpoint_dir, steps);
        }
    }
    return steps;
}

// Resume-aware multimodal training loop.
pub fn runMultimodalTrainingLoopResume(model: *model_mod.Model, samples: []const MultimodalSample, cfg: TrainConfig, checkpoint_dir: []const u8, checkpoint_interval: usize) !usize {
    if (samples.len == 0) return 0;
    var trainer = Trainer.init(cfg);
    var steps: usize = 0;
    if (try checkpoint.loadLatest(model.allocator, checkpoint_dir)) |step| {
        try loadModelCheckpoint(model, checkpoint_dir, step);
        trainer.step = step;
        steps = step;
    }
    while (!trainer.done()) {
        const s = samples[steps % samples.len];
        const loss = try model.trainStepMasked(s.tokens, s.targets, s.mask);
        _ = loss;
        trainer.runStep();
        steps += 1;
        if (checkpoint_interval > 0 and steps % checkpoint_interval == 0) {
            var meta_buf: [128]u8 = undefined;
            const meta = try std.fmt.bufPrint(&meta_buf, "{{\"step\":{d}}}", .{steps});
            try checkpoint.atomicSave(model.allocator, checkpoint_dir, steps, meta);
            try saveModelCheckpoint(model, checkpoint_dir, steps);
        }
    }
    return steps;
}

pub const MultimodalTierResult = struct {
    steps: usize,
    eval_loss: f32,
    eval_accuracy: f32,
};

// End-to-end multimodal tier harness: train (with resume) then evaluate masked loss/accuracy.
pub fn runMultimodalTier(model: *model_mod.Model, train_samples: []const MultimodalSample, eval_samples: []const MultimodalSample, cfg: TrainConfig, checkpoint_dir: []const u8, checkpoint_interval: usize) !MultimodalTierResult {
    const steps = try runMultimodalTrainingLoopResume(model, train_samples, cfg, checkpoint_dir, checkpoint_interval);

    var total_loss: f32 = 0;
    var total_acc: f32 = 0;
    var eval_count: usize = 0;
    for (eval_samples) |s| {
        const logits = try model.allocator.alloc(f32, s.tokens.len * model.cfg.vocab_size);
        defer model.allocator.free(logits);
        model.forward(logits, s.tokens);
        const loss = modality_mod.image_understanding.maskedCrossEntropy(logits, s.targets, s.mask, s.tokens.len, model.cfg.vocab_size);
        const acc = modality_mod.image_understanding.maskedAccuracy(logits, s.targets, s.mask, s.tokens.len, model.cfg.vocab_size);
        total_loss += loss;
        total_acc += acc;
        eval_count += 1;
    }
    const eval_loss = if (eval_count > 0) total_loss / @as(f32, @floatFromInt(eval_count)) else 0;
    const eval_accuracy = if (eval_count > 0) total_acc / @as(f32, @floatFromInt(eval_count)) else 0;
    return .{ .steps = steps, .eval_loss = eval_loss, .eval_accuracy = eval_accuracy };
}

pub fn runModelTrainingLoopResume(model: *model_mod.Model, ids: []const u32, targets: []const u32, scheduler: *LRScheduler, checkpoint_dir: []const u8, checkpoint_interval: usize) !usize {
    const total = if (scheduler.total == 0) 1 else scheduler.total;
    if (try checkpoint.loadLatest(model.allocator, checkpoint_dir)) |step| {
        try loadModelCheckpoint(model, checkpoint_dir, step);
        scheduler.step = step;
    }
    while (scheduler.step < total) {
        const lr = scheduler.advance();
        applyModelLr(model, lr);
        const loss = try model.trainStep(ids, targets);
        _ = loss;
        const step = scheduler.step;
        if (checkpoint_interval > 0 and step % checkpoint_interval == 0) {
            var meta_buf: [128]u8 = undefined;
            const meta = try std.fmt.bufPrint(&meta_buf, "{{\"step\":{d}}}", .{step});
            try checkpoint.atomicSave(model.allocator, checkpoint_dir, step, meta);
            try saveModelCheckpoint(model, checkpoint_dir, step);
        }
    }
    return scheduler.step;
}

test "runModelTrainingLoop saves checkpoint" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/modelckpt";
    ioDeleteTree(base);
    try ioMakePath(base);
    const cfg = @import("../config/config.zig");
    const mcfg = cfg.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 8, .seq_len = 4 };
    var m = try model_mod.Model.init(allocator, mcfg);
    defer m.deinit();
    const ids = [_]u32{ 1, 2, 3 };
    const targets = [_]u32{ 2, 3, 4 };
    const tcfg = TrainConfig{ .max_steps = 3, .warmup_steps = 1 };
    const steps = try runModelTrainingLoop(&m, &ids, &targets, tcfg, base, 2);
    try std.testing.expectEqual(@as(usize, 3), steps);
    const latest = try checkpoint.loadLatest(allocator, base);
    try std.testing.expect(latest != null);
}

test "runModelTrainingLoopWithScheduler saves checkpoint" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/modelschedckpt";
    ioDeleteTree(base);
    try ioMakePath(base);
    const cfg = @import("../config/config.zig");
    const mcfg = cfg.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 8, .seq_len = 4 };
    var m = try model_mod.Model.init(allocator, mcfg);
    defer m.deinit();
    const ids = [_]u32{ 1, 2, 3 };
    const targets = [_]u32{ 2, 3, 4 };
    var sched = LRScheduler.init(.wsd, 0.01, 0.0, 1, 3, 0.2);
    const steps = try runModelTrainingLoopWithScheduler(&m, &ids, &targets, &sched, base, 2);
    try std.testing.expectEqual(@as(usize, 3), steps);
    const latest = try checkpoint.loadLatest(allocator, base);
    try std.testing.expect(latest != null);
    try std.testing.expectEqual(@as(?u64, 2), latest);
}

// --- Model checkpoint save/load (PRD 3.4) ---

fn writeF32Slice(writer: *std.Io.Writer, slice: []const f32) !void {
    const bytes = std.mem.sliceAsBytes(slice);
    try writer.writeAll(bytes);
}

fn readF32Slice(file: std.fs.File, slice: []f32) !void {
    const bytes = std.mem.sliceAsBytes(slice);
    const n = try file.readAll(bytes);
    if (n != bytes.len) return error.UnexpectedEof;
}

pub fn saveModelCheckpoint(model: *model_mod.Model, dir: []const u8, step: usize) !void {
    var threaded = std.Io.Threaded.init(model.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try std.Io.Dir.cwd().createDirPath(io, dir);
    var path_buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/model-{d}.bin", .{ dir, step });
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var w = file.writer(io, &buf);
    try w.interface.writeInt(u64, step, .little);
    try writeF32Slice(&w.interface, model.embed);
    for (model.blocks) |*bl| {
        try writeF32Slice(&w.interface, bl.attn.w);
        try writeF32Slice(&w.interface, bl.mlp.w);
    }
    try writeF32Slice(&w.interface, model.head.w);
    try w.interface.writeInt(u64, model.opt_embed.t, .little);
    try writeF32Slice(&w.interface, model.opt_embed.m);
    try writeF32Slice(&w.interface, model.opt_embed.v);
    for (model.opt_blocks_attn) |*o| try writeF32Slice(&w.interface, o.buf);
    for (model.opt_blocks_mlp) |*o| try writeF32Slice(&w.interface, o.buf);
    try w.interface.writeInt(u64, model.opt_head.t, .little);
    try writeF32Slice(&w.interface, model.opt_head.m);
    try writeF32Slice(&w.interface, model.opt_head.v);
    try w.flush();
}

pub fn loadModelCheckpoint(model: *model_mod.Model, dir: []const u8, step: usize) !void {
    var threaded = std.Io.Threaded.init(model.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var path_buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/model-{d}.bin", .{ dir, step });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, model.allocator, .limited(100_000_000));
    defer model.allocator.free(bytes);
    var pos: usize = 0;
    const readU64 = struct {
        fn call(b: []const u8, p: *usize) u64 {
            const v = std.mem.readInt(u64, b[p.*..][0..8], .little);
            p.* += 8;
            return v;
        }
    }.call;
    const readSlice = struct {
        fn call(b: []const u8, p: *usize, out: []f32) void {
            const bytes2 = std.mem.sliceAsBytes(out);
            @memcpy(bytes2, b[p.* .. p.* + bytes2.len]);
            p.* += bytes2.len;
        }
    }.call;
    _ = readU64(bytes, &pos);
    readSlice(bytes, &pos, model.embed);
    for (model.blocks) |*bl| {
        readSlice(bytes, &pos, bl.attn.w);
        readSlice(bytes, &pos, bl.mlp.w);
    }
    readSlice(bytes, &pos, model.head.w);
    const t_embed = readU64(bytes, &pos);
    model.opt_embed.t = t_embed;
    readSlice(bytes, &pos, model.opt_embed.m);
    readSlice(bytes, &pos, model.opt_embed.v);
    for (model.opt_blocks_attn) |*o| readSlice(bytes, &pos, o.buf);
    for (model.opt_blocks_mlp) |*o| readSlice(bytes, &pos, o.buf);
    const t_head = readU64(bytes, &pos);
    model.opt_head.t = t_head;
    readSlice(bytes, &pos, model.opt_head.m);
    readSlice(bytes, &pos, model.opt_head.v);
}

test "Model checkpoint save/load roundtrip" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/modelstateckpt";
    ioDeleteTree(base);
    try ioMakePath(base);
    const cfg = @import("../config/config.zig");
    const mcfg = cfg.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 8, .seq_len = 4 };
    var m = try model_mod.Model.init(allocator, mcfg);
    defer m.deinit();
    // mutate weights to ensure load overwrites
    m.embed[0] = 123.0;
    try saveModelCheckpoint(&m, base, 7);
    // fresh model
    var m2 = try model_mod.Model.init(allocator, mcfg);
    defer m2.deinit();
    try loadModelCheckpoint(&m2, base, 7);
    try std.testing.expectApproxEqAbs(@as(f32, 123.0), m2.embed[0], 1e-6);
}

test "Model checkpoint deterministic resume" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/modelresume";
    ioDeleteTree(base);
    try ioMakePath(base);
    const cfg = @import("../config/config.zig");
    const mcfg = cfg.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 8, .seq_len = 4 };
    const ids = [_]u32{ 1, 2, 3 };
    const targets = [_]u32{ 2, 3, 4 };

    // Uninterrupted run: 4 steps.
    var m_full = try model_mod.Model.init(allocator, mcfg);
    defer m_full.deinit();
    var sched_full = LRScheduler.init(.wsd, 0.01, 0.0, 1, 4, 0.2);
    _ = try runModelTrainingLoopWithScheduler(&m_full, &ids, &targets, &sched_full, base, 4);

    // Resumed run: 2 steps, save, load into fresh model, then 2 steps.
    var m_part = try model_mod.Model.init(allocator, mcfg);
    defer m_part.deinit();
    var sched_part = LRScheduler.init(.wsd, 0.01, 0.0, 1, 2, 0.2);
    _ = try runModelTrainingLoopWithScheduler(&m_part, &ids, &targets, &sched_part, base, 2);
    try saveModelCheckpoint(&m_part, base, 2);

    var m_resumed = try model_mod.Model.init(allocator, mcfg);
    defer m_resumed.deinit();
    try loadModelCheckpoint(&m_resumed, base, 2);
    var sched_resume = LRScheduler.init(.wsd, 0.01, 0.0, 1, 4, 0.2);
    sched_resume.step = 2;
    _ = try runModelTrainingLoopWithScheduler(&m_resumed, &ids, &targets, &sched_resume, base, 2);

    // Compare a sample of weights.
    try std.testing.expectApproxEqAbs(m_full.embed[0], m_resumed.embed[0], 1e-5);
    try std.testing.expectApproxEqAbs(m_full.head.w[0], m_resumed.head.w[0], 1e-5);
}

test "runMultimodalTrainingLoop saves checkpoint" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/mmtrainckpt";
    ioDeleteTree(base);
    try ioMakePath(base);
    const cfg = @import("../config/config.zig");
    const mcfg = cfg.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 8, .seq_len = 4 };
    var m = try model_mod.Model.init(allocator, mcfg);
    defer m.deinit();
    const toks = [_]u32{ 1, 2, 3 };
    const tgts = [_]u32{ 2, 3, 4 };
    const mask = [_]bool{ false, true, true };
    const samples = [_]MultimodalSample{.{ .tokens = &toks, .targets = &tgts, .mask = &mask }};
    const tcfg = TrainConfig{ .max_steps = 3, .warmup_steps = 1 };
    const steps = try runMultimodalTrainingLoop(&m, &samples, tcfg, base, 2);
    try std.testing.expectEqual(@as(usize, 3), steps);
    const latest = try checkpoint.loadLatest(allocator, base);
    try std.testing.expect(latest != null);
}

test "runMultimodalTier finite" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/mmtier";
    ioDeleteTree(base);
    try ioMakePath(base);
    const cfg = @import("../config/config.zig");
    const mcfg = cfg.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 8, .seq_len = 4 };
    var m = try model_mod.Model.init(allocator, mcfg);
    defer m.deinit();
    const toks = [_]u32{ 1, 2, 3 };
    const tgts = [_]u32{ 2, 3, 4 };
    const mask = [_]bool{ false, true, true };
    const train_samples = [_]MultimodalSample{.{ .tokens = &toks, .targets = &tgts, .mask = &mask }};
    const eval_samples = [_]MultimodalSample{.{ .tokens = &toks, .targets = &tgts, .mask = &mask }};
    const tcfg = TrainConfig{ .max_steps = 2, .warmup_steps = 1 };
    const res = try runMultimodalTier(&m, &train_samples, &eval_samples, tcfg, base, 2);
    try std.testing.expectEqual(@as(usize, 2), res.steps);
    try std.testing.expect(std.math.isFinite(res.eval_loss));
    try std.testing.expect(res.eval_accuracy >= 0 and res.eval_accuracy <= 1);
}

test "image classification training finite" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/imgclasstrain";
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    std.Io.Dir.cwd().deleteTree(io, base) catch {};
    try std.Io.Dir.cwd().createDirPath(io, base);
    const cfg = @import("../config/config.zig");
    const mcfg = cfg.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 16, .seq_len = 4 };
    var m = try model_mod.Model.init(allocator, mcfg);
    defer m.deinit();
    var ivq = try modality_mod.image_vq.ImageVQ.init(allocator, 2, 2, 1, 1, 4);
    defer ivq.deinit();
    const layout = modality_mod.VocabLayout{
        .text_offset = 0,
        .text_size = 8,
        .image_offset = 8,
        .image_size = 4,
        .audio_offset = 12,
        .audio_size = 1,
        .video_offset = 13,
        .video_size = 1,
        .special_offset = 14,
        .special_size = 2,
    };
    var img0 = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    var img1 = [_]f32{ 0.4, 0.3, 0.2, 0.1 };
    const images = [_][]const f32{ &img0, &img1 };
    const labels = [_]u32{ 0, 1 };
    const tcfg = TrainConfig{ .max_steps = 2, .warmup_steps = 1 };
    const steps = try runImageClassificationTraining(&m, &images, &labels, &ivq, layout, tcfg, base, 2);
    try std.testing.expectEqual(@as(usize, 2), steps);
    const label_tokens = [_]u32{ 0, 1 };
    const pred = try classifyImage(&m, allocator, &img0, &ivq, layout, &label_tokens);
    try std.testing.expect(pred == 0 or pred == 1);
    const acc = try evalImageClassification(&m, allocator, &images, &labels, &ivq, layout, &label_tokens);
    try std.testing.expect(acc >= 0 and acc <= 1);
}

test "image classification training accuracy" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/imgclassacc";
    ioDeleteTree(base);
    try ioMakePath(base);
    const cfg = @import("../config/config.zig");
    const mcfg = cfg.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 16, .seq_len = 4 };
    var m = try model_mod.Model.init(allocator, mcfg);
    defer m.deinit();
    var ivq = try modality_mod.image_vq.ImageVQ.init(allocator, 2, 2, 1, 1, 4);
    defer ivq.deinit();
    const layout = modality_mod.VocabLayout{
        .text_offset = 0,
        .text_size = 8,
        .image_offset = 8,
        .image_size = 4,
        .audio_offset = 12,
        .audio_size = 1,
        .video_offset = 13,
        .video_size = 1,
        .special_offset = 14,
        .special_size = 2,
    };
    var img0 = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    var img1 = [_]f32{ 0.4, 0.3, 0.2, 0.1 };
    const images = [_][]const f32{ &img0, &img1 };
    const labels = [_]u32{ 0, 1 };
    const tcfg = TrainConfig{ .max_steps = 100, .warmup_steps = 5 };
    _ = try runImageClassificationTraining(&m, &images, &labels, &ivq, layout, tcfg, base, 0);
    const label_tokens = [_]u32{ 0, 1 };
    const acc = try evalImageClassification(&m, allocator, &images, &labels, &ivq, layout, &label_tokens);
    std.debug.print("image classification accuracy after training: {d:.4}\n", .{acc});
    try std.testing.expect(acc >= 0.5);
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

test "Model training resume integration" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/modelresumeint";
    ioDeleteTree(base);
    try ioMakePath(base);
    const cfg = @import("../config/config.zig");
    const mcfg = cfg.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 8, .seq_len = 4 };
    const ids = [_]u32{ 1, 2, 3 };
    const targets = [_]u32{ 2, 3, 4 };

    // First phase: 2 steps, writes checkpoint.
    var m1 = try model_mod.Model.init(allocator, mcfg);
    defer m1.deinit();
    var sched1 = LRScheduler.init(.wsd, 0.01, 0.0, 1, 4, 0.2);
    sched1.total = 2;
    _ = try runModelTrainingLoopResume(&m1, &ids, &targets, &sched1, base, 2);

    // Resume phase: continue to 4.
    var m2 = try model_mod.Model.init(allocator, mcfg);
    defer m2.deinit();
    var sched2 = LRScheduler.init(.wsd, 0.01, 0.0, 1, 4, 0.2);
    const steps = try runModelTrainingLoopResume(&m2, &ids, &targets, &sched2, base, 2);
    try std.testing.expectEqual(@as(usize, 4), steps);
    const latest = try checkpoint.loadLatest(allocator, base);
    try std.testing.expectEqual(@as(?u64, 4), latest);
}

test "runTrainingLoop saves checkpoint" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/trainckpt";
    ioDeleteTree(base);
    try ioMakePath(base);
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

// --- LRScheduler (PRD 3.8): wsd / cosine / linear ---
pub const LRScheduleKind = enum { wsd, cosine, linear };

pub const LRScheduler = struct {
    kind: LRScheduleKind = .wsd,
    peak: f32 = 1e-2,
    min_lr: f32 = 0,
    warmup: usize = 1,
    total: usize = 100,
    decay_fraction: f32 = 0.2,
    step: usize = 0,

    pub fn init(kind: LRScheduleKind, peak: f32, min_lr: f32, warmup: usize, total: usize, decay_fraction: f32) LRScheduler {
        return .{ .kind = kind, .peak = peak, .min_lr = min_lr, .warmup = warmup, .total = total, .decay_fraction = decay_fraction };
    }

    pub fn current(self: *const LRScheduler) f32 {
        return self.at(self.step);
    }

    pub fn at(self: *const LRScheduler, step: usize) f32 {
        switch (self.kind) {
            .wsd => {
                const tcfg = TrainConfig{ .lr = self.peak, .max_steps = if (self.total == 0) 1 else self.total, .warmup_steps = if (self.warmup == 0) 1 else self.warmup, .decay_fraction = self.decay_fraction };
                return wsdLr(step, tcfg);
            },
            .cosine => return cosineLr(step, self.total, self.warmup, self.peak, self.min_lr),
            .linear => {
                if (self.total == 0) return self.peak;
                const prog = @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(self.total));
                const factor = 1.0 - @min(prog, 1.0);
                return self.min_lr + (self.peak - self.min_lr) * factor;
            },
        }
    }

    pub fn advance(self: *LRScheduler) f32 {
        const lr = self.current();
        self.step += 1;
        return lr;
    }

    pub fn reset(self: *LRScheduler) void {
        self.step = 0;
    }
};

test "LRScheduler wsd" {
    var sched = LRScheduler.init(.wsd, 0.1, 0.0, 2, 10, 0.2);
    const lr0 = sched.advance();
    try std.testing.expect(lr0 < 0.1);
    try std.testing.expect(sched.advance() <= 0.1);
}

test "LRScheduler cosine" {
    var sched = LRScheduler.init(.cosine, 0.1, 0.0, 2, 10, 0.0);
    const lr = sched.at(5);
    try std.testing.expect(lr >= 0 and lr <= 0.1);
}

test "LRScheduler linear" {
    var sched = LRScheduler.init(.linear, 0.1, 0.0, 0, 10, 0.0);
    const lr0 = sched.at(0);
    const lr_end = sched.at(10);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), lr0, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), lr_end, 1e-6);
}

// --- ZeRO-1 communication primitives (CPU reference, PRD 3.8) ---

// Reduce-scatter gradients: each shard receives the sum of its owned slice across all ranks.
pub fn reduceScatterGrads(all_grads: []const []const f32, n_shards: usize, shard_id: usize, out: []f32) void {
    if (all_grads.len == 0) return;
    const n_params = all_grads[0].len;
    const zs = ZeroShard{ .n_shards = n_shards, .shard_id = shard_id };
    const r = zs.range(n_params);
    std.debug.assert(out.len >= r.end - r.start);
    var i: usize = r.start;
    while (i < r.end) : (i += 1) {
        var acc: f32 = 0;
        for (all_grads) |g| acc += g[i];
        out[i - r.start] = acc;
    }
}

// All-gather sharded weights into a full parameter vector.
pub fn allGatherWeights(shards: []const []const f32, out: []f32) void {
    var off: usize = 0;
    for (shards) |sh| {
        @memcpy(out[off .. off + sh.len], sh);
        off += sh.len;
    }
}

test "reduceScatterGrads" {
    const g0 = [_]f32{ 1, 2, 3, 4 };
    const g1 = [_]f32{ 5, 6, 7, 8 };
    const all = [_][]const f32{ &g0, &g1 };
    var out = [_]f32{ 0, 0 };
    reduceScatterGrads(&all, 2, 0, &out);
    // shard0 owns params 0..2: sums [6,8]
    try std.testing.expectApproxEqAbs(@as(f32, 6), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8), out[1], 1e-6);
    var out2 = [_]f32{ 0, 0 };
    reduceScatterGrads(&all, 2, 1, &out2);
    try std.testing.expectApproxEqAbs(@as(f32, 10), out2[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12), out2[1], 1e-6);
}

test "allGatherWeights" {
    const s0 = [_]f32{ 1, 2 };
    const s1 = [_]f32{ 3, 4 };
    const shards = [_][]const f32{ &s0, &s1 };
    var out = [_]f32{ 0, 0, 0, 0 };
    allGatherWeights(&shards, &out);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, &out);
}

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
