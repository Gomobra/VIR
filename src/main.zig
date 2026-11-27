const std = @import("std");
const root = @import("root.zig");
const model_mod = @import("model/model.zig");
const train_mod = @import("train/train.zig");
const modality = @import("modality/modality.zig");
const config = @import("config/config.zig");

const MnistData = struct {
    images: [][]f32,
    labels: []u32,
    n: usize,
    rows: usize,
    cols: usize,
};

fn loadMnist(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, image_name: []const u8, label_name: []const u8, max_n: usize) !MnistData {
    var img_buf: [512]u8 = undefined;
    var lbl_buf: [512]u8 = undefined;
    const img_path = try std.fmt.bufPrint(&img_buf, "{s}/{s}", .{ dir, image_name });
    const lbl_path = try std.fmt.bufPrint(&lbl_buf, "{s}/{s}", .{ dir, label_name });

    const img_bytes = try std.Io.Dir.cwd().readFileAlloc(io, img_path, allocator, .limited(100_000_000));
    defer allocator.free(img_bytes);
    const lbl_bytes = try std.Io.Dir.cwd().readFileAlloc(io, lbl_path, allocator, .limited(100_000_000));
    defer allocator.free(lbl_bytes);

    if (img_bytes.len < 16) return error.BadImageHeader;
    const magic = std.mem.readInt(u32, img_bytes[0..4], .big);
    if (magic != 0x00000803) return error.BadImageMagic;
    const n_images = std.mem.readInt(u32, img_bytes[4..8], .big);
    const rows = std.mem.readInt(u32, img_bytes[8..12], .big);
    const cols = std.mem.readInt(u32, img_bytes[12..16], .big);
    if (lbl_bytes.len < 8) return error.BadLabelHeader;
    const lbl_magic = std.mem.readInt(u32, lbl_bytes[0..4], .big);
    if (lbl_magic != 0x00000801) return error.BadLabelMagic;
    const n_labels = std.mem.readInt(u32, lbl_bytes[4..8], .big);

    const n = @min(max_n, @min(n_images, n_labels));
    const pixels = rows * cols;
    const images = try allocator.alloc([]f32, n);
    const labels = try allocator.alloc(u32, n);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const img = try allocator.alloc(f32, pixels);
        var j: usize = 0;
        while (j < pixels) : (j += 1) {
            const off = 16 + i * pixels + j;
            img[j] = @as(f32, @floatFromInt(img_bytes[off])) / 255.0;
        }
        images[i] = img;
        labels[i] = lbl_bytes[8 + i];
    }
    return .{ .images = images, .labels = labels, .n = n, .rows = rows, .cols = cols };
}

pub fn main(init: std.process.Init) void {
    std.debug.print("vir {s}\n", .{root.version});
    const allocator = std.heap.page_allocator;
    const io = init.io;

    const mnist_dir_c = std.c.getenv("MNIST_DIR");
    const mnist_dir: ?[]const u8 = if (mnist_dir_c) |p| std.mem.span(p) else null;
    if (mnist_dir) |dir| {
        std.debug.print("loading MNIST from {s}\n", .{dir});
        const train = loadMnist(io, allocator, dir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 300) catch |err| {
            std.debug.print("MNIST load failed: {s}\n", .{@errorName(err)});
            return;
        };
        const testdata = loadMnist(io, allocator, dir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 100) catch |err| {
            std.debug.print("MNIST test load failed: {s}\n", .{@errorName(err)});
            return;
        };

        const layout = modality.VocabLayout{
            .text_offset = 0,
            .text_size = 10,
            .image_offset = 10,
            .image_size = 16,
            .audio_offset = 26,
            .audio_size = 1,
            .video_offset = 27,
            .video_size = 1,
            .special_offset = 28,
            .special_size = 4,
        };

        var ivq = modality.image_vq.ImageVQ.init(allocator, 28, 28, 4, 4, 16) catch {
            std.debug.print("ivq init failed\n", .{});
            return;
        };
        defer ivq.deinit();

        const mcfg = config.ModelConfig{ .d_model = 64, .n_groups = 1, .vocab_size = 32, .seq_len = 64 };
        var model = model_mod.Model.init(allocator, mcfg) catch {
            std.debug.print("model init failed\n", .{});
            return;
        };
        defer model.deinit();
        model.opt_embed.lr = 0.001;
        model.opt_head.lr = 0.001;
        model.opt_mtp_head.lr = 0.001;
        for (model.opt_blocks_attn) |*o| o.lr = 0.001;
        for (model.opt_blocks_mlp) |*o| o.lr = 0.001;
        for (model.opt_gdn_q) |*o| o.lr = 0.001;
        for (model.opt_gdn_k) |*o| o.lr = 0.001;
        for (model.opt_gdn_v) |*o| o.lr = 0.001;
        for (model.opt_gdn_a) |*o| o.lr = 0.001;
        for (model.opt_gdn_b) |*o| o.lr = 0.001;
        model.opt_loop_embed.lr = 0.001;

        const base = "/tmp/scratchpad/mnisttrain";
        std.Io.Dir.cwd().deleteTree(io, base) catch {};
        std.Io.Dir.cwd().createDirPath(io, base) catch {};

        // Use a binary subset (digits 0 and 1) for clearer learning.
        var bin_images = std.ArrayList([]const f32).empty;
        defer bin_images.deinit(allocator);
        var bin_labels = std.ArrayList(u32).empty;
        defer bin_labels.deinit(allocator);
        var i: usize = 0;
        while (i < train.n and bin_images.items.len < 200) : (i += 1) {
            if (train.labels[i] == 0 or train.labels[i] == 1) {
                bin_images.append(allocator, train.images[i]) catch return;
                bin_labels.append(allocator, train.labels[i]) catch return;
            }
        }
        const train_images = bin_images.items;
        const train_labels = bin_labels.items;

        // Direct linear classifier baseline on VQ code histograms.
        const n_codes = 16; // comptime-known for array repeat
        const w = allocator.alloc(f32, 10 * n_codes) catch return;
        defer allocator.free(w);
        const bias = allocator.alloc(f32, 10) catch return;
        defer allocator.free(bias);
        @memset(w, 0);
        @memset(bias, 0);
        var e: usize = 0;
        while (e < 10) : (e += 1) {
            var ii: usize = 0;
            while (ii < train.n) : (ii += 1) {
                const codes = allocator.alloc(usize, 49) catch return;
                defer allocator.free(codes);
                _ = ivq.encode(train_images[ii], codes);
                var hist: [16]f32 = @splat(0);
                var ci: usize = 0;
                while (ci < 49) : (ci += 1) hist[codes[ci]] += 1;
                // softmax
                var logits: [10]f32 = @splat(0);
                var j: usize = 0;
                while (j < 10) : (j += 1) {
                    var acc: f32 = bias[j];
                    var k: usize = 0;
                    while (k < n_codes) : (k += 1) acc += w[j * n_codes + k] * hist[k];
                    logits[j] = acc;
                }
                var mx: f32 = logits[0];
                j = 1;
                while (j < 10) : (j += 1) if (logits[j] > mx) {
                    mx = logits[j];
                };
                var sum: f32 = 0;
                j = 0;
                while (j < 10) : (j += 1) {
                    logits[j] = @exp(logits[j] - mx);
                    sum += logits[j];
                }
                j = 0;
                while (j < 10) : (j += 1) logits[j] /= sum;
                const y = train_labels[ii];
                // gradient: dlogit = prob - onehot
                var grads: [10]f32 = @splat(0);
                j = 0;
                while (j < 10) : (j += 1) grads[j] = logits[j];
                grads[y] -= 1;
                // update
                j = 0;
                while (j < 10) : (j += 1) {
                    bias[j] -= 0.1 * grads[j];
                    var k: usize = 0;
                    while (k < n_codes) : (k += 1) {
                        w[j * n_codes + k] -= 0.1 * grads[j] * hist[k];
                    }
                }
            }
        }
        // eval
        var correct: usize = 0;
        var ei: usize = 0;
        while (ei < testdata.n) : (ei += 1) {
            const codes = allocator.alloc(usize, 49) catch return;
            defer allocator.free(codes);
            _ = ivq.encode(testdata.images[ei], codes);
            var hist: [16]f32 = @splat(0);
            var ci: usize = 0;
            while (ci < 49) : (ci += 1) hist[codes[ci]] += 1;
            var logits: [10]f32 = @splat(0);
            var j: usize = 0;
            while (j < 10) : (j += 1) {
                var acc: f32 = bias[j];
                var k: usize = 0;
                while (k < n_codes) : (k += 1) acc += w[j * n_codes + k] * hist[k];
                logits[j] = acc;
            }
            // argmax
            var best: usize = 0;
            var bestv: f32 = logits[0];
            j = 1;
            while (j < 10) : (j += 1) {
                if (logits[j] > bestv) {
                    bestv = logits[j];
                    best = j;
                }
            }
            if (best == testdata.labels[ei]) correct += 1;
        }
        const baseline_acc = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(testdata.n));
        std.debug.print("direct VQ-histogram classifier accuracy: {d:.4}\n", .{baseline_acc});

        // Train the VQ codebook on MNIST patches (EMA).
        var epoch: usize = 0;
        while (epoch < 3) : (epoch += 1) {
            // reset EMA accumulators
            @memset(ivq.vq.ema_count, 0);
            @memset(ivq.vq.ema_sum, 0);
            for (train_images) |img| {
                const n_patches = 7 * 7;
                const xs = allocator.alloc(f32, n_patches * ivq.vq.dim) catch return;
                defer allocator.free(xs);
                const codes = allocator.alloc(usize, n_patches) catch return;
                defer allocator.free(codes);
                var pi: usize = 0;
                var py: usize = 0;
                while (py < 7) : (py += 1) {
                    var px: usize = 0;
                    while (px < 7) : (px += 1) {
                        var ii: usize = 0;
                        var y: usize = 0;
                        while (y < 4) : (y += 1) {
                            var x: usize = 0;
                            while (x < 4) : (x += 1) {
                                const iy = py * 4 + y;
                                const ix = px * 4 + x;
                                xs[pi * ivq.vq.dim + ii] = img[iy * 28 + ix];
                                ii += 1;
                            }
                        }
                        codes[pi] = ivq.vq.quantize(xs[pi * ivq.vq.dim .. (pi + 1) * ivq.vq.dim]);
                        pi += 1;
                    }
                }
                ivq.vq.emaUpdate(xs, codes, n_patches);
            }
            // restart dead codes using first image patches
            var data_buf = allocator.alloc(f32, 49 * 16) catch return;
            defer allocator.free(data_buf);
            var pi: usize = 0;
            var py: usize = 0;
            while (py < 7) : (py += 1) {
                var px: usize = 0;
                while (px < 7) : (px += 1) {
                    var ii: usize = 0;
                    var y: usize = 0;
                    while (y < 4) : (y += 1) {
                        var x: usize = 0;
                        while (x < 4) : (x += 1) {
                            const iy = py * 4 + y;
                            const ix = px * 4 + x;
                            data_buf[pi * 16 + ii] = train_images[0][iy * 28 + ix];
                            ii += 1;
                        }
                    }
                    pi += 1;
                }
            }
            ivq.vq.restartDead(data_buf, 49);
            std.debug.print("VQ epoch {d} done\n", .{epoch});
        }

        // Debug: print VQ codes for first image.
        const dbg_codes = allocator.alloc(usize, 49) catch return;
        defer allocator.free(dbg_codes);
        const dbg_np = ivq.encode(train.images[0], dbg_codes);
        std.debug.print("first image VQ codes: ", .{});
        var di: usize = 0;
        while (di < dbg_np) : (di += 1) std.debug.print("{d} ", .{dbg_codes[di]});
        std.debug.print("\n", .{});

        const samples = train_mod.buildImageClassificationSamples(allocator, train_images, train_labels, &ivq, layout) catch {
            std.debug.print("sample build failed\n", .{});
            return;
        };
        defer train_mod.freeImageClassificationSamples(allocator, samples);

        const w_before = model.embed[0];
        const max_steps: usize = 300;
        var step: usize = 0;
        while (step < max_steps) : (step += 1) {
            const s = samples[step % samples.len];
            const loss = model.trainStepMasked(s.tokens, s.targets, s.mask) catch {
                std.debug.print("training step failed\n", .{});
                return;
            };
            if (step % 25 == 0 or step == max_steps - 1) {
                std.debug.print("step {d}: loss={d:.4}\n", .{ step, loss });
            }
        }

        std.debug.print("embed[0] before/after: {d:.6} -> {d:.6}\n", .{ w_before, model.embed[0] });
        // Debug: print last-patch VQ code per digit for first 10 training images.
        var k: usize = 0;
        while (k < 10 and k < train.n) : (k += 1) {
            const codes = allocator.alloc(usize, 49) catch return;
            defer allocator.free(codes);
            _ = ivq.encode(train.images[k], codes);
            std.debug.print("train image {d} label {d} last_code {d}\n", .{ k, train.labels[k], codes[48] });
        }
        // Debug: print VQ code sequences for first few images.
        var jj: usize = 0;
        while (jj < 3 and jj < train.n) : (jj += 1) {
            const codes = allocator.alloc(usize, 49) catch return;
            defer allocator.free(codes);
            _ = ivq.encode(train.images[jj], codes);
            std.debug.print("image {d} label {d} codes:", .{ jj, train.labels[jj] });
            var ci: usize = 0;
            while (ci < 49) : (ci += 1) std.debug.print(" {d}", .{codes[ci]});
            std.debug.print("\n", .{});
        }
        std.debug.print("MNIST training steps: {d}\n", .{max_steps});

        const label_tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        const test_images = train_images;
        const test_labels = train_labels;
        const acc = train_mod.evalImageClassification(&model, allocator, test_images, test_labels, &ivq, layout, &label_tokens) catch 0;
        std.debug.print("MNIST test accuracy after CPU training: {d:.4}\n", .{acc});

        var ii: usize = 0;
        while (ii < 5 and ii < testdata.n) : (ii += 1) {
            const pred = train_mod.classifyImage(&model, allocator, testdata.images[ii], &ivq, layout, &label_tokens) catch 0;
            std.debug.print("test image {d}: predicted {d}, expected {d}\n", .{ ii, pred, testdata.labels[ii] });
        }
        return;
    }

    // Synthetic fallback demo.
    const mcfg = config.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 16, .seq_len = 4 };
    var model = model_mod.Model.init(allocator, mcfg) catch {
        std.debug.print("model init failed\n", .{});
        return;
    };
    defer model.deinit();

    var ivq = modality.image_vq.ImageVQ.init(allocator, 2, 2, 1, 1, 4) catch {
        std.debug.print("ivq init failed\n", .{});
        return;
    };
    defer ivq.deinit();

    const layout = modality.VocabLayout{
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

    const base = "/tmp/scratchpad/imgclasstrain_main";
    std.Io.Dir.cwd().deleteTree(io, base) catch {};
    std.Io.Dir.cwd().createDirPath(io, base) catch {};

    const tcfg = train_mod.TrainConfig{ .max_steps = 100, .warmup_steps = 5 };
    const steps = train_mod.runImageClassificationTraining(&model, &images, &labels, &ivq, layout, tcfg, base, 0) catch {
        std.debug.print("training failed\n", .{});
        return;
    };
    std.debug.print("training steps: {d}\n", .{steps});

    const label_tokens = [_]u32{ 0, 1 };
    const acc = train_mod.evalImageClassification(&model, allocator, &images, &labels, &ivq, layout, &label_tokens) catch 0;
    std.debug.print("image classification accuracy after CPU training: {d:.4}\n", .{acc});

    for (images, 0..) |img, i| {
        const pred = train_mod.classifyImage(&model, allocator, img, &ivq, layout, &label_tokens) catch 0;
        std.debug.print("image {d}: predicted class token {d} (expected {d})\n", .{ i, pred, labels[i] });
    }
}
