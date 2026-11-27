const std = @import("std");
const root = @import("root.zig");
const model_mod = @import("model/model.zig");
const train_mod = @import("train/train.zig");
const modality = @import("modality/modality.zig");
const config = @import("config/config.zig");

pub fn main() void {
    std.debug.print("vir {s}\n", .{root.version});
    const allocator = std.heap.page_allocator;

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
    std.fs.cwd().deleteTree(base) catch {};
    std.fs.cwd().makePath(base) catch {};

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
