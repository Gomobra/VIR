// vir — Model AI 1B dari scratch di Zig
// Library root. Satu modul per berkas besar (lihat PRD §6.2).

pub const version = "0.1.0";

// Core modules (diisi bertahap)
pub const tensor = @import("tensor/tensor.zig");
pub const cpu = @import("cpu/cpu.zig");
pub const autograd = @import("autograd/autograd.zig");
pub const optim = @import("optim/optim.zig");
pub const nn = @import("nn/nn.zig");
pub const tokenizer = @import("tokenizer/tokenizer.zig");
pub const model = @import("model/model.zig");
pub const data = @import("data/data.zig");
pub const train = @import("train/train.zig");
pub const eval = @import("eval/eval.zig");
pub const infer = @import("infer/infer.zig");

test "smoke" {
    const std = @import("std");
    try std.testing.expectEqualStrings("0.1.0", version);
}
