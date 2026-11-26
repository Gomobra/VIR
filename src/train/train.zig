// Training loop skeleton (PRD 3.8).

const std = @import("std");

pub const TrainConfig = struct {
    lr: f32 = 1e-3,
    batch_tokens: usize = 1024,
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

test "Trainer steps" {
    var t = Trainer.init(.{ .max_steps = 3 });
    try std.testing.expect(!t.done());
    t.runStep();
    t.runStep();
    t.runStep();
    try std.testing.expect(t.done());
}
