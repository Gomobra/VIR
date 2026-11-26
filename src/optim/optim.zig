// Optimizer (PRD 3.2): AdamW.

const std = @import("std");

pub const AdamW = struct {
    lr: f32,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    wd: f32 = 0.01,
    t: usize = 0,
    m: []f32,
    v: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize, lr: f32) !AdamW {
        const m = try allocator.alloc(f32, n);
        const v = try allocator.alloc(f32, n);
        @memset(m, 0);
        @memset(v, 0);
        return .{ .lr = lr, .m = m, .v = v, .allocator = allocator };
    }

    pub fn deinit(self: *AdamW) void {
        self.allocator.free(self.m);
        self.allocator.free(self.v);
    }

    // in-place update of params using grads
    pub fn step(self: *AdamW, params: []f32, grads: []const f32) void {
        std.debug.assert(params.len == grads.len and grads.len == self.m.len);
        self.t += 1;
        const b1 = self.beta1;
        const b2 = self.beta2;
        const lr = self.lr;
        const wd = self.wd;
        const eps = self.eps;
        const bc1 = 1.0 - std.math.pow(f32, b1, @floatFromInt(self.t));
        const bc2 = 1.0 - std.math.pow(f32, b2, @floatFromInt(self.t));
        for (params, grads, self.m, self.v) |*p, g, *mi, *vi| {
            mi.* = b1 * mi.* + (1 - b1) * g;
            vi.* = b2 * vi.* + (1 - b2) * g * g;
            const mhat = mi.* / bc1;
            const vhat = vi.* / bc2;
            p.* = p.* - lr * (mhat / (@sqrt(vhat) + eps) + wd * p.*);
        }
    }
};

test "AdamW step reduces param toward gradient" {
    const allocator = std.testing.allocator;
    var opt = try AdamW.init(allocator, 1, 0.1);
    defer opt.deinit();
    var params = [_]f32{ 5.0 };
    const grads = [_]f32{ 1.0 };
    opt.step(&params, &grads);
    try std.testing.expect(params[0] < 5.0);
}
