// Neural network layers (PRD 3.2, 3.7). Minimal.

const std = @import("std");
const cpu = @import("../cpu/cpu.zig");

pub const Linear = struct {
    // W: [out, in], no bias
    w: []f32,
    in_dim: usize,
    out_dim: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, in_dim: usize, out_dim: usize) !Linear {
        const n = in_dim * out_dim;
        const w = try allocator.alloc(f32, n);
        // init small
        for (w, 0..) |*x, i| x.* = 0.02 * @as(f32, @floatFromInt(i % 7)) - 0.06;
        return .{ .w = w, .in_dim = in_dim, .out_dim = out_dim, .allocator = allocator };
    }

    pub fn deinit(self: *Linear) void {
        self.allocator.free(self.w);
    }

    // out[batch, out_dim] = in[batch, in_dim] @ W^T
    pub fn forward(self: *const Linear, out: []f32, in: []const f32, batch: usize) void {
        std.debug.assert(in.len == batch * self.in_dim);
        std.debug.assert(out.len == batch * self.out_dim);
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            const inrow = in[b * self.in_dim .. (b + 1) * self.in_dim];
            const outrow = out[b * self.out_dim .. (b + 1) * self.out_dim];
            // out[j] = sum_i in[i]*w[j*in+i]
            var j: usize = 0;
            while (j < self.out_dim) : (j += 1) {
                var acc: f32 = 0;
                var i: usize = 0;
                while (i < self.in_dim) : (i += 1) {
                    acc += inrow[i] * self.w[j * self.in_dim + i];
                }
                outrow[j] = acc;
            }
        }
        _ = cpu;
    }
};

pub const RMSNorm = struct {
    eps: f32 = 1e-6,
    pub fn forward(self: *const RMSNorm, out: []f32, x: []const f32) void {
        cpu.rmsnormF32(out, x, self.eps);
    }
};

test "Linear forward shape" {
    const allocator = std.testing.allocator;
    var lin = try Linear.init(allocator, 4, 3);
    defer lin.deinit();
    var in = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var out = [_]f32{ 0, 0, 0, 0, 0, 0 };
    lin.forward(&out, &in, 2);
    // just check no crash and finite
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}

test "RMSNorm forward" {
    var x = [_]f32{ 1, 2, 3, 4 };
    var o = [_]f32{ 0, 0, 0, 0 };
    const n = RMSNorm{};
    n.forward(&o, &x);
    var ss: f32 = 0;
    for (o) |v| ss += v * v;
    const rms = @sqrt(ss / 4.0);
    try std.testing.expect(@abs(rms - 1.0) < 1e-5);
}
