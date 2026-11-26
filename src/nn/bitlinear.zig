// BitLinear (PRD 3.7, BitNet b1.58): ternary weights {-1,0,+1}, 8-bit activations, STE.

const std = @import("std");
const cpu = @import("../cpu/cpu.zig");

pub const BitLinear = struct {
    in_dim: usize,
    out_dim: usize,
    // master weights (f32)
    w: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, in_dim: usize, out_dim: usize) !BitLinear {
        const n = in_dim * out_dim;
        const w = try allocator.alloc(f32, n);
        for (w, 0..) |*x, i| x.* = 0.02 * @as(f32, @floatFromInt(i % 7)) - 0.06;
        return .{ .in_dim = in_dim, .out_dim = out_dim, .w = w, .allocator = allocator };
    }

    pub fn deinit(self: *BitLinear) void {
        self.allocator.free(self.w);
    }

    // ternarize weights: gamma = mean(|w|); w_t = roundclip(w/gamma, -1, 1)
    pub fn ternarize(self: *const BitLinear, out: []i8) void {
        std.debug.assert(out.len == self.w.len);
        var sum: f32 = 0;
        for (self.w) |v| sum += @abs(v);
        const gamma = sum / @as(f32, @floatFromInt(self.w.len));
        const inv = if (gamma > 0) 1.0 / gamma else 0;
        for (self.w, out) |v, *o| {
            const scaled = v * inv;
            const r = @round(scaled);
            const clamped = if (r > 1) @as(i8, 1) else if (r < -1) @as(i8, -1) else @as(i8, @intFromFloat(r));
            o.* = clamped;
        }
    }

    // forward: out = (x_quant (int8) @ w_ternary (i8)) * scale_x * gamma
    pub fn forward(self: *const BitLinear, out: []f32, x: []const f32, batch: usize) void {
        std.debug.assert(x.len == batch * self.in_dim);
        std.debug.assert(out.len == batch * self.out_dim);
        // quantize activations per token: absmax -> int8
        // ternarize weights
        const wt = self.allocator.alloc(i8, self.w.len) catch return;
        defer self.allocator.free(wt);
        self.ternarize(wt);
        // gamma
        var sum: f32 = 0;
        for (self.w) |v| sum += @abs(v);
        const gamma = sum / @as(f32, @floatFromInt(self.w.len));

        var b: usize = 0;
        while (b < batch) : (b += 1) {
            const xrow = x[b * self.in_dim .. (b + 1) * self.in_dim];
            // absmax scale for this token
            var amax: f32 = 0;
            for (xrow) |v| {
                const a = @abs(v);
                if (a > amax) amax = a;
            }
            const scale_x = if (amax > 0) 127.0 / amax else 0;
            const inv_scale_x = if (scale_x > 0) amax / 127.0 else 0;
            // quantize x to int8
            var xq = self.allocator.alloc(i8, self.in_dim) catch return;
            defer self.allocator.free(xq);
            for (xrow, xq) |v, *q| {
                const qv = @round(v * scale_x);
                q.* = if (qv > 127) 127 else if (qv < -127) -127 else @intFromFloat(qv);
            }
            // matmul: out[j] = sum_i xq[i] * wt[j*in+i]  (int), then scale
            var j: usize = 0;
            while (j < self.out_dim) : (j += 1) {
                var acc: i32 = 0;
                var i: usize = 0;
                while (i < self.in_dim) : (i += 1) {
                    acc += @as(i32, xq[i]) * @as(i32, wt[j * self.in_dim + i]);
                }
                out[b * self.out_dim + j] = @as(f32, @floatFromInt(acc)) * inv_scale_x * gamma;
            }
        }
    }
};

test "BitLinear forward finite" {
    const allocator = std.testing.allocator;
    var bl = try BitLinear.init(allocator, 4, 3);
    defer bl.deinit();
    var x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var out = [_]f32{ 0, 0, 0, 0, 0, 0 };
    bl.forward(&out, &x, 2);
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}

test "BitLinear ternarize values" {
    const allocator = std.testing.allocator;
    var bl = try BitLinear.init(allocator, 4, 2);
    defer bl.deinit();
    const wt = try allocator.alloc(i8, bl.w.len);
    defer allocator.free(wt);
    bl.ternarize(wt);
    for (wt) |v| try std.testing.expect(v >= -1 and v <= 1);
}
