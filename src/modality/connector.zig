// Connector: project modality features into LM embedding space (PRD 6.3).

const std = @import("std");

pub const Connector = struct {
    in_dim: usize,
    out_dim: usize,
    w: []f32, // out_dim x in_dim
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, in_dim: usize, out_dim: usize) !Connector {
        const n = in_dim * out_dim;
        const w = try allocator.alloc(f32, n);
        for (w, 0..) |*x, i| x.* = 0.01 * @as(f32, @floatFromInt(i % 7)) - 0.03;
        return .{ .in_dim = in_dim, .out_dim = out_dim, .w = w, .allocator = allocator };
    }

    pub fn deinit(self: *Connector) void {
        self.allocator.free(self.w);
    }

    // out[batch, out_dim] = in[batch, in_dim] @ W^T
    pub fn forward(self: *const Connector, out: []f32, in: []const f32, batch: usize) void {
        std.debug.assert(in.len == batch * self.in_dim);
        std.debug.assert(out.len == batch * self.out_dim);
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            const inrow = in[b * self.in_dim .. (b + 1) * self.in_dim];
            var j: usize = 0;
            while (j < self.out_dim) : (j += 1) {
                var acc: f32 = 0;
                var i: usize = 0;
                while (i < self.in_dim) : (i += 1) acc += inrow[i] * self.w[j * self.in_dim + i];
                out[b * self.out_dim + j] = acc;
            }
        }
    }

    // Backward: dW += dOut^T @ in ; dIn += dOut @ W
    pub fn backward(self: *const Connector, dW: []f32, dIn: []f32, in: []const f32, dOut: []const f32, batch: usize) void {
        std.debug.assert(dW.len == self.in_dim * self.out_dim);
        std.debug.assert(dIn.len == in.len);
        std.debug.assert(dOut.len == batch * self.out_dim);
        // dW
        var j: usize = 0;
        while (j < self.out_dim) : (j += 1) {
            var i: usize = 0;
            while (i < self.in_dim) : (i += 1) {
                var acc: f32 = 0;
                var b: usize = 0;
                while (b < batch) : (b += 1) {
                    acc += in[b * self.in_dim + i] * dOut[b * self.out_dim + j];
                }
                dW[j * self.in_dim + i] += acc;
            }
        }
        // dIn
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            var i: usize = 0;
            while (i < self.in_dim) : (i += 1) {
                var acc: f32 = 0;
                var jj: usize = 0;
                while (jj < self.out_dim) : (jj += 1) {
                    acc += dOut[b * self.out_dim + jj] * self.w[jj * self.in_dim + i];
                }
                dIn[b * self.in_dim + i] += acc;
            }
        }
    }
};

test "Connector backward finite" {
    const allocator = std.testing.allocator;
    var c = try Connector.init(allocator, 4, 3);
    defer c.deinit();
    var in = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var out = [_]f32{ 0, 0, 0, 0, 0, 0 };
    c.forward(&out, &in, 2);
    const dW = try allocator.alloc(f32, 4 * 3);
    defer allocator.free(dW);
    @memset(dW, 0);
    var dIn = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var dOut = [_]f32{ 1, 0, 0, 0, 1, 0 };
    c.backward(dW, &dIn, &in, &dOut, 2);
    for (dW) |v| try std.testing.expect(std.math.isFinite(v));
    for (dIn) |v| try std.testing.expect(std.math.isFinite(v));
}

test "Connector forward shape" {
    const allocator = std.testing.allocator;
    var c = try Connector.init(allocator, 4, 3);
    defer c.deinit();
    var in = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var out = [_]f32{ 0, 0, 0, 0, 0, 0 };
    c.forward(&out, &in, 2);
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}
