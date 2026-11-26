// Tensor core module (PRD 3.1). Implementasi bertahap.

const std = @import("std");

pub const DType = enum { f32, f16, i8, u8, i32, u16, u32, ternary2 };
pub const Device = enum { cpu, gpu };

pub const Tensor = struct {
    data: [*]u8,
    dtype: DType,
    device: Device,
    ndim: u8,
    shape: [5]usize,
    strides: [5]usize,
    offset: usize,
    owns: bool,
    allocator: ?std.mem.Allocator = null,

    pub fn numel(self: *const Tensor) usize {
        var n: usize = 1;
        var i: usize = 0;
        while (i < self.ndim) : (i += 1) n *= self.shape[i];
        return n;
    }

    pub fn byteSize(self: *const Tensor) usize {
        return self.numel() * dtypeSize(self.dtype);
    }

    pub fn view(self: *const Tensor, new_shape: []const usize) Tensor {
        var t = self.*;
        t.ndim = @intCast(new_shape.len);
        var i: usize = 0;
        while (i < new_shape.len) : (i += 1) t.shape[i] = new_shape[i];
        return t;
    }
};

pub fn dtypeSize(dt: DType) usize {
    return switch (dt) {
        .f32 => 4,
        .f16 => 2,
        .i8, .u8, .ternary2 => 1,
        .i32, .u32 => 4,
        .u16 => 2,
    };
}

pub fn f32ToF16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    const sign: u32 = (bits >> 16) & 0x8000;
    var exp: i32 = @intCast((bits >> 23) & 0xff);
    var mant: u32 = bits & 0x7fffff;
    if (exp == 255) {
        if (mant != 0) return @intCast(sign | 0x7fff);
        return @intCast(sign | 0x7c00);
    }
    exp -= 127;
    if (exp > 15) return @intCast(sign | 0x7c00);
    if (exp < -14) {
        if (exp < -24) return @intCast(sign);
        mant = (mant | 0x800000) >> @intCast(-exp - 1 + 13);
        return @intCast(sign | mant);
    }
    mant = (mant + 0x1000) >> 13;
    if (mant == 0x400) {
        mant = 0;
        exp += 1;
        if (exp > 15) return @intCast(sign | 0x7c00);
    }
    return @intCast(sign | (@as(u32, @intCast(exp + 15)) << 10) | (mant & 0x3ff));
}

pub fn f16ToF32(h: u16) f32 {
    const sign: u32 = @as(u32, h & 0x8000) << 16;
    const exp: u32 = (h >> 10) & 0x1f;
    const mant: u32 = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) return @bitCast(sign);
        var e: i32 = -14;
        var m: u32 = mant;
        while (m < 0x400) : (m <<= 1) e -= 1;
        m &= 0x3ff;
        const bits = sign | (@as(u32, @intCast(e + 127)) << 23) | (m << 13);
        return @bitCast(bits);
    }
    if (exp == 31) {
        return @bitCast(sign | 0x7f800000 | (mant << 13));
    }
    const bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    return @bitCast(bits);
}

test "dtypeSize" {
    try std.testing.expectEqual(@as(usize, 4), dtypeSize(.f32));
    try std.testing.expectEqual(@as(usize, 2), dtypeSize(.f16));
    try std.testing.expectEqual(@as(usize, 1), dtypeSize(.ternary2));
}

test "f16 roundtrip" {
    const vals = [_]f32{ 0.0, 1.0, -1.0, 0.5, 2.0, 3.14159, 65504.0 };
    for (vals) |v| {
        const h = f32ToF16(v);
        const r = f16ToF32(h);
        const rel = if (v == 0) @abs(r) else @abs(r - v) / @abs(v);
        try std.testing.expect(rel < 1e-2);
    }
}
