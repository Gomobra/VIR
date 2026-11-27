// SHA-256 (PRD 3.4/Tier 3). Self-contained implementation, no std.crypto dependency.

const std = @import("std");

const K = [_]u32{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

fn rotr(x: u32, n: u5) u32 {
    return (x >> n) | (x << (@as(u5, 32) - n));
}

fn ch(x: u32, y: u32, z: u32) u32 {
    return (x & y) ^ (~x & z);
}

fn maj(x: u32, y: u32, z: u32) u32 {
    return (x & y) ^ (x & z) ^ (y & z);
}

fn ep0(x: u32) u32 {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

fn ep1(x: u32) u32 {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

fn sig0(x: u32) u32 {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

fn sig1(x: u32) u32 {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

pub fn sha256(data: []const u8) [32]u8 {
    var H = [_]u32{
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    // Padding.
    const bit_len: u64 = @as(u64, data.len) * 8;
    const pad_len = (64 - ((data.len + 9) % 64)) % 64; // number of zero bytes between 0x80 and length
    const total_len = data.len + 1 + pad_len + 8;

    // Process in 64-byte blocks without allocating full padded message when possible.
    // We'll stream using an index into data/padding.
    var offset: usize = 0;
    while (offset < total_len) : (offset += 64) {
        var W: [64]u32 = undefined;
        var t: usize = 0;
        while (t < 16) : (t += 1) {
            const idx = offset + t * 4;
            var byte0: u8 = 0;
            var byte1: u8 = 0;
            var byte2: u8 = 0;
            var byte3: u8 = 0;
            // read 4 bytes from logical padded message
            var i: usize = 0;
            while (i < 4) : (i += 1) {
                const pos = idx + i;
                var b: u8 = 0;
                if (pos < data.len) {
                    b = data[pos];
                } else if (pos == data.len) {
                    b = 0x80;
                } else if (pos >= total_len - 8) {
                    // length bytes (big-endian)
                    const li = pos - (total_len - 8);
                    const shift: u3 = @intCast((7 - li) * 8);
                    b = @intCast((bit_len >> shift) & 0xff);
                } else {
                    b = 0;
                }
                switch (i) {
                    0 => byte0 = b,
                    1 => byte1 = b,
                    2 => byte2 = b,
                    3 => byte3 = b,
                    else => {},
                }
            }
            W[t] = (@as(u32, byte0) << 24) | (@as(u32, byte1) << 16) | (@as(u32, byte2) << 8) | @as(u32, byte3);
        }
        var t2: usize = 16;
        while (t2 < 64) : (t2 += 1) {
            W[t2] = sig1(W[t2 - 2]) +% W[t2 - 7] +% sig0(W[t2 - 15]) +% W[t2 - 16];
        }

        var a = H[0];
        var b = H[1];
        var c = H[2];
        var d = H[3];
        var e = H[4];
        var f = H[5];
        var g = H[6];
        var h = H[7];

        var r: usize = 0;
        while (r < 64) : (r += 1) {
            const T1 = h +% ep1(e) +% ch(e, f, g) +% K[r] +% W[r];
            const T2 = ep0(a) +% maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d +% T1;
            d = c;
            c = b;
            b = a;
            a = T1 +% T2;
        }

        H[0] +%= a;
        H[1] +%= b;
        H[2] +%= c;
        H[3] +%= d;
        H[4] +%= e;
        H[5] +%= f;
        H[6] +%= g;
        H[7] +%= h;
    }

    var out: [32]u8 = undefined;
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        const w = H[i];
        out[i * 4] = @intCast((w >> 24) & 0xff);
        out[i * 4 + 1] = @intCast((w >> 16) & 0xff);
        out[i * 4 + 2] = @intCast((w >> 8) & 0xff);
        out[i * 4 + 3] = @intCast(w & 0xff);
    }
    return out;
}

pub fn sha256Hex(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    const digest = sha256(data);
    const hex = try allocator.alloc(u8, 64);
    const chars = "0123456789abcdef";
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        hex[i * 2] = chars[digest[i] >> 4];
        hex[i * 2 + 1] = chars[digest[i] & 0x0f];
    }
    return hex;
}

test "sha256 empty" {
    const d = sha256("");
    const expected = [_]u8{
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
        0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
        0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
        0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
    };
    try std.testing.expectEqualSlices(u8, &expected, &d);
}

test "sha256 abc" {
    const d = sha256("abc");
    const expected = [_]u8{
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
    };
    try std.testing.expectEqualSlices(u8, &expected, &d);
}

test "sha256 long" {
    const input = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    const d = sha256(input);
    const expected = [_]u8{
        0x24, 0x8d, 0x6a, 0x61, 0xd2, 0x06, 0x38, 0xb8,
        0xe5, 0xc0, 0x26, 0x93, 0x0c, 0x3e, 0x60, 0x39,
        0xa3, 0x3c, 0xe4, 0x59, 0x64, 0xff, 0x21, 0x67,
        0xf6, 0xec, 0xed, 0xd4, 0x19, 0xdb, 0x06, 0xc1,
    };
    try std.testing.expectEqualSlices(u8, &expected, &d);
}
