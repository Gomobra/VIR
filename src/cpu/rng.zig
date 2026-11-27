// RNG (PRD 3.1): xoshiro256** with splitmix64 seeding and Box-Muller normals.

const std = @import("std");

pub const Rng = struct {
    s: [4]u64,

    pub fn init(seed: u64) Rng {
        var r: Rng = undefined;
        var z = seed;
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            z +%= 0x9E3779B97F4A7C15;
            var x = z;
            x = (x ^ (x >> 30)) *% 0xBF58476D1CE4E5B9;
            x = (x ^ (x >> 27)) *% 0x94D049BB133111B9;
            r.s[i] = x ^ (x >> 31);
        }
        // avoid all-zero state
        if (r.s[0] == 0 and r.s[1] == 0 and r.s[2] == 0 and r.s[3] == 0) {
            r.s[0] = 1;
        }
        return r;
    }

    pub fn next(self: *Rng) u64 {
        const result = std.math.rotl(u64, self.s[1] *% 5, 7) *% 9;
        const t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = std.math.rotl(u64, self.s[3], 45);
        return result;
    }

    pub fn nextF32(self: *Rng) f32 {
        const u = self.next() >> 8; // 56 bits
        return @as(f32, @floatFromInt(u)) / @as(f32, @floatFromInt(@as(u64, 1) << 56));
    }

    pub fn normal(self: *Rng) f32 {
        const a = @max(self.nextF32(), 1e-7);
        const b = self.nextF32();
        const r = @sqrt(-2.0 * @log(a));
        const theta = 2.0 * std.math.pi * b;
        return r * @cos(theta);
    }

    // Deterministic seed from a global seed and a tensor name.
    pub fn seedFromName(global_seed: u64, name: []const u8) u64 {
        var h: u64 = global_seed ^ 0xcbf29ce484222325;
        for (name) |c| {
            h ^= c;
            h *%= 0x100000001b3;
        }
        return h;
    }
};

test "Rng deterministic" {
    var r1 = Rng.init(42);
    var r2 = Rng.init(42);
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        try std.testing.expectEqual(r1.next(), r2.next());
    }
}

test "Rng normal finite" {
    var r = Rng.init(7);
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const v = r.normal();
        try std.testing.expect(std.math.isFinite(v));
    }
}

test "seedFromName deterministic" {
    const a = Rng.seedFromName(1, "weight");
    const b = Rng.seedFromName(1, "weight");
    try std.testing.expectEqual(a, b);
    try std.testing.expect(Rng.seedFromName(1, "bias") != a);
}
