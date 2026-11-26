// Data pipeline (PRD 3.6). Minimal shard + sampler.

const std = @import("std");

pub const ShardHeader = struct {
    magic: u32 = 0x5A544F4B, // "ZTOK"
    version: u32 = 1,
    dtype: u8 = 0, // 0=u16, 1=u32
    n_tokens: u64,
    vocab_size: u32,
};

pub const Sampler = struct {
    seed: u64,
    step: u64,
    rank: u32,
    micro: u32,

    pub fn index(self: Sampler) u64 {
        // deterministic: mix seed, step, rank, micro
        var x = self.seed ^ (self.step * 0x9e3779b97f4a7c15) ^ (@as(u64, self.rank) << 32) ^ self.micro;
        // splitmix64
        x +%= 0x9e3779b97f4a7c15;
        var z = x;
        z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) *% 0x94d049bb133111b9;
        return z ^ (z >> 31);
    }
};

test "Sampler deterministic" {
    const s1 = Sampler{ .seed = 42, .step = 10, .rank = 0, .micro = 0 };
    const s2 = Sampler{ .seed = 42, .step = 10, .rank = 0, .micro = 0 };
    try std.testing.expectEqual(s1.index(), s2.index());
}

// Quality filter (PRD 3.6): returns true if document passes.
pub const FilterConfig = struct {
    min_chars: usize = 50,
    max_symbol_ratio: f32 = 0.3,
};

pub fn passesFilter(cfg: FilterConfig, text: []const u8) bool {
    if (text.len < cfg.min_chars) return false;
    // symbol ratio: count non-alphanumeric (excluding space) / len
    var symbols: usize = 0;
    for (text) |c| {
        const is_alnum = (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or c == ' ' or c == '\n' or c == '\t';
        if (!is_alnum) symbols += 1;
    }
    const ratio = @as(f32, @floatFromInt(symbols)) / @as(f32, @floatFromInt(text.len));
    if (ratio > cfg.max_symbol_ratio) return false;
    return true;
}

test "passesFilter basic" {
    const cfg = FilterConfig{};
    try std.testing.expect(!passesFilter(cfg, "short"));
    const txt = "This is a normal sentence with enough characters to pass the filter.";
    try std.testing.expect(passesFilter(cfg, txt));
    // high symbol ratio
    try std.testing.expect(!passesFilter(cfg, "!!!!@@@@####$$$$%%%%^^^^&&&&****((((()))))"));
}

// --- Dedup (PRD 3.6) ---

// 64-bit hash (FNV-1a).
pub fn hash64(s: []const u8) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (s) |c| {
        h ^= c;
        h *%= 0x100000001b3;
    }
    return h;
}

// Exact dedup hash: hash of normalized (lowercased, trimmed) text.
pub fn exactHash(text: []const u8) u64 {
    // simple normalize: lowercase ASCII
    var buf: [4096]u8 = undefined;
    const n = @min(text.len, buf.len);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var c = text[i];
        if (c >= 'A' and c <= 'Z') c += 32;
        buf[i] = c;
    }
    return hash64(buf[0..n]);
}

// MinHash: compute signature of H hashes over shingles.
pub const MINHASH_H: usize = 8;

pub fn shingleHashes(allocator: std.mem.Allocator, text: []const u8, k: usize) ![]u64 {
    // word shingles of size k (by bytes, simplified)
    var set = std.AutoHashMap(u64, void).init(allocator);
    defer set.deinit();
    if (text.len < k) {
        // single shingle
        try set.put(hash64(text), {});
        const out = try allocator.alloc(u64, set.count());
        var it = set.keyIterator();
        var idx: usize = 0;
        while (it.next()) |key| {
            out[idx] = key.*;
            idx += 1;
        }
        return out;
    }
    var i: usize = 0;
    while (i + k <= text.len) : (i += 1) {
        try set.put(hash64(text[i .. i + k]), {});
    }
    const out = try allocator.alloc(u64, set.count());
    var it = set.keyIterator();
    var idx: usize = 0;
    while (it.next()) |key| {
        out[idx] = key.*;
        idx += 1;
    }
    return out;
}

pub fn minHashSignature(allocator: std.mem.Allocator, shingles: []const u64, h: usize) ![]u64 {
    const sig = try allocator.alloc(u64, h);
    var hi: usize = 0;
    while (hi < h) : (hi += 1) {
        var mn: u64 = std.math.maxInt(u64);
        for (shingles) |sh| {
            // combine shingle with hash index
            var x = sh ^ (@as(u64, hi) *% 0x9e3779b97f4a7c15);
            x +%= 0x9e3779b97f4a7c15;
            var z = x;
            z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) *% 0x94d049bb133111b9;
            const hh = z ^ (z >> 31);
            if (hh < mn) mn = hh;
        }
        if (shingles.len == 0) mn = 0;
        sig[hi] = mn;
    }
    return sig;
}

// Estimated Jaccard similarity from two minhash signatures.
pub fn jaccardEstimate(sigA: []const u64, sigB: []const u64) f32 {
    if (sigA.len == 0 or sigA.len != sigB.len) return 0;
    var match: usize = 0;
    for (sigA, sigB) |a, b| if (a == b) {
        match += 1;
    };
    return @as(f32, @floatFromInt(match)) / @as(f32, @floatFromInt(sigA.len));
}

test "hash64 deterministic" {
    try std.testing.expectEqual(hash64("abc"), hash64("abc"));
    try std.testing.expect(hash64("abc") != hash64("def"));
}

test "exactHash case-insensitive" {
    try std.testing.expectEqual(exactHash("Hello"), exactHash("hello"));
}

test "minHash identical docs high similarity" {
    const allocator = std.testing.allocator;
    const text = "the quick brown fox jumps over the lazy dog";
    const shA = try shingleHashes(allocator, text, 5);
    defer allocator.free(shA);
    const shB = try shingleHashes(allocator, text, 5);
    defer allocator.free(shB);
    const sigA = try minHashSignature(allocator, shA, MINHASH_H);
    defer allocator.free(sigA);
    const sigB = try minHashSignature(allocator, shB, MINHASH_H);
    defer allocator.free(sigB);
    const sim = jaccardEstimate(sigA, sigB);
    try std.testing.expect(sim > 0.9);
}

test "minHash different docs lower similarity" {
    const allocator = std.testing.allocator;
    const t1 = "the quick brown fox jumps over the lazy dog";
    const t2 = "completely different text about quantum physics and math";
    const shA = try shingleHashes(allocator, t1, 5);
    defer allocator.free(shA);
    const shB = try shingleHashes(allocator, t2, 5);
    defer allocator.free(shB);
    const sigA = try minHashSignature(allocator, shA, MINHASH_H);
    defer allocator.free(sigA);
    const sigB = try minHashSignature(allocator, shB, MINHASH_H);
    defer allocator.free(sigB);
    const sim = jaccardEstimate(sigA, sigB);
    try std.testing.expect(sim < 0.5);
}
