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
