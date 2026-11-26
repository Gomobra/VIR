// Config (PRD 5.9). RunConfig, ModelConfig, TrainConfig.

const std = @import("std");

pub const AttnConfig = struct {
    n_heads: usize = 4,
    n_kv_heads: usize = 1,
    head_dim: usize = 64,
    rope_theta: f32 = 10000.0,
};

pub const GdnConfig = struct {
    n_key_heads: usize = 2,
    n_value_heads: usize = 4,
    key_dim: usize = 32,
    value_dim: usize = 64,
    conv_kernel: usize = 4,
    chunk_size: usize = 64,
};

pub const MlpConfig = struct {
    hidden: usize = 384,
};

pub const EngramConfig = struct {
    enabled: bool = false,
    layers: []const usize = &.{},
    orders: []const u8 = &.{ 2, 3 },
    n_heads: usize = 8,
    table_size: usize = 1 << 18,
    dim_per_head: usize = 32,
};

pub const LoopConfig = struct {
    enabled: bool = false,
    repeats: usize = 1,
    adaptive_exit: bool = false,
};

pub const BitConfig = struct {
    enabled: bool = false,
    act_bits: usize = 8,
};

pub const MtpConfig = struct {
    enabled: bool = false,
    loss_weight: f32 = 0.3,
};

pub const ModelConfig = struct {
    d_model: usize = 128,
    n_groups: usize = 1,
    vocab_size: usize = 2048,
    seq_len: usize = 256,
    attn: AttnConfig = .{},
    gdn: GdnConfig = .{},
    mlp: MlpConfig = .{},
    engram: EngramConfig = .{},
    loop: LoopConfig = .{},
    bit: BitConfig = .{},
    mtp: MtpConfig = .{},

    pub fn paramCount(self: *const ModelConfig) usize {
        // approximate: embedding + groups*(3*gdn + 1*attn) + mlp
        const embed = self.vocab_size * self.d_model;
        const gdn_params = 3 * (self.gdn.n_key_heads * self.gdn.key_dim * self.d_model +
            self.gdn.n_value_heads * self.gdn.value_dim * self.d_model);
        const attn_params = self.attn.n_heads * self.attn.head_dim * self.d_model +
            self.attn.n_kv_heads * self.attn.head_dim * self.d_model * 2;
        const mlp_params = self.d_model * self.mlp.hidden * 2;
        const group_params = gdn_params + attn_params + mlp_params;
        const total_groups = self.n_groups;
        return embed + total_groups * group_params + self.d_model; // + final norm
    }
};

pub const TrainConfig = struct {
    batch_tokens: usize = 1024,
    micro_batch: usize = 4,
    lr_muon: f32 = 0.02,
    lr_adamw: f32 = 3e-4,
    weight_decay: f32 = 0.1,
    warmup_steps: usize = 10,
    decay_fraction: f32 = 0.2,
    tst_fraction: f32 = 0.3,
    tst_bag: usize = 4,
    clip_norm: f32 = 1.0,
};

pub const RunConfig = struct {
    run_id: []const u8 = "run-0",
    seed: u64 = 0,
    zig_version: []const u8 = "0.17.1",
    model: ModelConfig = .{},
    train: TrainConfig = .{},
    mix_path: []const u8 = "configs/mix.yaml",
    tokenizer_hash: []const u8 = "",
};

test "ModelConfig paramCount positive" {
    const cfg = ModelConfig{};
    try std.testing.expect(cfg.paramCount() > 0);
}

test "RunConfig defaults" {
    const cfg = RunConfig{};
    try std.testing.expectEqualStrings("run-0", cfg.run_id);
    try std.testing.expect(cfg.model.seq_len > 0);
}

// Parse a simple key=value config (one per line). Returns number of keys parsed.
pub fn parseKeyValue(allocator: std.mem.Allocator, text: []const u8, out: *std.StringHashMap([]const u8)) !usize {
    _ = allocator;
    var count: usize = 0;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;
        if (trimmed[0] == '#') continue;
        const eq = std.mem.indexOfScalar(u8, trimmed, '=') orelse continue;
        const key = std.mem.trim(u8, trimmed[0..eq], " \t");
        const val = std.mem.trim(u8, trimmed[eq + 1 ..], " \t");
        if (key.len == 0) continue;
        try out.put(key, val);
        count += 1;
    }
    return count;
}

test "parseKeyValue" {
    const allocator = std.testing.allocator;
    var map = std.StringHashMap([]const u8).init(allocator);
    defer map.deinit();
    const text = "lr=0.01\nbatch=4\n# comment\n\nhidden=128";
    const n = try parseKeyValue(allocator, text, &map);
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqualStrings("0.01", map.get("lr").?);
    try std.testing.expectEqualStrings("128", map.get("hidden").?);
}
