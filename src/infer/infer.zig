// Inference (PRD 3.12). Forward-only sampling helpers.

const std = @import("std");

pub fn argmax(logits: []const f32) usize {
    if (logits.len == 0) return 0;
    var best: usize = 0;
    var bestv: f32 = logits[0];
    for (logits, 0..) |v, i| {
        if (v > bestv) {
            bestv = v;
            best = i;
        }
    }
    return best;
}

pub fn temperatureScale(logits: []f32, t: f32) void {
    if (t <= 0) return;
    for (logits) |*v| v.* /= t;
}

test "argmax" {
    const l = [_]f32{ 0.1, 0.5, 0.2 };
    try std.testing.expectEqual(@as(usize, 1), argmax(&l));
}

test "temperatureScale" {
    var l = [_]f32{ 2.0, 4.0 };
    temperatureScale(&l, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), l[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), l[1], 1e-6);
}

// Top-k: keep only top k logits, set rest to -inf.
pub fn topK(logits: []f32, k: usize) void {
    if (k >= logits.len) return;
    // find k-th largest threshold
    var sorted = std.heap.page_allocator.alloc(f32, logits.len) catch return;
    defer std.heap.page_allocator.free(sorted);
    @memcpy(sorted, logits);
    // simple selection sort for threshold
    var i: usize = 0;
    while (i < k) : (i += 1) {
        var maxidx = i;
        var j = i + 1;
        while (j < sorted.len) : (j += 1) if (sorted[j] > sorted[maxidx]) {
            maxidx = j;
        };
        const tmp = sorted[i];
        sorted[i] = sorted[maxidx];
        sorted[maxidx] = tmp;
    }
    const thresh = sorted[k - 1];
    for (logits) |*v| {
        if (v.* < thresh) v.* = -std.math.inf(f32);
    }
}

// Top-p (nucleus): keep smallest set with cumulative prob >= p.
pub fn topP(logits: []f32, p: f32) void {
    // softmax
    var mx: f32 = logits[0];
    for (logits) |v| if (v > mx) {
        mx = v;
    };
    var sum: f32 = 0;
    for (logits) |v| sum += @exp(v - mx);
    // sort indices by prob desc
    var idx = std.heap.page_allocator.alloc(usize, logits.len) catch return;
    defer std.heap.page_allocator.free(idx);
    for (idx, 0..) |*x, i| x.* = i;
    // insertion sort by prob desc
    var i: usize = 1;
    while (i < idx.len) : (i += 1) {
        const key = idx[i];
        var j = i;
        while (j > 0) {
            const pj = @exp(logits[idx[j - 1]] - mx) / sum;
            const pk = @exp(logits[key] - mx) / sum;
            if (pj >= pk) break;
            idx[j] = idx[j - 1];
            j -= 1;
        }
        idx[j] = key;
    }
    // cumulative
    var cum: f32 = 0;
    var cutoff: usize = logits.len;
    for (idx, 0..) |id, rank| {
        cum += @exp(logits[id] - mx) / sum;
        if (cum >= p) {
            cutoff = rank + 1;
            break;
        }
    }
    // mask out beyond cutoff
    var mask = std.heap.page_allocator.alloc(bool, logits.len) catch return;
    defer std.heap.page_allocator.free(mask);
    @memset(mask, false);
    var r: usize = 0;
    while (r < cutoff) : (r += 1) mask[idx[r]] = true;
    for (logits, 0..) |*v, ii| {
        if (!mask[ii]) v.* = -std.math.inf(f32);
    }
}

// Repetition penalty: divide logits of tokens that appeared by penalty.
pub fn repetitionPenalty(logits: []f32, appeared: []const u32, penalty: f32) void {
    for (appeared) |tok| {
        if (tok < logits.len) {
            if (logits[tok] > 0) logits[tok] /= penalty else logits[tok] *= penalty;
        }
    }
}

test "topK keeps top" {
    var l = [_]f32{ 1, 5, 2, 4, 3 };
    topK(&l, 2);
    // top 2 are 5 and 4; rest -inf
    try std.testing.expect(l[1] == 5);
    try std.testing.expect(l[3] == 4);
    try std.testing.expect(std.math.isInf(l[0]));
}

test "topP finite" {
    var l = [_]f32{ 1, 2, 3, 4 };
    topP(&l, 0.9);
    // at least one finite
    var finite: usize = 0;
    for (l) |v| if (std.math.isFinite(v)) {
        finite += 1;
    };
    try std.testing.expect(finite >= 1);
}

test "repetitionPenalty" {
    var l = [_]f32{ 2.0, 3.0 };
    const appeared = [_]u32{0};
    repetitionPenalty(&l, &appeared, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), l[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), l[1], 1e-6);
}

// Log-softmax in place.
pub fn logSoftmax(logits: []f32) void {
    if (logits.len == 0) return;
    var mx: f32 = logits[0];
    for (logits) |v| if (v > mx) {
        mx = v;
    };
    var sum: f32 = 0;
    for (logits) |v| sum += @exp(v - mx);
    const lse = mx + @log(sum);
    for (logits) |*v| v.* -= lse;
}

// Softmax in place.
pub fn softmax(logits: []f32) void {
    if (logits.len == 0) return;
    var mx: f32 = logits[0];
    for (logits) |v| if (v > mx) {
        mx = v;
    };
    var sum: f32 = 0;
    for (logits) |*v| {
        v.* = @exp(v.* - mx);
        sum += v.*;
    }
    for (logits) |*v| v.* /= sum;
}

// Sample from categorical distribution given probs and uniform u in [0,1).
pub fn sampleCategorical(probs: []const f32, u: f32) usize {
    var cum: f32 = 0;
    for (probs, 0..) |p, i| {
        cum += p;
        if (u < cum) return i;
    }
    return probs.len - 1;
}

test "logSoftmax sums to 1" {
    var l = [_]f32{ 1.0, 2.0, 3.0 };
    logSoftmax(&l);
    var sum: f32 = 0;
    for (l) |v| sum += @exp(v);
    try std.testing.expect(@abs(sum - 1.0) < 1e-5);
}

test "softmax" {
    var l = [_]f32{ 0, 0 };
    softmax(&l);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), l[0], 1e-6);
}

test "sampleCategorical" {
    const probs = [_]f32{ 0.5, 0.5 };
    try std.testing.expectEqual(@as(usize, 0), sampleCategorical(&probs, 0.25));
    try std.testing.expectEqual(@as(usize, 1), sampleCategorical(&probs, 0.75));
}

// KV cache for attention (PRD 3.12).
pub const KVCache = struct {
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    // k[layer][pos] = slice of n_kv_heads*head_dim
    k: []f32,
    v: []f32,
    seq_len: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize) !KVCache {
        const per = n_kv_heads * head_dim;
        const total = n_layers * max_seq * per;
        const k = try allocator.alloc(f32, total);
        const v = try allocator.alloc(f32, total);
        @memset(k, 0);
        @memset(v, 0);
        return .{ .n_layers = n_layers, .n_kv_heads = n_kv_heads, .head_dim = head_dim, .max_seq = max_seq, .k = k, .v = v, .seq_len = 0, .allocator = allocator };
    }

    pub fn deinit(self: *KVCache) void {
        self.allocator.free(self.k);
        self.allocator.free(self.v);
    }

    fn offset(self: *const KVCache, layer: usize, pos: usize) usize {
        const per = self.n_kv_heads * self.head_dim;
        return (layer * self.max_seq + pos) * per;
    }

    pub fn append(self: *KVCache, layer: usize, kvec: []const f32, vvec: []const f32) void {
        const per = self.n_kv_heads * self.head_dim;
        std.debug.assert(kvec.len == per and vvec.len == per);
        const off = self.offset(layer, self.seq_len);
        @memcpy(self.k[off .. off + per], kvec);
        @memcpy(self.v[off .. off + per], vvec);
    }

    pub fn advance(self: *KVCache) void {
        self.seq_len += 1;
    }

    pub fn reset(self: *KVCache) void {
        self.seq_len = 0;
    }

    pub fn getK(self: *const KVCache, layer: usize, pos: usize) []const f32 {
        const per = self.n_kv_heads * self.head_dim;
        const off = self.offset(layer, pos);
        return self.k[off .. off + per];
    }

    pub fn getV(self: *const KVCache, layer: usize, pos: usize) []const f32 {
        const per = self.n_kv_heads * self.head_dim;
        const off = self.offset(layer, pos);
        return self.v[off .. off + per];
    }
};

test "KVCache append/get" {
    const allocator = std.testing.allocator;
    var cache = try KVCache.init(allocator, 1, 1, 2, 4);
    defer cache.deinit();
    const k = [_]f32{ 1, 2 };
    const v = [_]f32{ 3, 4 };
    cache.append(0, &k, &v);
    const gk = cache.getK(0, 0);
    try std.testing.expectEqualSlices(f32, &k, gk);
    cache.advance();
    try std.testing.expectEqual(@as(usize, 1), cache.seq_len);
}

// GDN recurrent state cache (PRD 3.10): state per layer/head is value_dim x key_dim.
pub const GDNStateCache = struct {
    n_layers: usize,
    n_heads: usize,
    key_dim: usize,
    value_dim: usize,
    state: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n_layers: usize, n_heads: usize, key_dim: usize, value_dim: usize) !GDNStateCache {
        const total = n_layers * n_heads * value_dim * key_dim;
        const state = try allocator.alloc(f32, total);
        @memset(state, 0);
        return .{ .n_layers = n_layers, .n_heads = n_heads, .key_dim = key_dim, .value_dim = value_dim, .state = state, .allocator = allocator };
    }

    pub fn deinit(self: *GDNStateCache) void {
        self.allocator.free(self.state);
    }

    fn offset(self: *const GDNStateCache, layer: usize, head: usize) usize {
        return (layer * self.n_heads + head) * self.value_dim * self.key_dim;
    }

    pub fn getState(self: *const GDNStateCache, layer: usize, head: usize) []f32 {
        const off = self.offset(layer, head);
        return self.state[off .. off + self.value_dim * self.key_dim];
    }

    pub fn setState(self: *GDNStateCache, layer: usize, head: usize, src: []const f32) void {
        const off = self.offset(layer, head);
        @memcpy(self.state[off .. off + self.value_dim * self.key_dim], src);
    }

    pub fn reset(self: *GDNStateCache) void {
        @memset(self.state, 0);
    }
};

test "GDNStateCache basic" {
    const allocator = std.testing.allocator;
    var cache = try GDNStateCache.init(allocator, 1, 2, 2, 2);
    defer cache.deinit();
    const s = cache.getState(0, 1);
    try std.testing.expectEqual(@as(usize, 4), s.len);
    var v = [_]f32{ 1, 2, 3, 4 };
    cache.setState(0, 1, &v);
    const g = cache.getState(0, 1);
    try std.testing.expectEqualSlices(f32, &v, g);
    cache.reset();
    for (cache.getState(0, 1)) |x| try std.testing.expectEqual(@as(f32, 0), x);
}

// KV cache for attention (PRD 3.12).
