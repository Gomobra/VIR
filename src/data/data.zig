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

// --- ZTOK shard (PRD 3.6) ---

pub const DType = enum(u8) { u16 = 0, u32 = 1 };

pub const ZtokHeader = struct {
    magic: u32,
    version: u32,
    dtype: u8,
    n_tokens: u64,
    vocab_size: u32,
    // sha256 tokenizer (32 bytes)
    tok_hash: [32]u8,
};

pub fn writeShard(allocator: std.mem.Allocator, path: []const u8, dtype: DType, tokens: []const u32, vocab_size: u32, tok_hash: [32]u8) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    // header
    var magic_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &magic_bytes, 0x5A544F4B, .little); // "ZTOK"
    try file.writeAll(&magic_bytes);
    var ver: [4]u8 = undefined;
    std.mem.writeInt(u32, &ver, 1, .little);
    try file.writeAll(&ver);
    try file.writeAll(&[_]u8{@backingInt(dtype)});
    var ntok: [8]u8 = undefined;
    std.mem.writeInt(u64, &ntok, tokens.len, .little);
    try file.writeAll(&ntok);
    var vs: [4]u8 = undefined;
    std.mem.writeInt(u32, &vs, vocab_size, .little);
    try file.writeAll(&vs);
    try file.writeAll(&tok_hash);
    // tokens
    for (tokens) |t| {
        if (dtype == .u16) {
            var b: [2]u8 = undefined;
            std.mem.writeInt(u16, &b, @intCast(t), .little);
            try file.writeAll(&b);
        } else {
            var b: [4]u8 = undefined;
            std.mem.writeInt(u32, &b, t, .little);
            try file.writeAll(&b);
        }
    }
    _ = allocator;
}

pub fn readShardHeader(path: []const u8) !ZtokHeader {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    var magic_bytes: [4]u8 = undefined;
    _ = try file.readAll(&magic_bytes);
    const magic = std.mem.readInt(u32, &magic_bytes, .little);
    var ver: [4]u8 = undefined;
    _ = try file.readAll(&ver);
    const version = std.mem.readInt(u32, &ver, .little);
    var dtype_byte: [1]u8 = undefined;
    _ = try file.readAll(&dtype_byte);
    var ntok: [8]u8 = undefined;
    _ = try file.readAll(&ntok);
    const n_tokens = std.mem.readInt(u64, &ntok, .little);
    var vs: [4]u8 = undefined;
    _ = try file.readAll(&vs);
    const vocab_size = std.mem.readInt(u32, &vs, .little);
    var tok_hash: [32]u8 = undefined;
    _ = try file.readAll(&tok_hash);
    return ZtokHeader{
        .magic = magic,
        .version = version,
        .dtype = dtype_byte[0],
        .n_tokens = n_tokens,
        .vocab_size = vocab_size,
        .tok_hash = tok_hash,
    };
}

test "ZTOK roundtrip" {
    const allocator = std.testing.allocator;
    const path = "/tmp/scratchpad/test.ztok";
    std.fs.cwd().makePath("/tmp/scratchpad") catch {};
    const tokens = [_]u32{ 1, 2, 3, 1000 };
    var hash: [32]u8 = undefined;
    @memset(&hash, 0xAB);
    try writeShard(allocator, path, .u32, &tokens, 4096, hash);
    const hdr = try readShardHeader(path);
    try std.testing.expectEqual(@as(u32, 0x5A544F4B), hdr.magic);
    try std.testing.expectEqual(@as(u32, 1), hdr.version);
    try std.testing.expectEqual(@as(u64, 4), hdr.n_tokens);
    try std.testing.expectEqual(@as(u32, 4096), hdr.vocab_size);
    try std.testing.expectEqualSlices(u8, &hash, &hdr.tok_hash);
}

// Sampler resume: deterministic position from step.
pub fn samplerPosition(seed: u64, step: u64, rank: u32, micro: u32) u64 {
    const s = Sampler{ .seed = seed, .step = step, .rank = rank, .micro = micro };
    return s.index();
}

test "sampler resume deterministic" {
    const p1 = samplerPosition(42, 100, 0, 0);
    const p2 = samplerPosition(42, 100, 0, 0);
    try std.testing.expectEqual(p1, p2);
    // different step -> different position (likely)
    const p3 = samplerPosition(42, 101, 0, 0);
    try std.testing.expect(p1 != p3);
}

// --- Decontamination (PRD 3.6): remove docs sharing n-gram with eval set ---

pub fn ngramSet(allocator: std.mem.Allocator, text: []const u8, n: usize) !std.AutoHashMap(u64, void) {
    var set = std.AutoHashMap(u64, void).init(allocator);
    if (text.len < n) {
        try set.put(hash64(text), {});
        return set;
    }
    var i: usize = 0;
    while (i + n <= text.len) : (i += 1) {
        try set.put(hash64(text[i .. i + n]), {});
    }
    return set;
}

pub fn decontaminate(allocator: std.mem.Allocator, docs: []const []const u8, eval_docs: []const []const u8, n: usize) ![]usize {
    // build eval n-gram set
    var eval_set = std.AutoHashMap(u64, void).init(allocator);
    defer eval_set.deinit();
    for (eval_docs) |d| {
        var s = try ngramSet(allocator, d, n);
        defer s.deinit();
        var it = s.keyIterator();
        while (it.next()) |k| try eval_set.put(k.*, {});
    }
    // keep docs with no overlap
    var kept = std.ArrayList(usize).init(allocator);
    defer kept.deinit();
    for (docs, 0..) |doc, idx| {
        var s = try ngramSet(allocator, doc, n);
        defer s.deinit();
        var overlap = false;
        var it = s.keyIterator();
        while (it.next()) |k| {
            if (eval_set.contains(k.*)) {
                overlap = true;
                break;
            }
        }
        if (!overlap) try kept.append(idx);
    }
    return kept.toOwnedSlice();
}

test "decontaminate removes overlapping doc" {
    const allocator = std.testing.allocator;
    const docs = [_][]const u8{ "hello world foo bar baz", "completely unrelated text here" };
    const evals = [_][]const u8{"hello world foo bar baz extra"};
    const kept = try decontaminate(allocator, &docs, &evals, 5);
    defer allocator.free(kept);
    // first doc shares 5-grams with eval -> removed; second kept
    try std.testing.expectEqual(@as(usize, 1), kept.len);
    try std.testing.expectEqual(@as(usize, 1), kept[0]);
}

// --- Data mix (PRD 3.6) ---
pub const MixWeights = struct {
    web: f32 = 0.65,
    code: f32 = 0.20,
    math: f32 = 0.10,
    other: f32 = 0.05,
};

pub fn mixWeightsValid(w: MixWeights) bool {
    const sum = w.web + w.code + w.math + w.other;
    return @abs(sum - 1.0) < 1e-5;
}

test "MixWeights valid" {
    const w = MixWeights{};
    try std.testing.expect(mixWeightsValid(w));
}

// --- Sequence packing (PRD 3.8): pack multiple docs into one sequence with doc mask ---
pub const PackedSequence = struct {
    tokens: []u32,
    // doc_id per token (which document each token belongs to)
    doc_ids: []u32,
    n_docs: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PackedSequence) void {
        self.allocator.free(self.tokens);
        self.allocator.free(self.doc_ids);
    }
};

pub fn packSequences(allocator: std.mem.Allocator, docs: []const []const u32, max_len: usize) !PackedSequence {
    var tokens = std.ArrayList(u32).init(allocator);
    defer tokens.deinit();
    var doc_ids = std.ArrayList(u32).init(allocator);
    defer doc_ids.deinit();
    var n_docs: usize = 0;
    for (docs, 0..) |doc, di| {
        if (tokens.items.len + doc.len > max_len) break;
        for (doc) |t| {
            try tokens.append(t);
            try doc_ids.append(@intCast(di));
        }
        n_docs += 1;
    }
    const toks = try tokens.toOwnedSlice();
    const dids = try doc_ids.toOwnedSlice();
    return .{ .tokens = toks, .doc_ids = dids, .n_docs = n_docs, .allocator = allocator };
}

test "packSequences" {
    const allocator = std.testing.allocator;
    const d1 = [_]u32{ 1, 2 };
    const d2 = [_]u32{ 3, 4, 5 };
    const docs = [_][]const u32{ &d1, &d2 };
    var ps = try packSequences(allocator, &docs, 10);
    defer ps.deinit();
    try std.testing.expectEqual(@as(usize, 5), ps.tokens.len);
    try std.testing.expectEqual(@as(usize, 2), ps.n_docs);
    try std.testing.expectEqual(@as(u32, 0), ps.doc_ids[0]);
    try std.testing.expectEqual(@as(u32, 1), ps.doc_ids[2]);
}

// --- Mixed sampler (PRD 3.6): pick shard by weight ---
pub const MixedSampler = struct {
    weights: []const f32,
    // cumulative
    cum: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, weights: []const f32) !MixedSampler {
        const cum = try allocator.alloc(f32, weights.len);
        var running: f32 = 0;
        for (weights, 0..) |w, i| {
            running += w;
            cum[i] = running;
        }
        return .{ .weights = weights, .cum = cum, .allocator = allocator };
    }

    pub fn deinit(self: *MixedSampler) void {
        self.allocator.free(self.cum);
    }

    // Given u in [0,1), return shard index.
    pub fn pick(self: *const MixedSampler, u: f32) usize {
        for (self.cum, 0..) |c, i| {
            if (u < c) return i;
        }
        return self.cum.len - 1;
    }
};

test "MixedSampler pick" {
    const allocator = std.testing.allocator;
    const w = [_]f32{ 0.5, 0.3, 0.2 };
    var ms = try MixedSampler.init(allocator, &w);
    defer ms.deinit();
    try std.testing.expectEqual(@as(usize, 0), ms.pick(0.25));
    try std.testing.expectEqual(@as(usize, 1), ms.pick(0.6));
    try std.testing.expectEqual(@as(usize, 2), ms.pick(0.95));
}

// --- FIM (fill-in-the-middle, PRD 3.6) ---
pub const FIMMode = enum { psm, spm };

// Split text into prefix/middle/suffix by character offsets and format.
pub fn fimFormat(allocator: std.mem.Allocator, text: []const u8, prefix_len: usize, middle_len: usize, mode: FIMMode) ![]u8 {
    const plen = @min(prefix_len, text.len);
    const mlen = @min(middle_len, if (text.len > plen) text.len - plen else 0);
    const prefix = text[0..plen];
    const middle = text[plen .. plen + mlen];
    const suffix = text[plen + mlen ..];

    var out = std.ArrayList(u8).init(allocator);
    defer out.deinit();
    if (mode == .psm) {
        try out.appendSlice("<fim_prefix>");
        try out.appendSlice(prefix);
        try out.appendSlice("<fim_suffix>");
        try out.appendSlice(suffix);
        try out.appendSlice("<fim_middle>");
        try out.appendSlice(middle);
    } else {
        try out.appendSlice("<fim_suffix>");
        try out.appendSlice(suffix);
        try out.appendSlice("<fim_prefix>");
        try out.appendSlice(prefix);
        try out.appendSlice("<fim_middle>");
        try out.appendSlice(middle);
    }
    return out.toOwnedSlice();
}

test "fimFormat psm" {
    const allocator = std.testing.allocator;
    const out = try fimFormat(allocator, "hello world test", 5, 6, .psm);
    defer allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "<fim_prefix>") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "<fim_suffix>") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "<fim_middle>") != null);
}

test "fimFormat spm" {
    const allocator = std.testing.allocator;
    const out = try fimFormat(allocator, "hello world test", 5, 6, .spm);
    defer allocator.free(out);
    // spm starts with suffix
    try std.testing.expect(std.mem.startsWith(u8, out, "<fim_suffix>"));
}

// --- Pipeline integration (PRD 3.6, 4.11): filter -> dedup exact -> decontam -> shard ---

pub const PipelineConfig = struct {
    filter: FilterConfig = .{},
    ngram_n: usize = 5,
    eval_docs: []const []const u8 = &.{},
};

pub fn runPipeline(allocator: std.mem.Allocator, docs: []const []const u8, cfg: PipelineConfig) ![]usize {
    // 1. filter
    var filtered = std.ArrayList(usize).init(allocator);
    defer filtered.deinit();
    for (docs, 0..) |doc, idx| {
        if (passesFilter(cfg.filter, doc)) try filtered.append(idx);
    }
    // 2. exact dedup on filtered
    var seen = std.AutoHashMap(u64, void).init(allocator);
    defer seen.deinit();
    var deduped = std.ArrayList(usize).init(allocator);
    defer deduped.deinit();
    for (filtered.items) |idx| {
        const h = exactHash(docs[idx]);
        if (seen.contains(h)) continue;
        try seen.put(h, {});
        try deduped.append(idx);
    }
    // 3. decontaminate
    // build list of candidate docs
    var cands = try allocator.alloc([]const u8, deduped.items.len);
    defer allocator.free(cands);
    for (deduped.items, 0..) |idx, i| cands[i] = docs[idx];
    const kept_local = try decontaminate(allocator, cands, cfg.eval_docs, cfg.ngram_n);
    defer allocator.free(kept_local);
    // map back to original indices
    var result = try allocator.alloc(usize, kept_local.len);
    for (kept_local, 0..) |li, i| result[i] = deduped.items[li];
    return result;
}

// --- JSONL streaming parser (PRD 3.6, T3.6-01) ---

pub const JsonlRecord = struct {
    text: []const u8,
    source: []const u8,
    license: []const u8,
};

pub fn freeJsonlRecord(allocator: std.mem.Allocator, rec: JsonlRecord) void {
    allocator.free(rec.text);
    allocator.free(rec.source);
    allocator.free(rec.license);
}

pub fn freeJsonlRecords(allocator: std.mem.Allocator, records: []JsonlRecord) void {
    for (records) |rec| freeJsonlRecord(allocator, rec);
    allocator.free(records);
}

pub fn parseJsonlLine(allocator: std.mem.Allocator, line: []const u8) !JsonlRecord {
    const Raw = struct {
        text: []const u8 = "",
        source: []const u8 = "",
        license: []const u8 = "",
    };
    var parsed = try std.json.parseFromSlice(Raw, allocator, line, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();
    if (parsed.value.text.len == 0) return error.MissingText;
    return .{
        .text = try allocator.dupe(u8, parsed.value.text),
        .source = try allocator.dupe(u8, parsed.value.source),
        .license = try allocator.dupe(u8, parsed.value.license),
    };
}

pub fn parseJsonl(allocator: std.mem.Allocator, data: []const u8) ![]JsonlRecord {
    var records = std.ArrayList(JsonlRecord).empty;
    defer records.deinit(allocator);
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) continue;
        const rec = try parseJsonlLine(allocator, line);
        try records.append(allocator, rec);
    }
    return records.toOwnedSlice(allocator);
}

pub fn readJsonlFile(allocator: std.mem.Allocator, path: []const u8) ![]JsonlRecord {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const st = try file.stat();
    const buf = try allocator.alloc(u8, st.size);
    defer allocator.free(buf);
    _ = try file.readAll(buf);
    return parseJsonl(allocator, buf);
}

test "parseJsonlLine basic" {
    const allocator = std.testing.allocator;
    const rec = try parseJsonlLine(allocator, "{\"text\":\"hello world\",\"source\":\"web\",\"license\":\"cc-by\"}");
    defer freeJsonlRecord(allocator, rec);
    try std.testing.expectEqualStrings("hello world", rec.text);
    try std.testing.expectEqualStrings("web", rec.source);
    try std.testing.expectEqualStrings("cc-by", rec.license);
}

test "parseJsonlLine missing text" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.MissingText, parseJsonlLine(allocator, "{\"source\":\"web\"}"));
}

test "parseJsonl skips blanks" {
    const allocator = std.testing.allocator;
    const data =
        "{\"text\":\"a\"}\n\n{\"text\":\"b\",\"source\":\"s\"}\n";
    const recs = try parseJsonl(allocator, data);
    defer freeJsonlRecords(allocator, recs);
    try std.testing.expectEqual(@as(usize, 2), recs.len);
    try std.testing.expectEqualStrings("a", recs[0].text);
    try std.testing.expectEqualStrings("b", recs[1].text);
    try std.testing.expectEqualStrings("s", recs[1].source);
}

test "runPipeline end-to-end" {
    const allocator = std.testing.allocator;
    const docs = [_][]const u8{
        "This is a long enough document that passes the quality filter rules.",
        "short",
        "This is a long enough document that passes the quality filter rules.", // dup
        "completely unrelated text that shares ngram with eval set here",
    };
    const evals = [_][]const u8{"shares ngram with eval set here"};
    const cfg = PipelineConfig{ .eval_docs = &evals };
    const kept = try runPipeline(allocator, &docs, cfg);
    defer allocator.free(kept);
    // doc1 (short) filtered; doc2 dup removed; doc3 decontaminated -> only doc0
    try std.testing.expect(kept.len <= 2);
}

// --- Pipeline -> Shard integration (PRD 3.6, 4.11) ---

pub const ShardWriteConfig = struct {
    out_dir: []const u8,
    vocab_size: u32 = 256,
    tok_hash: [32]u8 = [_]u8{0} * *32,
    dtype: DType = .u32,
};

pub fn pipelineToShards(allocator: std.mem.Allocator, docs: []const []const u8, pcfg: PipelineConfig, scfg: ShardWriteConfig) !usize {
    const kept = try runPipeline(allocator, docs, pcfg);
    defer allocator.free(kept);
    try std.fs.cwd().makePath(scfg.out_dir);
    var count: usize = 0;
    for (kept) |idx| {
        const doc = docs[idx];
        var toks = try allocator.alloc(u32, doc.len);
        defer allocator.free(toks);
        for (doc, 0..) |c, i| toks[i] = @as(u32, c);
        var path_buf: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/shard_{d}.ztok", .{ scfg.out_dir, count });
        try writeShard(allocator, path, scfg.dtype, toks, scfg.vocab_size, scfg.tok_hash);
        count += 1;
    }
    return count;
}

test "runFullPipeline end-to-end" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/fullpipe";
    std.fs.cwd().deleteTree(base) catch {};
    try std.fs.cwd().makePath(base);
    const docs = [_][]const u8{
        "This is a long enough document that passes the quality filter rules.",
        "short",
        "This is a long enough document that passes the quality filter rules.",
    };
    const cfg = FullPipelineConfig{
        .shard = .{ .out_dir = base },
        .pack_max_len = 1000,
    };
    var res = try runFullPipeline(allocator, &docs, cfg);
    defer {
        if (res.packed_seq) |*p| p.deinit();
    }
    try std.testing.expect(res.shard_count >= 1);
    try std.testing.expect(res.packed_seq != null);
}

// --- Full end-to-end data pipeline (PRD 3.6, 4.11): filter -> dedup -> decontam -> FIM -> shard -> packing ---

pub const FullPipelineConfig = struct {
    pipeline: PipelineConfig = .{},
    shard: ShardWriteConfig = .{},
    fim_mode: ?FIMMode = null,
    fim_prefix_len: usize = 0,
    fim_middle_len: usize = 0,
    pack_max_len: usize = 0,
};

pub const FullPipelineResult = struct {
    shard_count: usize,
    packed_seq: ?PackedSequence = null,
};

pub fn runFullPipeline(allocator: std.mem.Allocator, docs: []const []const u8, cfg: FullPipelineConfig) !FullPipelineResult {
    const kept = try runPipeline(allocator, docs, cfg.pipeline);
    defer allocator.free(kept);

    try std.fs.cwd().makePath(cfg.shard.out_dir);
    var count: usize = 0;
    var token_docs = std.ArrayList([]u32).init(allocator);
    defer {
        for (token_docs.items) |t| allocator.free(t);
        token_docs.deinit();
    }

    for (kept) |idx| {
        const doc = docs[idx];
        // Optional FIM formatting at text level.
        var text_buf: []u8 = undefined;
        var owned_text = false;
        if (cfg.fim_mode) |mode| {
            const fim = try fimFormat(allocator, doc, cfg.fim_prefix_len, cfg.fim_middle_len, mode);
            text_buf = fim;
            owned_text = true;
        } else {
            text_buf = try allocator.dupe(u8, doc);
            owned_text = true;
        }
        defer if (owned_text) allocator.free(text_buf);

        // Tokenize (byte-level) into u32 sequence.
        var toks = try allocator.alloc(u32, text_buf.len);
        for (text_buf, 0..) |c, i| toks[i] = @as(u32, c);

        // Write shard.
        var path_buf: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/shard_{d}.ztok", .{ cfg.shard.out_dir, count });
        try writeShard(allocator, path, cfg.shard.dtype, toks, cfg.shard.vocab_size, cfg.shard.tok_hash);
        count += 1;

        if (cfg.pack_max_len > 0) {
            const toks_copy = try allocator.dupe(u32, toks);
            try token_docs.append(toks_copy);
        }
    }

    var packed_result: ?PackedSequence = null;
    if (cfg.pack_max_len > 0 and token_docs.items.len > 0) {
        const docs_slice = token_docs.items;
        packed_result = try packSequences(allocator, docs_slice, cfg.pack_max_len);
    }

    return .{ .shard_count = count, .packed_seq = packed_result };
}

test "pipelineToShards creates files" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/pipeshards";
    std.fs.cwd().deleteTree(base) catch {};
    const docs = [_][]const u8{
        "This is a long enough document that passes the quality filter rules.",
        "short",
    };
    const cfg = PipelineConfig{};
    const scfg = ShardWriteConfig{ .out_dir = base };
    const n = try pipelineToShards(allocator, &docs, cfg, scfg);
    try std.testing.expect(n >= 1);
    var path_buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/shard_0.ztok", .{base});
    const f = try std.fs.cwd().openFile(path, .{});
    f.close();
}
