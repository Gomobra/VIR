// Tokenizer BPE baseline (PRD 3.5). Vocab + BPE trainer.

const std = @import("std");

pub const Vocab = struct {
    tokens: std.ArrayList([]u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Vocab {
        return .{ .tokens = std.ArrayList([]u8).init(allocator), .allocator = allocator };
    }

    pub fn deinit(self: *Vocab) void {
        for (self.tokens.items) |t| self.allocator.free(t);
        self.tokens.deinit();
    }

    pub fn addToken(self: *Vocab, bytes: []const u8) !u32 {
        const copy = try self.allocator.dupe(u8, bytes);
        try self.tokens.append(copy);
        return @intCast(self.tokens.items.len - 1);
    }

    pub fn size(self: *const Vocab) usize {
        return self.tokens.items.len;
    }
};

pub fn encodeBytes(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    return allocator.dupe(u8, s);
}

// --- BPE trainer (byte-level) ---

pub const BPETrainer = struct {
    allocator: std.mem.Allocator,
    vocab_size: usize,
    // corpus: list of pretokens (each []u8) with counts
    words: std.ArrayList([]u8),
    counts: std.ArrayList(usize),
    // each word's current token sequence (list of token ids)
    seqs: std.ArrayList(std.ArrayList(u32)),
    vocab: Vocab,
    // merges: pairs (a,b) -> new id
    merges: std.ArrayList(struct { a: u32, b: u32, newid: u32 }),

    pub fn init(allocator: std.mem.Allocator, vocab_size: usize) BPETrainer {
        return .{
            .allocator = allocator,
            .vocab_size = vocab_size,
            .words = std.ArrayList([]u8).init(allocator),
            .counts = std.ArrayList(usize).init(allocator),
            .seqs = std.ArrayList(std.ArrayList(u32)).init(allocator),
            .vocab = Vocab.init(allocator),
            .merges = std.ArrayList(struct { a: u32, b: u32, newid: u32 }).init(allocator),
        };
    }

    pub fn deinit(self: *BPETrainer) void {
        for (self.words.items) |w| self.allocator.free(w);
        self.words.deinit();
        self.counts.deinit();
        for (self.seqs.items) |*s| s.deinit();
        self.seqs.deinit();
        self.vocab.deinit();
        self.merges.deinit();
    }

    // Add a pretoken with count.
    pub fn addWord(self: *BPETrainer, word: []const u8, count: usize) !void {
        const copy = try self.allocator.dupe(u8, word);
        try self.words.append(copy);
        try self.counts.append(count);
        var seq = std.ArrayList(u32).init(self.allocator);
        for (word) |c| try seq.append(@as(u32, c)); // byte id = char value (0..255)
        try self.seqs.append(seq);
    }

    // Train: perform merges until vocab_size reached.
    // Encode text to token ids using trained merges. Preserves spaces as byte-32 tokens.
    pub fn encode(self: *const BPETrainer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var out = std.ArrayList(u32).init(allocator);
        defer out.deinit();
        var word_buf = std.ArrayList(u8).init(allocator);
        defer word_buf.deinit();
        for (text) |c| {
            if (c == ' ' or c == '\t' or c == '\n') {
                // flush word
                if (word_buf.items.len > 0) {
                    var seq = std.ArrayList(u32).init(allocator);
                    defer seq.deinit();
                    for (word_buf.items) |ch| try seq.append(@as(u32, ch));
                    for (self.merges.items) |mg| {
                        var i: usize = 0;
                        while (i + 1 < seq.items.len) {
                            if (seq.items[i] == mg.a and seq.items[i + 1] == mg.b) {
                                seq.items[i] = mg.newid;
                                var j = i + 1;
                                while (j + 1 < seq.items.len) : (j += 1) seq.items[j] = seq.items[j + 1];
                                seq.items.len -= 1;
                            } else {
                                i += 1;
                            }
                        }
                    }
                    for (seq.items) |id| try out.append(id);
                    word_buf.clearRetainingCapacity();
                }
                // emit space/newline as byte token
                try out.append(@as(u32, c));
            } else {
                try word_buf.append(c);
            }
        }
        // flush last word
        if (word_buf.items.len > 0) {
            var seq = std.ArrayList(u32).init(allocator);
            defer seq.deinit();
            for (word_buf.items) |ch| try seq.append(@as(u32, ch));
            for (self.merges.items) |mg| {
                var i: usize = 0;
                while (i + 1 < seq.items.len) {
                    if (seq.items[i] == mg.a and seq.items[i + 1] == mg.b) {
                        seq.items[i] = mg.newid;
                        var j = i + 1;
                        while (j + 1 < seq.items.len) : (j += 1) seq.items[j] = seq.items[j + 1];
                        seq.items.len -= 1;
                    } else {
                        i += 1;
                    }
                }
            }
            for (seq.items) |id| try out.append(id);
        }
        return out.toOwnedSlice();
    }

    // Decode token ids to bytes.
    pub fn decode(self: *const BPETrainer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        var out = std.ArrayList(u8).init(allocator);
        defer out.deinit();
        for (ids) |id| {
            if (id >= self.vocab.tokens.items.len) return error.InvalidToken;
            const tok = self.vocab.tokens.items[id];
            try out.appendSlice(tok);
        }
        return out.toOwnedSlice();
    }

    pub fn train(self: *BPETrainer) !void {
        // initial vocab: 256 bytes
        var b: usize = 0;
        while (b < 256) : (b += 1) {
            var buf: [1]u8 = .{@intCast(b)};
            _ = try self.vocab.addToken(&buf);
        }
        // merge loop
        while (self.vocab.size() < self.vocab_size) {
            // count pairs
            var pair_counts = std.AutoHashMap(u64, usize).init(self.allocator);
            defer pair_counts.deinit();
            for (self.seqs.items, 0..) |seq, wi| {
                const cnt = self.counts.items[wi];
                var i: usize = 0;
                while (i + 1 < seq.items.len) : (i += 1) {
                    const key = (@as(u64, seq.items[i]) << 32) | seq.items[i + 1];
                    const gop = try pair_counts.getOrPut(key);
                    if (!gop.found_existing) gop.value_ptr.* = 0;
                    gop.value_ptr.* += cnt;
                }
            }
            // find best pair
            var best_key: u64 = 0;
            var best_count: usize = 0;
            var it = pair_counts.iterator();
            while (it.next()) |e| {
                if (e.value_ptr.* > best_count) {
                    best_count = e.value_ptr.*;
                    best_key = e.key_ptr.*;
                }
            }
            if (best_count == 0) break; // no more pairs
            const a: u32 = @intCast(best_key >> 32);
            const bb: u32 = @intCast(best_key & 0xffffffff);

            // create new token bytes: concatenate the two token strings
            const tok_a = self.vocab.tokens.items[a];
            const tok_b = self.vocab.tokens.items[bb];
            const new_bytes = try self.allocator.alloc(u8, tok_a.len + tok_b.len);
            @memcpy(new_bytes[0..tok_a.len], tok_a);
            @memcpy(new_bytes[tok_a.len..], tok_b);
            const newid = try self.vocab.addToken(new_bytes);
            self.allocator.free(new_bytes);

            // merge in all seqs
            for (self.seqs.items) |*seq| {
                var i: usize = 0;
                while (i + 1 < seq.items.len) {
                    if (seq.items[i] == a and seq.items[i + 1] == bb) {
                        seq.items[i] = newid;
                        // remove i+1
                        var j = i + 1;
                        while (j + 1 < seq.items.len) : (j += 1) {
                            seq.items[j] = seq.items[j + 1];
                        }
                        seq.items.len -= 1;
                    } else {
                        i += 1;
                    }
                }
            }

            try self.merges.append(.{ .a = a, .b = bb, .newid = newid });
        }
    }
};

test "BPETrainer basic" {
    const allocator = std.testing.allocator;
    var tr = BPETrainer.init(allocator, 258); // 256 + 2 merges
    defer tr.deinit();
    try tr.addWord("ab", 5);
    try tr.addWord("abc", 3);
    try tr.train();
    // vocab should have grown
    try std.testing.expect(tr.vocab.size() >= 256);
    try std.testing.expect(tr.merges.items.len >= 1);
}

test "BPE encode/decode roundtrip" {
    const allocator = std.testing.allocator;
    var tr = BPETrainer.init(allocator, 258);
    defer tr.deinit();
    try tr.addWord("ab", 5);
    try tr.addWord("abc", 3);
    try tr.train();
    const text = "ab abc";
    const ids = try tr.encode(allocator, text);
    defer allocator.free(ids);
    const dec = try tr.decode(allocator, ids);
    defer allocator.free(dec);
    try std.testing.expectEqualStrings(text, dec);
}

// --- SuperBPE two-stage trainer (PRD 3.5) ---
// Stage 1: BPE with whitespace pretokenization (word-level units).
// Stage 2: BPE without whitespace pretokenization (document-level units),
// allowing cross-word "superword" merges.

pub const SuperBPETrainer = struct {
    allocator: std.mem.Allocator,
    vocab_size: usize,
    transition: usize,
    vocab: Vocab,
    merges: std.ArrayList(Merge),
    docs: std.ArrayList([]u8),
    units: std.ArrayList([]u8),
    counts: std.ArrayList(usize),
    seqs: std.ArrayList(std.ArrayList(u32)),
    cache: std.StringHashMap([]u32),

    pub const Merge = struct { a: u32, b: u32, newid: u32 };

    pub fn init(allocator: std.mem.Allocator, vocab_size: usize, transition: usize) SuperBPETrainer {
        return .{
            .allocator = allocator,
            .vocab_size = vocab_size,
            .transition = transition,
            .vocab = Vocab.init(allocator),
            .merges = std.ArrayList(Merge).init(allocator),
            .docs = std.ArrayList([]u8).init(allocator),
            .units = std.ArrayList([]u8).init(allocator),
            .counts = std.ArrayList(usize).init(allocator),
            .seqs = std.ArrayList(std.ArrayList(u32)).init(allocator),
            .cache = std.StringHashMap([]u32).init(allocator),
        };
    }

    pub fn deinit(self: *SuperBPETrainer) void {
        self.vocab.deinit();
        self.merges.deinit();
        for (self.docs.items) |d| self.allocator.free(d);
        self.docs.deinit();
        for (self.units.items) |u| self.allocator.free(u);
        self.units.deinit();
        self.counts.deinit();
        for (self.seqs.items) |*s| s.deinit();
        self.seqs.deinit();
        var cit = self.cache.iterator();
        while (cit.next()) |e| {
            self.allocator.free(e.key_ptr.*);
            self.allocator.free(e.value_ptr.*);
        }
        self.cache.deinit();
    }

    pub fn addText(self: *SuperBPETrainer, text: []const u8) !void {
        const copy = try self.allocator.dupe(u8, text);
        try self.docs.append(copy);
    }

    fn initVocab(self: *SuperBPETrainer) !void {
        if (self.vocab.size() != 0) return;
        var b: usize = 0;
        while (b < 256) : (b += 1) {
            var buf: [1]u8 = .{@intCast(b)};
            _ = try self.vocab.addToken(&buf);
        }
    }

    fn clearUnits(self: *SuperBPETrainer) void {
        for (self.units.items) |u| self.allocator.free(u);
        self.units.clearRetainingCapacity();
        self.counts.clearRetainingCapacity();
        for (self.seqs.items) |*s| s.deinit();
        self.seqs.clearRetainingCapacity();
    }

    fn appendUnit(self: *SuperBPETrainer, bytes: []const u8, count: usize) !void {
        const copy = try self.allocator.dupe(u8, bytes);
        try self.units.append(copy);
        try self.counts.append(count);
        var seq = std.ArrayList(u32).init(self.allocator);
        for (bytes) |c| try seq.append(@as(u32, c));
        // apply existing merges so stage 2 starts from current vocab
        for (self.merges.items) |mg| {
            mergePair(&seq, mg.a, mg.b, mg.newid);
        }
        try self.seqs.append(seq);
    }

    fn mergePair(seq: *std.ArrayList(u32), a: u32, b: u32, newid: u32) void {
        var i: usize = 0;
        while (i + 1 < seq.items.len) {
            if (seq.items[i] == a and seq.items[i + 1] == b) {
                seq.items[i] = newid;
                var j = i + 1;
                while (j + 1 < seq.items.len) : (j += 1) seq.items[j] = seq.items[j + 1];
                seq.items.len -= 1;
            } else {
                i += 1;
            }
        }
    }

    fn mergeOnce(self: *SuperBPETrainer) bool {
        var pair_counts = std.AutoHashMap(u64, usize).init(self.allocator);
        defer pair_counts.deinit();
        for (self.seqs.items, 0..) |seq, wi| {
            const cnt = self.counts.items[wi];
            var i: usize = 0;
            while (i + 1 < seq.items.len) : (i += 1) {
                const key = (@as(u64, seq.items[i]) << 32) | seq.items[i + 1];
                const gop = pair_counts.getOrPut(key) catch continue;
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* += cnt;
            }
        }
        var best_key: u64 = 0;
        var best_count: usize = 0;
        var it = pair_counts.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* > best_count) {
                best_count = e.value_ptr.*;
                best_key = e.key_ptr.*;
            }
        }
        if (best_count == 0) return false;
        const a: u32 = @intCast(best_key >> 32);
        const bb: u32 = @intCast(best_key & 0xffffffff);
        const tok_a = self.vocab.tokens.items[a];
        const tok_b = self.vocab.tokens.items[bb];
        const new_bytes = self.allocator.alloc(u8, tok_a.len + tok_b.len) catch return false;
        defer self.allocator.free(new_bytes);
        @memcpy(new_bytes[0..tok_a.len], tok_a);
        @memcpy(new_bytes[tok_a.len..], tok_b);
        const newid = self.vocab.addToken(new_bytes) catch return false;
        for (self.seqs.items) |*seq| {
            mergePair(seq, a, bb, newid);
        }
        self.merges.append(.{ .a = a, .b = bb, .newid = newid }) catch return false;
        return true;
    }

    pub fn train(self: *SuperBPETrainer) !void {
        try self.initVocab();

        // Stage 1: whitespace-split words.
        var word_counts = std.StringHashMap(usize).init(self.allocator);
        defer word_counts.deinit();
        for (self.docs.items) |doc| {
            var start: usize = 0;
            var i: usize = 0;
            while (i <= doc.len) : (i += 1) {
                const is_space = i == doc.len or doc[i] == ' ' or doc[i] == '\t' or doc[i] == '\n' or doc[i] == '\r';
                if (is_space) {
                    if (i > start) {
                        const w = doc[start..i];
                        const gop = try word_counts.getOrPut(w);
                        if (!gop.found_existing) {
                            gop.key_ptr.* = try self.allocator.dupe(u8, w);
                            gop.value_ptr.* = 0;
                        }
                        gop.value_ptr.* += 1;
                    }
                    start = i + 1;
                }
            }
        }
        var w_it = word_counts.iterator();
        while (w_it.next()) |e| {
            try self.appendUnit(e.key_ptr.*, e.value_ptr.*);
        }
        const stage1_target = @min(self.transition, self.vocab_size);
        while (self.vocab.size() < stage1_target) {
            if (!self.mergeOnce()) break;
        }

        // Stage 2: whole documents (no whitespace barrier).
        self.clearUnits();
        // dedupe docs by pointer equality is enough for small corpora; count duplicates
        var doc_counts = std.StringHashMap(usize).init(self.allocator);
        defer doc_counts.deinit();
        for (self.docs.items) |doc| {
            const gop = try doc_counts.getOrPut(doc);
            if (!gop.found_existing) {
                gop.key_ptr.* = doc;
                gop.value_ptr.* = 0;
            }
            gop.value_ptr.* += 1;
        }
        var d_it = doc_counts.iterator();
        while (d_it.next()) |e| {
            try self.appendUnit(e.key_ptr.*, e.value_ptr.*);
        }
        while (self.vocab.size() < self.vocab_size) {
            if (!self.mergeOnce()) break;
        }
    }

    // Encode with merge cache.
    pub fn encode(self: *SuperBPETrainer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        if (self.cache.get(text)) |cached| {
            return allocator.dupe(u32, cached);
        }
        var seq = std.ArrayList(u32).init(allocator);
        defer seq.deinit();
        for (text) |c| try seq.append(@as(u32, c));
        for (self.merges.items) |mg| {
            mergePair(&seq, mg.a, mg.b, mg.newid);
        }
        const owned = try seq.toOwnedSlice();
        const key = try self.allocator.dupe(u8, text);
        const val = try self.allocator.dupe(u32, owned);
        try self.cache.put(key, val);
        // return a copy owned by caller
        const ret = try allocator.dupe(u32, owned);
        return ret;
    }

    pub fn decode(self: *const SuperBPETrainer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        var out = std.ArrayList(u8).init(allocator);
        defer out.deinit();
        for (ids) |id| {
            if (id >= self.vocab.tokens.items.len) return error.InvalidToken;
            const tok = self.vocab.tokens.items[id];
            try out.appendSlice(tok);
        }
        return out.toOwnedSlice();
    }
};

test "SuperBPETrainer basic" {
    const allocator = std.testing.allocator;
    var tr = SuperBPETrainer.init(allocator, 260, 258);
    defer tr.deinit();
    try tr.addText("ab ab abc");
    try tr.train();
    try std.testing.expect(tr.vocab.size() >= 256);
    try std.testing.expect(tr.merges.items.len >= 1);
}

test "SuperBPE encode/decode roundtrip" {
    const allocator = std.testing.allocator;
    var tr = SuperBPETrainer.init(allocator, 262, 260);
    defer tr.deinit();
    try tr.addText("of the of the");
    try tr.train();
    const text = "of the of the";
    const ids = try tr.encode(allocator, text);
    defer allocator.free(ids);
    const dec = try tr.decode(allocator, ids);
    defer allocator.free(dec);
    try std.testing.expectEqualStrings(text, dec);
}

test "SuperBPE cache" {
    const allocator = std.testing.allocator;
    var tr = SuperBPETrainer.init(allocator, 260, 258);
    defer tr.deinit();
    try tr.addText("ab ab");
    try tr.train();
    const a = try tr.encode(allocator, "ab");
    defer allocator.free(a);
    const b = try tr.encode(allocator, "ab");
    defer allocator.free(b);
    try std.testing.expectEqualSlices(u32, a, b);
}
