// Engram (PRD 3.7): lookup n-gram hashed memory with context gating.

const std = @import("std");
const cpu = @import("../cpu/cpu.zig");

pub const Engram = struct {
    d: usize, // hidden dim
    mem_dim: usize, // memory vector dim (concatenated)
    n_heads: usize,
    orders: []const u8, // e.g. {2,3}
    table_size: usize, // prime-ish
    // tables: for each (order, head) a table of mem_dim/n_heads? simplified: one table per (order,head) of size table_size x (mem_dim/n_heads_per_order)
    // We'll use a single flat table: num_tables = orders.len * n_heads, each table_size * sub_dim
    num_tables: usize,
    sub_dim: usize,
    tables: []f32, // flattened
    wk: []f32, // [mem_dim, mem_dim] (key projection) simplified square
    wv: []f32, // [mem_dim, mem_dim]
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, d: usize, n_heads: usize, orders: []const u8, table_size: usize) !Engram {
        const num_tables = orders.len * n_heads;
        const sub_dim = 4; // small
        const mem_dim = num_tables * sub_dim;
        const tables = try allocator.alloc(f32, num_tables * table_size * sub_dim);
        for (tables) |*x| x.* = 0.01;
        const wk = try allocator.alloc(f32, mem_dim * mem_dim);
        const wv = try allocator.alloc(f32, mem_dim * mem_dim);
        for (wk) |*x| x.* = 0.01;
        for (wv) |*x| x.* = 0.01;
        // need owned copy of orders
        const orders_copy = try allocator.dupe(u8, orders);
        return .{
            .d = d,
            .mem_dim = mem_dim,
            .n_heads = n_heads,
            .orders = orders_copy,
            .table_size = table_size,
            .num_tables = num_tables,
            .sub_dim = sub_dim,
            .tables = tables,
            .wk = wk,
            .wv = wv,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Engram) void {
        self.allocator.free(self.tables);
        self.allocator.free(self.wk);
        self.allocator.free(self.wv);
        self.allocator.free(self.orders);
    }

    // hash n-gram (list of canonical ids) to index
    fn hashNgram(ids: []const u32, table_size: usize, head: usize) usize {
        var h: u64 = 0xcbf29ce484222325;
        for (ids) |id| {
            h ^= id +% @as(u32, @intCast(head));
            h *%= 0x100000001b3;
        }
        return @intCast(h % table_size);
    }

    // forward: given hidden h (seq x d) and token ids (seq), produce output added to out (seq x d).
    // simplified: mem_dim must equal d for residual add; here we project mem->d? We'll just require mem_dim==d for simplicity.
    pub fn forward(self: *const Engram, out: []f32, h: []const f32, token_ids: []const u32, seq: usize) void {
        std.debug.assert(out.len == seq * self.d);
        std.debug.assert(h.len == seq * self.d);
        std.debug.assert(token_ids.len == seq);
        // canonical ids = identity here
        var t: usize = 0;
        while (t < seq) : (t += 1) {
            // build n-grams ending at t
            // retrieve memory vector e (mem_dim)
            var e = self.allocator.alloc(f32, self.mem_dim) catch return;
            defer self.allocator.free(e);
            @memset(e, 0);
            for (self.orders, 0..) |ord, oi| {
                if (t + 1 < ord) continue;
                // n-gram = token_ids[t-ord+1 .. t+1]
                var ng: [4]u32 = undefined;
                var ni: usize = 0;
                var start = t + 1 - ord;
                while (start + ni <= t) : (ni += 1) {
                    if (ni >= 4) break;
                    ng[ni] = token_ids[start + ni];
                }
                const gram = ng[0..ord];
                var hi: usize = 0;
                while (hi < self.n_heads) : (hi += 1) {
                    const table_idx = oi * self.n_heads + hi;
                    const idx = hashNgram(gram, self.table_size, hi);
                    const base_table = table_idx * self.table_size * self.sub_dim;
                    const base_e = table_idx * self.sub_dim;
                    var s: usize = 0;
                    while (s < self.sub_dim) : (s += 1) {
                        e[base_e + s] = self.tables[base_table + idx * self.sub_dim + s];
                    }
                }
            }
            // context gating: k = Wk @ e, v = Wv @ e; alpha = sigmoid(RMSNorm(h)^T RMSNorm(k)/sqrt(mem_dim))
            // simplified: compute k, v as linear (mem_dim x mem_dim) -> mem_dim
            var kvec = self.allocator.alloc(f32, self.mem_dim) catch return;
            defer self.allocator.free(kvec);
            var vvec = self.allocator.alloc(f32, self.mem_dim) catch return;
            defer self.allocator.free(vvec);
            // k = Wk @ e
            var i: usize = 0;
            while (i < self.mem_dim) : (i += 1) {
                var acc: f32 = 0;
                var j: usize = 0;
                while (j < self.mem_dim) : (j += 1) acc += self.wk[i * self.mem_dim + j] * e[j];
                kvec[i] = acc;
            }
            i = 0;
            while (i < self.mem_dim) : (i += 1) {
                var acc: f32 = 0;
                var j: usize = 0;
                while (j < self.mem_dim) : (j += 1) acc += self.wv[i * self.mem_dim + j] * e[j];
                vvec[i] = acc;
            }
            // rmsnorm h and k
            var hrms = self.allocator.alloc(f32, self.d) catch return;
            defer self.allocator.free(hrms);
            var krms = self.allocator.alloc(f32, self.mem_dim) catch return;
            defer self.allocator.free(krms);
            // h slice for this timestep
            const hslice = h[t * self.d .. (t + 1) * self.d];
            cpu.rmsnormF32(hrms, hslice, 1e-6);
            cpu.rmsnormF32(krms, kvec, 1e-6);
            // alpha = sigmoid(dot(hrms padded? ) / sqrt(d)) -- mem_dim may != d; use min
            const dd = @min(self.d, self.mem_dim);
            var dot: f32 = 0;
            var ii: usize = 0;
            while (ii < dd) : (ii += 1) dot += hrms[ii] * krms[ii];
            const alpha = 1.0 / (1.0 + @exp(-dot / @sqrt(@as(f32, @floatFromInt(dd)))));
            // output contribution: alpha * v, added to out (if dims match; pad/truncate)
            ii = 0;
            while (ii < self.d) : (ii += 1) {
                const vi = if (ii < self.mem_dim) vvec[ii] else 0;
                out[t * self.d + ii] += alpha * vi;
            }
        }
    }
};

test "Engram forward finite" {
    const allocator = std.testing.allocator;
    const orders = [_]u8{ 2, 3 };
    var eng = try Engram.init(allocator, 4, 2, &orders, 16);
    defer eng.deinit();
    var h = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var out = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const ids = [_]u32{ 1, 2 };
    eng.forward(&out, &h, &ids, 2);
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}
