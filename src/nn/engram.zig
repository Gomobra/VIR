// Engram (PRD 3.7): lookup n-gram hashed memory with context gating.

const std = @import("std");
const cpu = @import("../cpu/cpu.zig");
const autograd = @import("../autograd/autograd.zig");

pub const Engram = struct {
    d: usize,
    mem_dim: usize,
    n_heads: usize,
    orders: []u8,
    table_size: usize,
    num_tables: usize,
    sub_dim: usize,
    tables: []f32,
    wk: []f32,
    wv: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, d: usize, n_heads: usize, orders: []const u8, table_size: usize) !Engram {
        const num_tables = orders.len * n_heads;
        const sub_dim: usize = 4;
        const mem_dim = num_tables * sub_dim;
        const tables = try allocator.alloc(f32, num_tables * table_size * sub_dim);
        for (tables) |*x| x.* = 0.01;
        const wk = try allocator.alloc(f32, mem_dim * mem_dim);
        const wv = try allocator.alloc(f32, mem_dim * mem_dim);
        for (wk) |*x| x.* = 0.01;
        for (wv) |*x| x.* = 0.01;
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

    fn hashNgram(ids: []const u32, table_size: usize, head: usize) usize {
        var h: u64 = 0xcbf29ce484222325;
        for (ids) |id| {
            h ^= id +% @as(u32, @intCast(head));
            h *%= 0x100000001b3;
        }
        return @intCast(h % table_size);
    }

    fn retrieve(self: *const Engram, e: []f32, token_ids: []const u32, t: usize) void {
        @memset(e, 0);
        for (self.orders, 0..) |ord, oi| {
            if (t + 1 < ord) continue;
            var ng: [4]u32 = undefined;
            var ni: usize = 0;
            const start = t + 1 - ord;
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
    }

    pub fn forward(self: *const Engram, out: []f32, h: []const f32, token_ids: []const u32, seq: usize) void {
        std.debug.assert(out.len == seq * self.d);
        std.debug.assert(h.len == seq * self.d);
        std.debug.assert(token_ids.len == seq);
        const mem = self.mem_dim;
        const dd = @min(self.d, mem);
        var t: usize = 0;
        while (t < seq) : (t += 1) {
            const e = self.allocator.alloc(f32, mem) catch return;
            defer self.allocator.free(e);
            self.retrieve(e, token_ids, t);
            const kvec = self.allocator.alloc(f32, mem) catch return;
            defer self.allocator.free(kvec);
            const vvec = self.allocator.alloc(f32, mem) catch return;
            defer self.allocator.free(vvec);
            var i: usize = 0;
            while (i < mem) : (i += 1) {
                var acc: f32 = 0;
                var j: usize = 0;
                while (j < mem) : (j += 1) acc += self.wk[i * mem + j] * e[j];
                kvec[i] = acc;
            }
            i = 0;
            while (i < mem) : (i += 1) {
                var acc: f32 = 0;
                var j: usize = 0;
                while (j < mem) : (j += 1) acc += self.wv[i * mem + j] * e[j];
                vvec[i] = acc;
            }
            const hrms = self.allocator.alloc(f32, self.d) catch return;
            defer self.allocator.free(hrms);
            const krms = self.allocator.alloc(f32, mem) catch return;
            defer self.allocator.free(krms);
            const hslice = h[t * self.d .. (t + 1) * self.d];
            cpu.rmsnormF32(hrms, hslice, 1e-6);
            cpu.rmsnormF32(krms, kvec, 1e-6);
            var dot: f32 = 0;
            var ii: usize = 0;
            while (ii < dd) : (ii += 1) dot += hrms[ii] * krms[ii];
            const alpha = 1.0 / (1.0 + @exp(-dot / @sqrt(@as(f32, @floatFromInt(dd)))));
            ii = 0;
            while (ii < self.d) : (ii += 1) {
                const vi = if (ii < mem) vvec[ii] else 0;
                out[t * self.d + ii] += alpha * vi;
            }
        }
    }

    // Deterministic prefetch addresses for host-side table lookup (PRD 2.5/3.7).
    // Returns a flat list of (table_idx * table_size + entry_idx) for each (position, order, head).
    pub fn prefetchIndices(self: *const Engram, allocator: std.mem.Allocator, token_ids: []const u32, seq: usize) ![]usize {
        var out = std.ArrayList(usize).init(allocator);
        defer out.deinit();
        var t: usize = 0;
        while (t < seq) : (t += 1) {
            for (self.orders, 0..) |ord, oi| {
                if (t + 1 < ord) continue;
                var ng: [4]u32 = undefined;
                var ni: usize = 0;
                const start = t + 1 - ord;
                while (start + ni <= t) : (ni += 1) {
                    if (ni >= 4) break;
                    ng[ni] = token_ids[start + ni];
                }
                const gram = ng[0..ord];
                var hi: usize = 0;
                while (hi < self.n_heads) : (hi += 1) {
                    const table_idx = oi * self.n_heads + hi;
                    const idx = hashNgram(gram, self.table_size, hi);
                    try out.append(table_idx * self.table_size + idx);
                }
            }
        }
        return out.toOwnedSlice();
    }

    // Backward: given dOut (seq x d), compute dH (seq x d) (engram contribution) and accumulate dWk, dWv.
    pub fn backward(self: *Engram, dH: []f32, dWk: []f32, dWv: []f32, dTables: []f32, h: []const f32, token_ids: []const u32, dOut: []const f32, seq: usize) void {
        const d = self.d;
        const mem = self.mem_dim;
        const dd = @min(d, mem);
        for (dH) |*x| x.* = 0;
        for (dWk) |*x| x.* = 0;
        for (dWv) |*x| x.* = 0;

        var t: usize = 0;
        while (t < seq) : (t += 1) {
            const hslice = h[t * d .. (t + 1) * d];
            const dout = dOut[t * d .. (t + 1) * d];
            const e = self.allocator.alloc(f32, mem) catch return;
            defer self.allocator.free(e);
            self.retrieve(e, token_ids, t);
            const kvec = self.allocator.alloc(f32, mem) catch return;
            defer self.allocator.free(kvec);
            const vvec = self.allocator.alloc(f32, mem) catch return;
            defer self.allocator.free(vvec);
            var i: usize = 0;
            while (i < mem) : (i += 1) {
                var acc: f32 = 0;
                var j: usize = 0;
                while (j < mem) : (j += 1) acc += self.wk[i * mem + j] * e[j];
                kvec[i] = acc;
            }
            i = 0;
            while (i < mem) : (i += 1) {
                var acc: f32 = 0;
                var j: usize = 0;
                while (j < mem) : (j += 1) acc += self.wv[i * mem + j] * e[j];
                vvec[i] = acc;
            }
            const hrms = self.allocator.alloc(f32, d) catch return;
            defer self.allocator.free(hrms);
            const krms = self.allocator.alloc(f32, mem) catch return;
            defer self.allocator.free(krms);
            cpu.rmsnormF32(hrms, hslice, 1e-6);
            cpu.rmsnormF32(krms, kvec, 1e-6);
            var dot: f32 = 0;
            var ii: usize = 0;
            while (ii < dd) : (ii += 1) dot += hrms[ii] * krms[ii];
            const alpha = 1.0 / (1.0 + @exp(-dot / @sqrt(@as(f32, @floatFromInt(dd)))));

            // dalpha = sum_i dout[i]*v[i]
            var dalpha: f32 = 0;
            ii = 0;
            while (ii < d) : (ii += 1) {
                const vi = if (ii < mem) vvec[ii] else 0;
                dalpha += dout[ii] * vi;
            }
            const sig = alpha * (1 - alpha);
            const scale = sig / @sqrt(@as(f32, @floatFromInt(dd)));

            // dHrms = dalpha * scale * krms (padded)
            var dhrms = self.allocator.alloc(f32, d) catch return;
            defer self.allocator.free(dhrms);
            @memset(dhrms, 0);
            ii = 0;
            while (ii < dd) : (ii += 1) dhrms[ii] = dalpha * scale * krms[ii];
            // rmsnorm backward for h
            var ss: f32 = 0;
            for (hslice) |v| ss += v * v;
            const rms = @sqrt(ss / @as(f32, @floatFromInt(d)) + 1e-6);
            var dot2: f32 = 0;
            for (hslice, dhrms) |xi, gi| dot2 += xi * gi;
            const sc = dot2 / (ss + 1e-6 * @as(f32, @floatFromInt(d)));
            ii = 0;
            while (ii < d) : (ii += 1) {
                dH[t * d + ii] += (dhrms[ii] - hslice[ii] * sc) / rms;
            }

            // dV = alpha * dOut (for first d dims, padded to mem)
            var dvvec = self.allocator.alloc(f32, mem) catch return;
            defer self.allocator.free(dvvec);
            @memset(dvvec, 0);
            ii = 0;
            while (ii < d) : (ii += 1) {
                if (ii < mem) dvvec[ii] = alpha * dout[ii];
            }
            // dWv += dvvec (outer) e
            var ia: usize = 0;
            while (ia < mem) : (ia += 1) {
                var j: usize = 0;
                while (j < mem) : (j += 1) {
                    dWv[ia * mem + j] += dvvec[ia] * e[j];
                }
            }

            // dKrms = dalpha * scale * hrms (padded to mem)
            var dkrms = self.allocator.alloc(f32, mem) catch return;
            defer self.allocator.free(dkrms);
            @memset(dkrms, 0);
            ii = 0;
            while (ii < dd) : (ii += 1) dkrms[ii] = dalpha * scale * hrms[ii];
            // rmsnorm backward for k
            var ss2: f32 = 0;
            for (kvec) |v| ss2 += v * v;
            const rms2 = @sqrt(ss2 / @as(f32, @floatFromInt(mem)) + 1e-6);
            var dot3: f32 = 0;
            for (kvec, dkrms) |xi, gi| dot3 += xi * gi;
            const sc2 = dot3 / (ss2 + 1e-6 * @as(f32, @floatFromInt(mem)));
            var dkvec = self.allocator.alloc(f32, mem) catch return;
            defer self.allocator.free(dkvec);
            @memset(dkvec, 0);
            ii = 0;
            while (ii < mem) : (ii += 1) {
                dkvec[ii] = (dkrms[ii] - kvec[ii] * sc2) / rms2;
            }
            // dE via Wk: dE += Wk^T @ dkvec (accumulate into dTables? no, dE is intermediate)
            // dWk += dkvec (outer) e
            i = 0;
            while (i < mem) : (i += 1) {
                var j: usize = 0;
                while (j < mem) : (j += 1) {
                    dWk[i * mem + j] += dkvec[i] * e[j];
                }
            }
            // dTables: scatter dE (from both projections) into retrieved table entries.
            // dE = Wk^T dkvec + Wv^T dvvec
            var de = self.allocator.alloc(f32, mem) catch return;
            defer self.allocator.free(de);
            @memset(de, 0);
            i = 0;
            while (i < mem) : (i += 1) {
                var acc: f32 = 0;
                var j: usize = 0;
                while (j < mem) : (j += 1) acc += self.wk[j * mem + i] * dkvec[j];
                de[i] += acc;
            }
            i = 0;
            while (i < mem) : (i += 1) {
                var acc: f32 = 0;
                var j: usize = 0;
                while (j < mem) : (j += 1) acc += self.wv[j * mem + i] * dvvec[j];
                de[i] += acc;
            }
            for (self.orders, 0..) |ord, oi| {
                if (t + 1 < ord) continue;
                var ng: [4]u32 = undefined;
                var ni: usize = 0;
                const start = t + 1 - ord;
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
                        dTables[base_table + idx * self.sub_dim + s] += de[base_e + s];
                    }
                }
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

test "Engram backward finite" {
    const allocator = std.testing.allocator;
    const orders = [_]u8{ 2, 3 };
    var eng = try Engram.init(allocator, 4, 2, &orders, 16);
    defer eng.deinit();
    var h = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const ids = [_]u32{ 1, 2 };
    var dOut = [_]f32{ 1, 0, 0, 1, 0, 1, 1, 0 };
    var dH = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const mem = eng.mem_dim;
    const dWk = try allocator.alloc(f32, mem * mem);
    defer allocator.free(dWk);
    @memset(dWk, 0);
    const dWv = try allocator.alloc(f32, mem * mem);
    defer allocator.free(dWv);
    @memset(dWv, 0);
    const dTables = try allocator.alloc(f32, eng.tables.len);
    defer allocator.free(dTables);
    @memset(dTables, 0);
    eng.backward(&dH, dWk, dWv, dTables, &h, &ids, &dOut, 2);
    for (dH) |v| try std.testing.expect(std.math.isFinite(v));
}

test "Engram prefetch deterministic" {
    const allocator = std.testing.allocator;
    const orders = [_]u8{ 2, 3 };
    var eng = try Engram.init(allocator, 4, 2, &orders, 16);
    defer eng.deinit();
    const ids = [_]u32{ 1, 2, 3 };
    const a = try eng.prefetchIndices(allocator, &ids, 3);
    defer allocator.free(a);
    const b = try eng.prefetchIndices(allocator, &ids, 3);
    defer allocator.free(b);
    try std.testing.expectEqualSlices(usize, a, b);
    try std.testing.expect(a.len > 0);
}

test "Engram gradcheck h" {
    const allocator = std.testing.allocator;
    const orders = [_]u8{ 2, 3 };
    var eng = try Engram.init(allocator, 4, 2, &orders, 16);
    defer eng.deinit();
    var h = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const ids = [_]u32{ 1, 2 };
    // analytic backward
    var dOut = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    var dH = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const mem = eng.mem_dim;
    const dWk = try allocator.alloc(f32, mem * mem);
    defer allocator.free(dWk);
    @memset(dWk, 0);
    const dWv = try allocator.alloc(f32, mem * mem);
    defer allocator.free(dWv);
    @memset(dWv, 0);
    const dTables = try allocator.alloc(f32, eng.tables.len);
    defer allocator.free(dTables);
    @memset(dTables, 0);
    eng.backward(&dH, dWk, dWv, dTables, &h, &ids, &dOut, 2);
    // finite diff for h[0]: loss = sum(engram output), perturb h[0]
    const eps: f32 = 1e-3;
    var hp = h;
    hp[0] += eps;
    var hm = h;
    hm[0] -= eps;
    var op = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var om = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    eng.forward(&op, &hp, &ids, 2);
    eng.forward(&om, &hm, &ids, 2);
    var sp: f32 = 0;
    var sm: f32 = 0;
    for (op) |v| sp += v;
    for (om) |v| sm += v;
    const fd = (sp - sm) / (2 * eps);
    try std.testing.expect(@abs(fd - dH[0]) < 1e-2);
}
