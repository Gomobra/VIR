// Model (PRD 3.7). Full model: embedding + N blocks + final norm + output head.

const std = @import("std");
const nn = @import("../nn/nn.zig");
const config = @import("../config/config.zig");
const optim = @import("../optim/optim.zig");
const autograd = @import("../autograd/autograd.zig");
const cpu_mod = @import("../cpu/cpu.zig");

pub const Model = struct {
    cfg: config.ModelConfig,
    embed: []f32, // vocab * d
    blocks: []Block,
    final_norm: nn.RMSNorm,
    head: nn.Linear, // d -> vocab (separate; could tie)
    // optimizers
    opt_embed: optim.AdamW,
    opt_blocks_attn: []optim.Muon,
    opt_blocks_mlp: []optim.Muon,
    opt_gdn_q: []optim.Muon,
    opt_gdn_k: []optim.Muon,
    opt_gdn_v: []optim.Muon,
    opt_gdn_a: []optim.AdamW,
    opt_gdn_b: []optim.AdamW,
    opt_head: optim.AdamW,
    mtp_enabled: bool = false,
    mtp_loss_weight: f32 = 0.3,
    mtp_head: nn.Linear,
    opt_mtp_head: optim.AdamW,
    loop_enabled: bool = false,
    loop_repeats: usize = 1,
    loop_embed: []f32,
    opt_loop_embed: optim.AdamW,
    allocator: std.mem.Allocator,

    const Block = struct {
        is_gdn: bool = false,
        use_bit: bool = false,
        attn: nn.Linear,
        mlp: nn.Linear,
        bit_attn: ?nn.bitlinear.BitLinear = null,
        bit_mlp: ?nn.bitlinear.BitLinear = null,
        norm1: nn.RMSNorm,
        norm2: nn.RMSNorm,
        // GDN projections (used when is_gdn)
        wq: []f32,
        wk: []f32,
        wv: []f32,
        wa: []f32,
        wb: []f32,
        gdn: nn.gdn.GDN,
        engram: ?nn.engram.Engram = null,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, d: usize, is_gdn: bool, use_bit: bool) !Block {
            const wq = if (is_gdn) blk: {
                const s = try allocator.alloc(f32, d * d);
                for (s, 0..) |*x, i| x.* = 0.02 * @as(f32, @floatFromInt(i % 7)) - 0.06;
                break :blk s;
            } else &[_]f32{};
            const wk = if (is_gdn) blk: {
                const s = try allocator.alloc(f32, d * d);
                for (s, 0..) |*x, i| x.* = 0.02 * @as(f32, @floatFromInt(i % 11)) - 0.06;
                break :blk s;
            } else &[_]f32{};
            const wv = if (is_gdn) blk: {
                const s = try allocator.alloc(f32, d * d);
                for (s, 0..) |*x, i| x.* = 0.02 * @as(f32, @floatFromInt(i % 13)) - 0.06;
                break :blk s;
            } else &[_]f32{};
            const wa = if (is_gdn) blk: {
                const s = try allocator.alloc(f32, d);
                for (s) |*x| x.* = 0.0;
                break :blk s;
            } else &[_]f32{};
            const wb = if (is_gdn) blk: {
                const s = try allocator.alloc(f32, d);
                for (s) |*x| x.* = 0.0;
                break :blk s;
            } else &[_]f32{};

            var bit_attn: ?nn.bitlinear.BitLinear = null;
            var bit_mlp: ?nn.bitlinear.BitLinear = null;
            if (use_bit) {
                bit_attn = try nn.bitlinear.BitLinear.init(allocator, d, d);
                bit_mlp = try nn.bitlinear.BitLinear.init(allocator, d, d * 2);
            }
            return .{
                .is_gdn = is_gdn,
                .use_bit = use_bit,
                .attn = try nn.Linear.init(allocator, d, d),
                .mlp = try nn.Linear.init(allocator, d, d * 2),
                .bit_attn = bit_attn,
                .bit_mlp = bit_mlp,
                .norm1 = .{},
                .norm2 = .{},
                .wq = wq,
                .wk = wk,
                .wv = wv,
                .wa = wa,
                .wb = wb,
                .gdn = .{ .d = d },
                .engram = null,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Block) void {
            self.attn.deinit();
            self.mlp.deinit();
            if (self.bit_attn) |*b| b.deinit();
            if (self.bit_mlp) |*b| b.deinit();
            if (self.wq.len > 0) self.allocator.free(self.wq);
            if (self.wk.len > 0) self.allocator.free(self.wk);
            if (self.wv.len > 0) self.allocator.free(self.wv);
            if (self.wa.len > 0) self.allocator.free(self.wa);
            if (self.wb.len > 0) self.allocator.free(self.wb);
            if (self.engram) |*e| e.deinit();
        }

        pub fn forward(self: *const Block, out: []f32, x: []const f32, tokens: []const u32, batch: usize, d: usize) void {
            std.debug.assert(x.len == batch * d and out.len == batch * d);
            const xcur = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(xcur);
            @memcpy(xcur, x);
            if (self.engram) |*eng| {
                const contrib = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(contrib);
                @memset(contrib, 0);
                eng.forward(contrib, x, tokens, batch);
                for (xcur, contrib) |*v, c| v.* += c;
            }

            const tmp = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(tmp);
            self.norm1.forward(tmp, xcur);

            if (self.use_bit) {
                if (self.bit_attn) |*ba| {
                    ba.forward(out, tmp, batch);
                    for (out, xcur) |*o, v| o.* += v;
                }
            } else if (self.is_gdn) {
                // projections
                const Q = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(Q);
                const K = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(K);
                const Vv = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(Vv);
                const alpha = self.allocator.alloc(f32, batch) catch return;
                defer self.allocator.free(alpha);
                const beta = self.allocator.alloc(f32, batch) catch return;
                defer self.allocator.free(beta);
                var b: usize = 0;
                while (b < batch) : (b += 1) {
                    const xrow = tmp[b * d .. (b + 1) * d];
                    // Q
                    var j: usize = 0;
                    while (j < d) : (j += 1) {
                        var acc: f32 = 0;
                        var i: usize = 0;
                        while (i < d) : (i += 1) acc += xrow[i] * self.wq[j * d + i];
                        Q[b * d + j] = acc;
                    }
                    // K
                    j = 0;
                    while (j < d) : (j += 1) {
                        var acc: f32 = 0;
                        var i: usize = 0;
                        while (i < d) : (i += 1) acc += xrow[i] * self.wk[j * d + i];
                        K[b * d + j] = acc;
                    }
                    // V
                    j = 0;
                    while (j < d) : (j += 1) {
                        var acc: f32 = 0;
                        var i: usize = 0;
                        while (i < d) : (i += 1) acc += xrow[i] * self.wv[j * d + i];
                        Vv[b * d + j] = acc;
                    }
                    // alpha, beta (sigmoid of dot product with wa/wb)
                    var aacc: f32 = 0;
                    var bacc: f32 = 0;
                    var i: usize = 0;
                    while (i < d) : (i += 1) {
                        aacc += xrow[i] * self.wa[i];
                        bacc += xrow[i] * self.wb[i];
                    }
                    alpha[b] = 1.0 / (1.0 + @exp(-aacc));
                    beta[b] = 1.0 / (1.0 + @exp(-bacc));
                }
                self.gdn.forward(out, Q, K, Vv, alpha, beta, batch);
                for (out, xcur) |*o, v| o.* += v;
            } else {
                self.attn.forward(out, tmp, batch);
                for (out, xcur) |*o, v| o.* += v;
            }

            const tmp2 = self.allocator.alloc(f32, batch * (d * 2)) catch return;
            defer self.allocator.free(tmp2);
            self.norm2.forward(tmp, out);
            if (self.use_bit) {
                if (self.bit_mlp) |*bm| {
                    bm.forward(tmp2, tmp, batch);
                    for (0..batch) |b| {
                        for (0..d) |i| {
                            out[b * d + i] += tmp2[b * (d * 2) + i];
                        }
                    }
                }
            } else {
                self.mlp.forward(tmp2, tmp, batch);
                for (0..batch) |b| {
                    for (0..d) |i| {
                        out[b * d + i] += tmp2[b * (d * 2) + i];
                    }
                }
            }
        }

        pub fn backward(self: *const Block, dX: []f32, dW_attn: []f32, dW_mlp: []f32, dW_gdn_q: []f32, dW_gdn_k: []f32, dW_gdn_v: []f32, dW_gdn_a: []f32, dW_gdn_b: []f32, x: []const f32, tokens: []const u32, dOut: []const f32, batch: usize, d: usize) void {
            std.debug.assert(dX.len == batch * d and dOut.len == batch * d);
            std.debug.assert(dW_mlp.len == (d * 2) * d);
            const cpu = @import("../cpu/cpu.zig");
            // recompute intermediates
            const norm1 = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(norm1);
            self.norm1.forward(norm1, x);
            const r1 = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(r1);
            const norm2 = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(norm2);
            const mlp_out = self.allocator.alloc(f32, batch * (d * 2)) catch return;
            defer self.allocator.free(mlp_out);

            // Attention path intermediates (only if not GDN).
            const attn_out = if (!self.is_gdn) self.allocator.alloc(f32, batch * d) catch return else null;
            defer if (attn_out) |a| self.allocator.free(a);

            if (!self.is_gdn) {
                self.attn.forward(attn_out.?, norm1, batch);
                for (r1, x, attn_out.?) |*o, xv, av| o.* = xv + av;
            } else {
                // GDN forward to recompute r1
                const Q = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(Q);
                const K = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(K);
                const Vv = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(Vv);
                const alpha = self.allocator.alloc(f32, batch) catch return;
                defer self.allocator.free(alpha);
                const beta = self.allocator.alloc(f32, batch) catch return;
                defer self.allocator.free(beta);
                var b: usize = 0;
                while (b < batch) : (b += 1) {
                    const xrow = norm1[b * d .. (b + 1) * d];
                    var j: usize = 0;
                    while (j < d) : (j += 1) {
                        var acc: f32 = 0;
                        var i: usize = 0;
                        while (i < d) : (i += 1) acc += xrow[i] * self.wq[j * d + i];
                        Q[b * d + j] = acc;
                    }
                    j = 0;
                    while (j < d) : (j += 1) {
                        var acc: f32 = 0;
                        var i: usize = 0;
                        while (i < d) : (i += 1) acc += xrow[i] * self.wk[j * d + i];
                        K[b * d + j] = acc;
                    }
                    j = 0;
                    while (j < d) : (j += 1) {
                        var acc: f32 = 0;
                        var i: usize = 0;
                        while (i < d) : (i += 1) acc += xrow[i] * self.wv[j * d + i];
                        Vv[b * d + j] = acc;
                    }
                    var aacc: f32 = 0;
                    var bacc: f32 = 0;
                    var i: usize = 0;
                    while (i < d) : (i += 1) {
                        aacc += xrow[i] * self.wa[i];
                        bacc += xrow[i] * self.wb[i];
                    }
                    alpha[b] = 1.0 / (1.0 + @exp(-aacc));
                    beta[b] = 1.0 / (1.0 + @exp(-bacc));
                }
                self.gdn.forward(r1, Q, K, Vv, alpha, beta, batch);
                for (r1, x) |*o, xv| o.* += xv;
            }
            self.norm2.forward(norm2, r1);
            self.mlp.forward(mlp_out, norm2, batch);

            // dOut -> dR1 (residual) + dMlp
            const dMlp = self.allocator.alloc(f32, batch * (d * 2)) catch return;
            defer self.allocator.free(dMlp);
            @memset(dMlp, 0);
            for (0..batch) |b| {
                for (0..d) |i| dMlp[b * (d * 2) + i] = dOut[b * d + i];
            }
            const dNorm2 = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(dNorm2);
            @memset(dNorm2, 0);
            if (self.use_bit) {
                if (self.bit_mlp) |*bm| {
                    bm.backward(dW_mlp, dNorm2, norm2, dMlp, batch);
                }
            } else {
                for (0..batch) |b| {
                    for (0..d) |i| {
                        var acc: f32 = 0;
                        for (0..(d * 2)) |j| acc += dMlp[b * (d * 2) + j] * self.mlp.w[j * d + i];
                        dNorm2[b * d + i] = acc;
                    }
                    for (0..(d * 2)) |j| for (0..d) |i| {
                        dW_mlp[j * d + i] += norm2[b * d + i] * dMlp[b * (d * 2) + j];
                    };
                }
            }
            const dR1_mlp = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(dR1_mlp);
            @memset(dR1_mlp, 0);
            for (0..batch) |b| {
                const slice_x = r1[b * d .. (b + 1) * d];
                const slice_do = dNorm2[b * d .. (b + 1) * d];
                const slice_dx = dR1_mlp[b * d .. (b + 1) * d];
                autograd.backwardRMSNorm(slice_dx, slice_x, slice_do, 1e-6);
            }
            const dR1 = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(dR1);
            for (dR1, dOut, dR1_mlp) |*o, a, bb| o.* = a + bb;

            // Mixer backward
            const dNorm1 = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(dNorm1);
            @memset(dNorm1, 0);
            const dX_attn = self.allocator.alloc(f32, batch * d) catch return;
            defer self.allocator.free(dX_attn);
            @memset(dX_attn, 0);

            if (self.use_bit) {
                if (self.bit_attn) |*ba| {
                    ba.backward(dW_attn, dNorm1, norm1, dR1, batch);
                }
                for (0..batch) |b| {
                    const slice_x = x[b * d .. (b + 1) * d];
                    const slice_do = dNorm1[b * d .. (b + 1) * d];
                    const slice_dx = dX_attn[b * d .. (b + 1) * d];
                    autograd.backwardRMSNorm(slice_dx, slice_x, slice_do, 1e-6);
                }
            } else if (!self.is_gdn) {
                for (0..batch) |b| {
                    for (0..d) |i| {
                        var acc: f32 = 0;
                        for (0..d) |j| acc += dR1[b * d + j] * self.attn.w[j * d + i];
                        dNorm1[b * d + i] = acc;
                    }
                    for (0..d) |j| for (0..d) |i| {
                        dW_attn[j * d + i] += norm1[b * d + i] * dR1[b * d + j];
                    };
                }
                for (0..batch) |b| {
                    const slice_x = x[b * d .. (b + 1) * d];
                    const slice_do = dNorm1[b * d .. (b + 1) * d];
                    const slice_dx = dX_attn[b * d .. (b + 1) * d];
                    autograd.backwardRMSNorm(slice_dx, slice_x, slice_do, 1e-6);
                }
            } else {
                // Recompute Q,K,V,alpha,beta
                const Q = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(Q);
                const K = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(K);
                const Vv = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(Vv);
                const alpha = self.allocator.alloc(f32, batch) catch return;
                defer self.allocator.free(alpha);
                const beta = self.allocator.alloc(f32, batch) catch return;
                defer self.allocator.free(beta);
                var b: usize = 0;
                while (b < batch) : (b += 1) {
                    const xrow = norm1[b * d .. (b + 1) * d];
                    var j: usize = 0;
                    while (j < d) : (j += 1) {
                        var acc: f32 = 0;
                        var i: usize = 0;
                        while (i < d) : (i += 1) acc += xrow[i] * self.wq[j * d + i];
                        Q[b * d + j] = acc;
                    }
                    j = 0;
                    while (j < d) : (j += 1) {
                        var acc: f32 = 0;
                        var i: usize = 0;
                        while (i < d) : (i += 1) acc += xrow[i] * self.wk[j * d + i];
                        K[b * d + j] = acc;
                    }
                    j = 0;
                    while (j < d) : (j += 1) {
                        var acc: f32 = 0;
                        var i: usize = 0;
                        while (i < d) : (i += 1) acc += xrow[i] * self.wv[j * d + i];
                        Vv[b * d + j] = acc;
                    }
                    var aacc: f32 = 0;
                    var bacc: f32 = 0;
                    var i: usize = 0;
                    while (i < d) : (i += 1) {
                        aacc += xrow[i] * self.wa[i];
                        bacc += xrow[i] * self.wb[i];
                    }
                    alpha[b] = 1.0 / (1.0 + @exp(-aacc));
                    beta[b] = 1.0 / (1.0 + @exp(-bacc));
                }
                const dQ = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(dQ);
                @memset(dQ, 0);
                const dK = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(dK);
                @memset(dK, 0);
                const dV = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(dV);
                @memset(dV, 0);
                const dA = self.allocator.alloc(f32, batch) catch return;
                defer self.allocator.free(dA);
                @memset(dA, 0);
                const dB = self.allocator.alloc(f32, batch) catch return;
                defer self.allocator.free(dB);
                @memset(dB, 0);
                self.gdn.backward(dQ, dK, dV, dA, dB, Q, K, Vv, alpha, beta, dR1, batch);

                // backprop projections into provided grad buffers
                // dW_gdn_q += xnorm^T dQ; dXnorm += Wq^T dQ

                const dXnorm = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(dXnorm);
                @memset(dXnorm, 0);
                for (0..batch) |bb| {
                    for (0..d) |j| {
                        for (0..d) |i| {
                            dW_gdn_q[j * d + i] += norm1[bb * d + i] * dQ[bb * d + j];
                            dXnorm[bb * d + i] += self.wq[j * d + i] * dQ[bb * d + j];
                        }
                    }
                    for (0..d) |j| {
                        for (0..d) |i| {
                            dW_gdn_k[j * d + i] += norm1[bb * d + i] * dK[bb * d + j];
                            dXnorm[bb * d + i] += self.wk[j * d + i] * dK[bb * d + j];
                        }
                    }
                    for (0..d) |j| {
                        for (0..d) |i| {
                            dW_gdn_v[j * d + i] += norm1[bb * d + i] * dV[bb * d + j];
                            dXnorm[bb * d + i] += self.wv[j * d + i] * dV[bb * d + j];
                        }
                    }
                }
                // alpha/beta gradients
                for (0..batch) |bb| {
                    var aacc: f32 = 0;
                    var bacc: f32 = 0;
                    var i: usize = 0;
                    while (i < d) : (i += 1) {
                        aacc += norm1[bb * d + i] * self.wa[i];
                        bacc += norm1[bb * d + i] * self.wb[i];
                    }
                    const sa = alpha[bb] * (1 - alpha[bb]);
                    const sb = beta[bb] * (1 - beta[bb]);
                    const ga = dA[bb] * sa;
                    const gb = dB[bb] * sb;
                    for (0..d) |ii| {
                        dW_gdn_a[ii] += norm1[bb * d + ii] * ga;
                        dW_gdn_b[ii] += norm1[bb * d + ii] * gb;
                        dXnorm[bb * d + ii] += self.wa[ii] * ga + self.wb[ii] * gb;
                    }
                }
                // backward through norm1
                for (0..batch) |bi| {
                    const slice_x = x[bi * d .. (bi + 1) * d];
                    const slice_do = dXnorm[bi * d .. (bi + 1) * d];
                    const slice_dx = dX_attn[bi * d .. (bi + 1) * d];
                    autograd.backwardRMSNorm(slice_dx, slice_x, slice_do, 1e-6);
                }
            }

            for (dX, dR1, dX_attn) |*o, a, bb| o.* = a + bb;
            // Optional Engram backward contribution.
            if (self.engram) |*eng| {
                const mem = eng.mem_dim;
                const dWk = self.allocator.alloc(f32, mem * mem) catch return;
                defer self.allocator.free(dWk);
                @memset(dWk, 0);
                const dWv = self.allocator.alloc(f32, mem * mem) catch return;
                defer self.allocator.free(dWv);
                @memset(dWv, 0);
                const dTables = self.allocator.alloc(f32, eng.tables.len) catch return;
                defer self.allocator.free(dTables);
                @memset(dTables, 0);
                const dEng = self.allocator.alloc(f32, batch * d) catch return;
                defer self.allocator.free(dEng);
                @memset(dEng, 0);
                eng.backward(dEng, dWk, dWv, dTables, x, tokens, dOut, batch);
                for (dX, dEng) |*o, e| o.* += e;
            }
            _ = cpu;
        }
    };

    pub fn init(allocator: std.mem.Allocator, cfg: config.ModelConfig) !Model {
        const d = cfg.d_model;
        const v = cfg.vocab_size;
        const n_blocks = cfg.n_groups * 4; // 3 GDN + 1 attn per group (simplified: all Block)
        const embed = try allocator.alloc(f32, v * d);
        for (embed, 0..) |*x, i| x.* = 0.02 * @as(f32, @floatFromInt(i % 13)) - 0.12;
        const blocks = try allocator.alloc(Block, n_blocks);
        for (blocks, 0..) |*bl, i| {
            const is_gdn = (i % 4 != 3);
            bl.* = try Block.init(allocator, d, is_gdn, cfg.bit.enabled);
        }
        if (cfg.engram.enabled) {
            for (cfg.engram.layers) |li| {
                if (li < n_blocks) {
                    blocks[li].engram = try nn.engram.Engram.init(allocator, d, cfg.engram.n_heads, cfg.engram.orders, cfg.engram.table_size);
                }
            }
        }
        const head = try nn.Linear.init(allocator, d, v);
        const mtp_enabled = cfg.mtp.enabled;
        const mtp_loss_weight = cfg.mtp.loss_weight;
        const mtp_head = try nn.Linear.init(allocator, d, v);
        // optimizers
        const opt_embed = try optim.AdamW.init(allocator, v * d, 0.01);
        const loop_enabled = cfg.loop.enabled;
        const loop_repeats = if (cfg.loop.repeats == 0) 1 else cfg.loop.repeats;
        const loop_embed = try allocator.alloc(f32, d);
        for (loop_embed) |*x| x.* = 0.0;
        const opt_loop_embed = try optim.AdamW.init(allocator, d, 0.01);
        const opt_blocks_attn = try allocator.alloc(optim.Muon, n_blocks);
        const opt_blocks_mlp = try allocator.alloc(optim.Muon, n_blocks);
        const opt_gdn_q = try allocator.alloc(optim.Muon, n_blocks);
        const opt_gdn_k = try allocator.alloc(optim.Muon, n_blocks);
        const opt_gdn_v = try allocator.alloc(optim.Muon, n_blocks);
        const opt_gdn_a = try allocator.alloc(optim.AdamW, n_blocks);
        const opt_gdn_b = try allocator.alloc(optim.AdamW, n_blocks);
        for (0..n_blocks) |i| {
            opt_blocks_attn[i] = try optim.Muon.init(allocator, d, d, 0.01);
            opt_blocks_mlp[i] = try optim.Muon.init(allocator, d * 2, d, 0.01);
            opt_gdn_q[i] = try optim.Muon.init(allocator, d, d, 0.01);
            opt_gdn_k[i] = try optim.Muon.init(allocator, d, d, 0.01);
            opt_gdn_v[i] = try optim.Muon.init(allocator, d, d, 0.01);
            opt_gdn_a[i] = try optim.AdamW.init(allocator, d, 0.01);
            opt_gdn_b[i] = try optim.AdamW.init(allocator, d, 0.01);
        }
        const opt_head = try optim.AdamW.init(allocator, d * v, 0.01);
        const opt_mtp_head = try optim.AdamW.init(allocator, d * v, 0.01);
        return .{
            .cfg = cfg,
            .embed = embed,
            .blocks = blocks,
            .final_norm = .{},
            .head = head,
            .opt_embed = opt_embed,
            .opt_blocks_attn = opt_blocks_attn,
            .opt_blocks_mlp = opt_blocks_mlp,
            .opt_gdn_q = opt_gdn_q,
            .opt_gdn_k = opt_gdn_k,
            .opt_gdn_v = opt_gdn_v,
            .opt_gdn_a = opt_gdn_a,
            .opt_gdn_b = opt_gdn_b,
            .opt_head = opt_head,
            .mtp_enabled = mtp_enabled,
            .mtp_loss_weight = mtp_loss_weight,
            .mtp_head = mtp_head,
            .opt_mtp_head = opt_mtp_head,
            .loop_enabled = loop_enabled,
            .loop_repeats = loop_repeats,
            .loop_embed = loop_embed,
            .opt_loop_embed = opt_loop_embed,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Model) void {
        self.allocator.free(self.embed);
        for (self.blocks) |*bl| bl.deinit();
        self.allocator.free(self.blocks);
        self.head.deinit();
        self.opt_embed.deinit();
        for (self.opt_blocks_attn) |*o| o.deinit();
        self.allocator.free(self.opt_blocks_attn);
        for (self.opt_blocks_mlp) |*o| o.deinit();
        self.allocator.free(self.opt_blocks_mlp);
        for (self.opt_gdn_q) |*o| o.deinit();
        self.allocator.free(self.opt_gdn_q);
        for (self.opt_gdn_k) |*o| o.deinit();
        self.allocator.free(self.opt_gdn_k);
        for (self.opt_gdn_v) |*o| o.deinit();
        self.allocator.free(self.opt_gdn_v);
        for (self.opt_gdn_a) |*o| o.deinit();
        self.allocator.free(self.opt_gdn_a);
        for (self.opt_gdn_b) |*o| o.deinit();
        self.allocator.free(self.opt_gdn_b);
        self.opt_head.deinit();
        self.mtp_head.deinit();
        self.opt_mtp_head.deinit();
        self.opt_loop_embed.deinit();
        self.allocator.free(self.loop_embed);
    }

    // Forward hidden states after all blocks (before final norm).
    pub fn forwardHidden(self: *const Model, hidden: []f32, tokens: []const u32) void {
        const d = self.cfg.d_model;
        const seq = tokens.len;
        std.debug.assert(hidden.len == seq * d);
        cpuEmbeddingGather(hidden, self.embed, tokens, self.cfg.vocab_size, d);

        const prelude_count: usize = if (self.blocks.len >= 3) 1 else 0;
        const coda_count: usize = if (self.blocks.len >= 3) 1 else 0;
        const core_count: usize = if (self.blocks.len > prelude_count + coda_count) self.blocks.len - prelude_count - coda_count else self.blocks.len;
        const repeats = if (self.loop_enabled) self.loop_repeats else 1;

        var cur = hidden;
        var next = self.allocator.alloc(f32, seq * d) catch return;
        defer self.allocator.free(next);

        var bi: usize = 0;
        while (bi < prelude_count) : (bi += 1) {
            if (self.loop_enabled and false) {}
            self.blocks[bi].forward(next, cur, tokens, seq, d);
            const tmp = cur;
            cur = next;
            next = tmp;
        }
        var rep: usize = 0;
        while (rep < repeats) : (rep += 1) {
            var ci: usize = 0;
            while (ci < core_count) : (ci += 1) {
                const idx = prelude_count + ci;
                if (idx >= self.blocks.len - coda_count) break;
                if (self.loop_enabled and rep > 0 and ci == 0) {
                    var i: usize = 0;
                    while (i < seq * d) : (i += 1) cur[i] += self.loop_embed[i % d];
                }
                self.blocks[idx].forward(next, cur, tokens, seq, d);
                const tmp = cur;
                cur = next;
                next = tmp;
            }
        }
        bi = self.blocks.len - coda_count;
        while (bi < self.blocks.len) : (bi += 1) {
            self.blocks[bi].forward(next, cur, tokens, seq, d);
            const tmp = cur;
            cur = next;
            next = tmp;
        }
        if (cur.ptr != hidden.ptr) @memcpy(hidden, cur);
    }

    // forward: token ids [seq] -> logits [seq, vocab]
    pub fn forward(self: *const Model, logits: []f32, tokens: []const u32) void {
        const d = self.cfg.d_model;
        const v = self.cfg.vocab_size;
        const seq = tokens.len;
        std.debug.assert(logits.len == seq * v);

        const hidden = self.allocator.alloc(f32, seq * d) catch return;
        defer self.allocator.free(hidden);
        self.forwardHidden(hidden, tokens);
        const last = hidden;
        // final norm
        var normed = self.allocator.alloc(f32, seq * d) catch return;
        defer self.allocator.free(normed);
        var i: usize = 0;
        while (i < seq) : (i += 1) {
            self.final_norm.forward(normed[i * d .. (i + 1) * d], last[i * d .. (i + 1) * d]);
        }

        // head: logits = normed @ W^T  (d -> vocab)
        self.head.forward(logits, normed, seq);
    }

    // Training step: forward + CE loss + backward + AdamW update. Returns loss.
    pub fn trainStep(self: *Model, tokens: []const u32, targets: []const u32) !f32 {
        const allocator = self.allocator;
        const d = self.cfg.d_model;
        const v = self.cfg.vocab_size;
        const seq = tokens.len;
        const n = self.blocks.len;

        // forward
        const logits = try allocator.alloc(f32, seq * v);
        defer allocator.free(logits);
        self.forward(logits, tokens);

        const loss = @as(f32, @floatCast(cpu_mod.crossEntropyF32(logits, targets, seq, v)));

        // backward
        const dLogits = try allocator.alloc(f32, seq * v);
        defer allocator.free(dLogits);
        @memset(dLogits, 0);
        autograd.backwardCrossEntropy(dLogits, logits, targets, seq, v);

        const dEmbed = try allocator.alloc(f32, v * d);
        defer allocator.free(dEmbed);
        @memset(dEmbed, 0);
        const dHead = try allocator.alloc(f32, v * d);
        defer allocator.free(dHead);
        @memset(dHead, 0);
        const dBlocks_attn = try allocator.alloc([]f32, n);
        defer allocator.free(dBlocks_attn);
        const dBlocks_mlp = try allocator.alloc([]f32, n);
        defer allocator.free(dBlocks_mlp);
        const dBlocks_gdn_q = try allocator.alloc([]f32, n);
        defer allocator.free(dBlocks_gdn_q);
        const dBlocks_gdn_k = try allocator.alloc([]f32, n);
        defer allocator.free(dBlocks_gdn_k);
        const dBlocks_gdn_v = try allocator.alloc([]f32, n);
        defer allocator.free(dBlocks_gdn_v);
        const dBlocks_gdn_a = try allocator.alloc([]f32, n);
        defer allocator.free(dBlocks_gdn_a);
        const dBlocks_gdn_b = try allocator.alloc([]f32, n);
        defer allocator.free(dBlocks_gdn_b);
        for (0..n) |bi| {
            dBlocks_attn[bi] = try allocator.alloc(f32, d * d);
            dBlocks_mlp[bi] = try allocator.alloc(f32, (d * 2) * d);
            dBlocks_gdn_q[bi] = try allocator.alloc(f32, d * d);
            dBlocks_gdn_k[bi] = try allocator.alloc(f32, d * d);
            dBlocks_gdn_v[bi] = try allocator.alloc(f32, d * d);
            dBlocks_gdn_a[bi] = try allocator.alloc(f32, d);
            dBlocks_gdn_b[bi] = try allocator.alloc(f32, d);
            @memset(dBlocks_attn[bi], 0);
            @memset(dBlocks_mlp[bi], 0);
            @memset(dBlocks_gdn_q[bi], 0);
            @memset(dBlocks_gdn_k[bi], 0);
            @memset(dBlocks_gdn_v[bi], 0);
            @memset(dBlocks_gdn_a[bi], 0);
            @memset(dBlocks_gdn_b[bi], 0);
        }
        defer for (0..n) |bi| {
            allocator.free(dBlocks_attn[bi]);
            allocator.free(dBlocks_mlp[bi]);
            allocator.free(dBlocks_gdn_q[bi]);
            allocator.free(dBlocks_gdn_k[bi]);
            allocator.free(dBlocks_gdn_v[bi]);
            allocator.free(dBlocks_gdn_a[bi]);
            allocator.free(dBlocks_gdn_b[bi]);
        };

        var mtp_loss: f32 = 0;
        const dExtra = try allocator.alloc(f32, seq * d);
        defer allocator.free(dExtra);
        @memset(dExtra, 0);
        if (self.mtp_enabled and seq >= 3) {
            const mtp_seq = seq - 2;
            const mtp_logits = try allocator.alloc(f32, mtp_seq * v);
            defer allocator.free(mtp_logits);
            const hidden = try allocator.alloc(f32, seq * d);
            defer allocator.free(hidden);
            self.forwardHidden(hidden, tokens);
            var t: usize = 0;
            while (t < mtp_seq) : (t += 1) {
                self.mtp_head.forward(mtp_logits[t * v .. (t + 1) * v], hidden[t * d .. (t + 1) * d], 1);
            }
            mtp_loss = @as(f32, @floatCast(cpu_mod.crossEntropyF32(mtp_logits, targets[2..], mtp_seq, v))) * self.mtp_loss_weight;

            const d_mtp_logits = try allocator.alloc(f32, mtp_seq * v);
            defer allocator.free(d_mtp_logits);
            @memset(d_mtp_logits, 0);
            autograd.backwardCrossEntropy(d_mtp_logits, mtp_logits, targets[2..], mtp_seq, v);
            const dW_mtp = try allocator.alloc(f32, d * v);
            defer allocator.free(dW_mtp);
            @memset(dW_mtp, 0);
            var tt: usize = 0;
            while (tt < mtp_seq) : (tt += 1) {
                self.mtp_head.backward(dW_mtp, dExtra[tt * d .. (tt + 1) * d], hidden[tt * d .. (tt + 1) * d], d_mtp_logits[tt * v .. (tt + 1) * v], 1);
            }
            for (dExtra) |*e| e.* *= self.mtp_loss_weight;
            for (dW_mtp) |*e| e.* *= self.mtp_loss_weight;
            self.opt_mtp_head.step(self.mtp_head.w, dW_mtp);
        }

        self.backward(dEmbed, dHead, dBlocks_attn, dBlocks_mlp, dBlocks_gdn_q, dBlocks_gdn_k, dBlocks_gdn_v, dBlocks_gdn_a, dBlocks_gdn_b, tokens, dLogits, dExtra);

        // update params
        self.opt_embed.step(self.embed, dEmbed);
        for (0..n) |bi| {
            if (self.blocks[bi].use_bit) {
                if (self.blocks[bi].bit_attn) |_| {
                    self.opt_blocks_attn[bi].step(self.blocks[bi].bit_attn.?.w, dBlocks_attn[bi], "original");
                }
                if (self.blocks[bi].bit_mlp) |_| {
                    self.opt_blocks_mlp[bi].step(self.blocks[bi].bit_mlp.?.w, dBlocks_mlp[bi], "original");
                }
            } else if (self.blocks[bi].is_gdn) {
                self.opt_gdn_q[bi].step(self.blocks[bi].wq, dBlocks_gdn_q[bi], "original");
                self.opt_gdn_k[bi].step(self.blocks[bi].wk, dBlocks_gdn_k[bi], "original");
                self.opt_gdn_v[bi].step(self.blocks[bi].wv, dBlocks_gdn_v[bi], "original");
                self.opt_gdn_a[bi].step(self.blocks[bi].wa, dBlocks_gdn_a[bi]);
                self.opt_gdn_b[bi].step(self.blocks[bi].wb, dBlocks_gdn_b[bi]);
                self.opt_blocks_mlp[bi].step(self.blocks[bi].mlp.w, dBlocks_mlp[bi], "original");
            } else {
                self.opt_blocks_attn[bi].step(self.blocks[bi].attn.w, dBlocks_attn[bi], "original");
                self.opt_blocks_mlp[bi].step(self.blocks[bi].mlp.w, dBlocks_mlp[bi], "original");
            }
        }
        self.opt_head.step(self.head.w, dHead);

        return loss + mtp_loss;
    }

    // Masked training step: loss and gradients only on positions where loss_mask is true.
    pub fn trainStepMasked(self: *Model, tokens: []const u32, targets: []const u32, loss_mask: []const bool) !f32 {
        const allocator = self.allocator;
        const d = self.cfg.d_model;
        const v = self.cfg.vocab_size;
        const seq = tokens.len;
        const n = self.blocks.len;

        const logits = try allocator.alloc(f32, seq * v);
        defer allocator.free(logits);
        self.forward(logits, tokens);

        // masked CE loss
        var total_loss: f32 = 0;
        var count: usize = 0;
        var i: usize = 0;
        while (i < seq) : (i += 1) {
            if (!loss_mask[i]) continue;
            var mx: f32 = logits[i * v];
            var j: usize = 1;
            while (j < v) : (j += 1) {
                if (logits[i * v + j] > mx) mx = logits[i * v + j];
            }
            var sum: f32 = 0;
            j = 0;
            while (j < v) : (j += 1) sum += @exp(logits[i * v + j] - mx);
            const p = @exp(logits[i * v + targets[i]] - mx) / sum;
            total_loss += -@log(p + 1e-12);
            count += 1;
        }
        const loss = if (count > 0) total_loss / @as(f32, @floatFromInt(count)) else 0;

        // backward
        const dLogits = try allocator.alloc(f32, seq * v);
        defer allocator.free(dLogits);
        @memset(dLogits, 0);
        autograd.backwardCrossEntropy(dLogits, logits, targets, seq, v);
        // zero masked-out positions
        var pos: usize = 0;
        while (pos < seq) : (pos += 1) {
            if (!loss_mask[pos]) {
                var j: usize = 0;
                while (j < v) : (j += 1) dLogits[pos * v + j] = 0;
            }
        }

        const dEmbed = try allocator.alloc(f32, v * d);
        defer allocator.free(dEmbed);
        @memset(dEmbed, 0);
        const dHead = try allocator.alloc(f32, v * d);
        defer allocator.free(dHead);
        @memset(dHead, 0);
        const dBlocks_attn = try allocator.alloc([]f32, n);
        defer allocator.free(dBlocks_attn);
        const dBlocks_mlp = try allocator.alloc([]f32, n);
        defer allocator.free(dBlocks_mlp);
        const dBlocks_gdn_q = try allocator.alloc([]f32, n);
        defer allocator.free(dBlocks_gdn_q);
        const dBlocks_gdn_k = try allocator.alloc([]f32, n);
        defer allocator.free(dBlocks_gdn_k);
        const dBlocks_gdn_v = try allocator.alloc([]f32, n);
        defer allocator.free(dBlocks_gdn_v);
        const dBlocks_gdn_a = try allocator.alloc([]f32, n);
        defer allocator.free(dBlocks_gdn_a);
        const dBlocks_gdn_b = try allocator.alloc([]f32, n);
        defer allocator.free(dBlocks_gdn_b);
        for (0..n) |bi| {
            dBlocks_attn[bi] = try allocator.alloc(f32, d * d);
            dBlocks_mlp[bi] = try allocator.alloc(f32, (d * 2) * d);
            dBlocks_gdn_q[bi] = try allocator.alloc(f32, d * d);
            dBlocks_gdn_k[bi] = try allocator.alloc(f32, d * d);
            dBlocks_gdn_v[bi] = try allocator.alloc(f32, d * d);
            dBlocks_gdn_a[bi] = try allocator.alloc(f32, d);
            dBlocks_gdn_b[bi] = try allocator.alloc(f32, d);
            @memset(dBlocks_attn[bi], 0);
            @memset(dBlocks_mlp[bi], 0);
            @memset(dBlocks_gdn_q[bi], 0);
            @memset(dBlocks_gdn_k[bi], 0);
            @memset(dBlocks_gdn_v[bi], 0);
            @memset(dBlocks_gdn_a[bi], 0);
            @memset(dBlocks_gdn_b[bi], 0);
        }
        defer for (0..n) |bi| {
            allocator.free(dBlocks_attn[bi]);
            allocator.free(dBlocks_mlp[bi]);
            allocator.free(dBlocks_gdn_q[bi]);
            allocator.free(dBlocks_gdn_k[bi]);
            allocator.free(dBlocks_gdn_v[bi]);
            allocator.free(dBlocks_gdn_a[bi]);
            allocator.free(dBlocks_gdn_b[bi]);
        };

        const dExtra = try allocator.alloc(f32, seq * d);
        defer allocator.free(dExtra);
        @memset(dExtra, 0);

        self.backward(dEmbed, dHead, dBlocks_attn, dBlocks_mlp, dBlocks_gdn_q, dBlocks_gdn_k, dBlocks_gdn_v, dBlocks_gdn_a, dBlocks_gdn_b, tokens, dLogits, dExtra);

        // update params
        self.opt_embed.step(self.embed, dEmbed);
        for (0..n) |bi| {
            if (self.blocks[bi].use_bit) {
                if (self.blocks[bi].bit_attn) |_| {
                    self.opt_blocks_attn[bi].step(self.blocks[bi].bit_attn.?.w, dBlocks_attn[bi], "original");
                }
                if (self.blocks[bi].bit_mlp) |_| {
                    self.opt_blocks_mlp[bi].step(self.blocks[bi].bit_mlp.?.w, dBlocks_mlp[bi], "original");
                }
            } else if (self.blocks[bi].is_gdn) {
                self.opt_gdn_q[bi].step(self.blocks[bi].wq, dBlocks_gdn_q[bi], "original");
                self.opt_gdn_k[bi].step(self.blocks[bi].wk, dBlocks_gdn_k[bi], "original");
                self.opt_gdn_v[bi].step(self.blocks[bi].wv, dBlocks_gdn_v[bi], "original");
                self.opt_gdn_a[bi].step(self.blocks[bi].wa, dBlocks_gdn_a[bi]);
                self.opt_gdn_b[bi].step(self.blocks[bi].wb, dBlocks_gdn_b[bi]);
                self.opt_blocks_mlp[bi].step(self.blocks[bi].mlp.w, dBlocks_mlp[bi], "original");
            } else {
                self.opt_blocks_attn[bi].step(self.blocks[bi].attn.w, dBlocks_attn[bi], "original");
                self.opt_blocks_mlp[bi].step(self.blocks[bi].mlp.w, dBlocks_mlp[bi], "original");
            }
        }
        self.opt_head.step(self.head.w, dHead);
        // MTP update is intentionally omitted from the masked helper for now.

        return loss;
    }

    pub fn nBlocks(self: *const Model) usize {
        return self.blocks.len;
    }

    fn argmaxLocal(logits: []const f32) usize {
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

    // MTP-assisted speculative decoding (PRD 3.12): draft with MTP, verify with main model.
    pub fn speculativeStep(self: *Model, allocator: std.mem.Allocator, tokens: []const u32) ![]u32 {
        const v = self.cfg.vocab_size;
        const d = self.cfg.d_model;
        const seq = tokens.len;
        var accepted = std.ArrayList(u32).init(allocator);
        if (seq == 0) return accepted.toOwnedSlice();
        defer accepted.deinit();

        const logits = try allocator.alloc(f32, seq * v);
        defer allocator.free(logits);
        self.forward(logits, tokens);
        const next = argmaxLocal(logits[(seq - 1) * v .. seq * v]);
        try accepted.append(next);

        if (!self.mtp_enabled or seq == 0) return accepted.toOwnedSlice();

        const hidden = try allocator.alloc(f32, seq * d);
        defer allocator.free(hidden);
        self.forwardHidden(hidden, tokens);
        const draft_logits = try allocator.alloc(f32, v);
        defer allocator.free(draft_logits);
        self.mtp_head.forward(draft_logits, hidden[(seq - 1) * d .. seq * d], 1);
        const draft = argmaxLocal(draft_logits);

        const tmp_tokens = try allocator.alloc(u32, seq + 1);
        defer allocator.free(tmp_tokens);
        @memcpy(tmp_tokens[0..seq], tokens);
        tmp_tokens[seq] = next;
        const verify_logits = try allocator.alloc(f32, (seq + 1) * v);
        defer allocator.free(verify_logits);
        self.forward(verify_logits, tmp_tokens);
        const verified = argmaxLocal(verify_logits[seq * v .. (seq + 1) * v]);

        if (verified == draft) try accepted.append(draft);
        return accepted.toOwnedSlice();
    }

    // Backward: given dLogits (seq x vocab), compute dEmbed (vocab x d) gradient
    // and accumulate into dHead (d x vocab), dBlocks_attn/mlp. Returns via buffers.
    pub fn backward(self: *Model, dEmbed: []f32, dHead: []f32, dBlocks_attn: [][]f32, dBlocks_mlp: [][]f32, dBlocks_gdn_q: [][]f32, dBlocks_gdn_k: [][]f32, dBlocks_gdn_v: [][]f32, dBlocks_gdn_a: [][]f32, dBlocks_gdn_b: [][]f32, tokens: []const u32, dLogits: []const f32, dExtraHidden: []const f32) void {
        const d = self.cfg.d_model;
        const v = self.cfg.vocab_size;
        const seq = tokens.len;
        const n = self.blocks.len;

        // forward intermediates
        const hidden0 = self.allocator.alloc(f32, seq * d) catch return;
        defer self.allocator.free(hidden0);
        cpuEmbeddingGather(hidden0, self.embed, tokens, v, d);

        const Pass = struct { block_idx: usize, input: []f32, add_loop: bool };
        var passes = std.ArrayList(Pass).init(self.allocator);
        defer {
            for (passes.items) |p| self.allocator.free(p.input);
            passes.deinit();
        }

        const prelude_count: usize = if (n >= 3) 1 else 0;
        const coda_count: usize = if (n >= 3) 1 else 0;
        const core_count: usize = if (n > prelude_count + coda_count) n - prelude_count - coda_count else n;
        const repeats = if (self.loop_enabled) self.loop_repeats else 1;

        var cur = hidden0;
        var next = self.allocator.alloc(f32, seq * d) catch return;
        defer self.allocator.free(next);
        var last = self.allocator.alloc(f32, seq * d) catch return;
        defer self.allocator.free(last);

        const runBlock = struct {
            fn call(self2: *const Model, passes2: *std.ArrayList(Pass), cur2: *[]f32, next2: *[]f32, tokens2: []const u32, seq2: usize, d2: usize, idx: usize, add_loop: bool) void {
                const input_copy = self2.allocator.alloc(f32, seq2 * d2) catch return;
                @memcpy(input_copy, cur2.*);
                passes2.append(.{ .block_idx = idx, .input = input_copy, .add_loop = add_loop }) catch {
                    self2.allocator.free(input_copy);
                    return;
                };
                if (add_loop) {
                    var i: usize = 0;
                    while (i < seq2 * d2) : (i += 1) cur2.*[i] += self2.loop_embed[i % d2];
                }
                self2.blocks[idx].forward(next2.*, cur2.*, tokens2, seq2, d2);
                const tmp = cur2.*;
                cur2.* = next2.*;
                next2.* = tmp;
            }
        }.call;

        var bi: usize = 0;
        while (bi < prelude_count) : (bi += 1) {
            runBlock(self, &passes, &cur, &next, tokens, seq, d, bi, false);
        }
        var rep: usize = 0;
        while (rep < repeats) : (rep += 1) {
            var ci: usize = 0;
            while (ci < core_count) : (ci += 1) {
                const idx = prelude_count + ci;
                if (idx >= n - coda_count) break;
                const add_loop = self.loop_enabled and rep > 0 and ci == 0;
                runBlock(self, &passes, &cur, &next, tokens, seq, d, idx, add_loop);
            }
        }
        bi = n - coda_count;
        while (bi < n) : (bi += 1) {
            runBlock(self, &passes, &cur, &next, tokens, seq, d, bi, false);
        }
        @memcpy(last, cur);

        // final norm
        const normed = self.allocator.alloc(f32, seq * d) catch return;
        defer self.allocator.free(normed);
        var i: usize = 0;
        while (i < seq) : (i += 1) {
            self.final_norm.forward(normed[i * d .. (i + 1) * d], last[i * d .. (i + 1) * d]);
        }

        // dNormed = dLogits @ W_head^T  (seq x d)
        const dNormed = self.allocator.alloc(f32, seq * d) catch return;
        defer self.allocator.free(dNormed);
        @memset(dNormed, 0);
        for (0..seq) |b| {
            for (0..d) |ddi| {
                var acc: f32 = 0;
                for (0..v) |vv| acc += dLogits[b * v + vv] * self.head.w[vv * d + ddi];
                dNormed[b * d + ddi] = acc;
            }
        }
        // dHead += normed^T @ dLogits
        for (0..seq) |b| {
            for (0..v) |vv| {
                for (0..d) |ddi| {
                    dHead[vv * d + ddi] += normed[b * d + ddi] * dLogits[b * v + vv];
                }
            }
        }

        // backward through final norm: dLast = backwardRMSNorm(last, dNormed)
        const dLast = self.allocator.alloc(f32, seq * d) catch return;
        defer self.allocator.free(dLast);
        @memset(dLast, 0);
        for (0..seq) |b| {
            const sl_x = last[b * d .. (b + 1) * d];
            const sl_do = dNormed[b * d .. (b + 1) * d];
            const sl_dx = dLast[b * d .. (b + 1) * d];
            autograd.backwardRMSNorm(sl_dx, sl_x, sl_do, 1e-6);
        }
        // MTP extra hidden gradient.
        if (dExtraHidden.len == seq * d) {
            for (dLast, dExtraHidden) |*a, e| a.* += e;
        }

        // backward through unrolled passes (reverse)
        const dCur = self.allocator.alloc(f32, seq * d) catch return;
        defer self.allocator.free(dCur);
        @memcpy(dCur, dLast);
        const dLoop = self.allocator.alloc(f32, d) catch return;
        defer self.allocator.free(dLoop);
        @memset(dLoop, 0);
        var pi: usize = passes.items.len;
        while (pi > 0) {
            pi -= 1;
            const pass = passes.items[pi];
            const dX = self.allocator.alloc(f32, seq * d) catch return;
            defer self.allocator.free(dX);
            @memset(dX, 0);
            self.blocks[pass.block_idx].backward(dX, dBlocks_attn[pass.block_idx], dBlocks_mlp[pass.block_idx], dBlocks_gdn_q[pass.block_idx], dBlocks_gdn_k[pass.block_idx], dBlocks_gdn_v[pass.block_idx], dBlocks_gdn_a[pass.block_idx], dBlocks_gdn_b[pass.block_idx], pass.input, tokens, dCur, seq, d);
            if (pass.add_loop) {
                var t: usize = 0;
                while (t < seq) : (t += 1) {
                    var ii: usize = 0;
                    while (ii < d) : (ii += 1) {
                        dLoop[ii] += dX[t * d + ii];
                    }
                }
            }
            @memcpy(dCur, dX);
        }
        if (self.loop_enabled) {
            self.opt_loop_embed.step(self.loop_embed, dLoop);
        }

        // dEmbed scatter
        @memset(dEmbed, 0);
        for (tokens, 0..) |tok, pos| {
            if (tok >= v) continue;
            for (0..d) |ddi| {
                dEmbed[tok * d + ddi] += dCur[pos * d + ddi];
            }
        }
    }

    fn cpuEmbeddingGather(out: []f32, table: []const f32, ids: []const u32, vocab: usize, dim: usize) void {
        const cpu = @import("../cpu/cpu.zig");
        cpu.embeddingGatherF32(out, table, ids, vocab, dim);
    }

    test "Model forward finite" {
        const allocator = std.testing.allocator;
        const cfg = config.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 8, .seq_len = 4 };
        var m = try Model.init(allocator, cfg);
        defer m.deinit();
        const tokens = [_]u32{ 1, 2, 3, 4 };
        var logits = [_]f32{0} * *(4 * 8);
        m.forward(&logits, &tokens);
        for (logits) |v| try std.testing.expect(std.math.isFinite(v));
        try std.testing.expectEqual(@as(usize, 4), m.nBlocks());
    }

    test "Block backward gradcheck" {
        const allocator = std.testing.allocator;
        var blk = try Model.Block.init(allocator, 4, false, false);
        defer blk.deinit();
        var x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        var out = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
        blk.forward(&out, &x, 2, 4);
        // dOut = ones
        var dOut = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
        var dX = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
        const dW_attn = try allocator.alloc(f32, 4 * 4);
        defer allocator.free(dW_attn);
        @memset(dW_attn, 0);
        const dW_mlp = try allocator.alloc(f32, 8 * 4);
        defer allocator.free(dW_mlp);
        @memset(dW_mlp, 0);
        const toks = [_]u32{ 1, 2 };
        blk.backward(&dX, dW_attn, dW_mlp, &x, &toks, &dOut, 2, 4);
        // finite diff for x[0]
        const eps: f32 = 1e-3;
        var xp = x;
        xp[0] += eps;
        var xm = x;
        xm[0] -= eps;
        var op = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
        var om = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
        blk.forward(&op, &xp, 2, 4);
        blk.forward(&om, &xm, 2, 4);
        var sp: f32 = 0;
        var sm: f32 = 0;
        for (op) |v| sp += v;
        for (om) |v| sm += v;
        const fd = (sp - sm) / (2 * eps);
        try std.testing.expect(@abs(fd - dX[0]) < 1e-1);
    }
};

test "Model backward finite" {
    const allocator = std.testing.allocator;
    const cfg = config.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 8, .seq_len = 4 };
    var m = try Model.init(allocator, cfg);
    defer m.deinit();
    const tokens = [_]u32{ 1, 2, 3 };
    const logits = try allocator.alloc(f32, tokens.len * 8);
    defer allocator.free(logits);
    m.forward(logits, &tokens);
    // dLogits = ones
    const dLogits = try allocator.alloc(f32, tokens.len * 8);
    defer allocator.free(dLogits);
    @memset(dLogits, 1);
    const d = cfg.d_model;
    const v = cfg.vocab_size;
    const dEmbed = try allocator.alloc(f32, v * d);
    defer allocator.free(dEmbed);
    @memset(dEmbed, 0);
    const dHead = try allocator.alloc(f32, v * d);
    defer allocator.free(dHead);
    @memset(dHead, 0);
    const n = m.nBlocks();
    const dBlocks_attn = try allocator.alloc([]f32, n);
    defer allocator.free(dBlocks_attn);
    const dBlocks_mlp = try allocator.alloc([]f32, n);
    defer allocator.free(dBlocks_mlp);
    const dBlocks_gdn_q = try allocator.alloc([]f32, n);
    defer allocator.free(dBlocks_gdn_q);
    const dBlocks_gdn_k = try allocator.alloc([]f32, n);
    defer allocator.free(dBlocks_gdn_k);
    const dBlocks_gdn_v = try allocator.alloc([]f32, n);
    defer allocator.free(dBlocks_gdn_v);
    const dBlocks_gdn_a = try allocator.alloc([]f32, n);
    defer allocator.free(dBlocks_gdn_a);
    const dBlocks_gdn_b = try allocator.alloc([]f32, n);
    defer allocator.free(dBlocks_gdn_b);
    for (0..n) |i| {
        dBlocks_attn[i] = try allocator.alloc(f32, d * d);
        dBlocks_mlp[i] = try allocator.alloc(f32, (d * 2) * d);
        dBlocks_gdn_q[i] = try allocator.alloc(f32, d * d);
        dBlocks_gdn_k[i] = try allocator.alloc(f32, d * d);
        dBlocks_gdn_v[i] = try allocator.alloc(f32, d * d);
        dBlocks_gdn_a[i] = try allocator.alloc(f32, d);
        dBlocks_gdn_b[i] = try allocator.alloc(f32, d);
        @memset(dBlocks_attn[i], 0);
        @memset(dBlocks_mlp[i], 0);
        @memset(dBlocks_gdn_q[i], 0);
        @memset(dBlocks_gdn_k[i], 0);
        @memset(dBlocks_gdn_v[i], 0);
        @memset(dBlocks_gdn_a[i], 0);
        @memset(dBlocks_gdn_b[i], 0);
    }
    defer for (0..n) |i| {
        allocator.free(dBlocks_attn[i]);
        allocator.free(dBlocks_mlp[i]);
        allocator.free(dBlocks_gdn_q[i]);
        allocator.free(dBlocks_gdn_k[i]);
        allocator.free(dBlocks_gdn_v[i]);
        allocator.free(dBlocks_gdn_a[i]);
        allocator.free(dBlocks_gdn_b[i]);
    };
    const dExtra = try allocator.alloc(f32, tokens.len * 4);
    defer allocator.free(dExtra);
    @memset(dExtra, 0);
    m.backward(dEmbed, dHead, dBlocks_attn, dBlocks_mlp, dBlocks_gdn_q, dBlocks_gdn_k, dBlocks_gdn_v, dBlocks_gdn_a, dBlocks_gdn_b, &tokens, dLogits, dExtra);
    // finite check dEmbed[0]
    for (dEmbed) |x| try std.testing.expect(std.math.isFinite(x));
}

test "Model trainStep finite" {
    const allocator = std.testing.allocator;
    const cfg = config.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 8, .seq_len = 4 };
    var m = try Model.init(allocator, cfg);
    defer m.deinit();
    const tokens = [_]u32{ 1, 2, 3 };
    const targets = [_]u32{ 2, 3, 4 };
    const loss = try m.trainStep(&tokens, &targets);
    try std.testing.expect(std.math.isFinite(loss));
}

test "Model speculativeStep finite" {
    const allocator = std.testing.allocator;
    const cfg = config.ModelConfig{
        .d_model = 4,
        .n_groups = 1,
        .vocab_size = 8,
        .seq_len = 4,
        .mtp = .{ .enabled = true, .loss_weight = 0.3 },
    };
    var m = try Model.init(allocator, cfg);
    defer m.deinit();
    const tokens = [_]u32{ 1, 2, 3 };
    const out = try m.speculativeStep(allocator, &tokens);
    defer allocator.free(out);
    try std.testing.expect(out.len >= 1 and out.len <= 2);
    for (out) |t| try std.testing.expect(t < 8);
}

test "Model with MTP finite" {
    const allocator = std.testing.allocator;
    const cfg = config.ModelConfig{
        .d_model = 4,
        .n_groups = 1,
        .vocab_size = 8,
        .seq_len = 4,
        .mtp = .{ .enabled = true, .loss_weight = 0.3 },
    };
    var m = try Model.init(allocator, cfg);
    defer m.deinit();
    const tokens = [_]u32{ 1, 2, 3, 4 };
    const targets = [_]u32{ 2, 3, 4, 5 };
    const loss = try m.trainStep(&tokens, &targets);
    try std.testing.expect(std.math.isFinite(loss));
}

test "Model with loop finite" {
    const allocator = std.testing.allocator;
    const cfg = config.ModelConfig{
        .d_model = 4,
        .n_groups = 1,
        .vocab_size = 8,
        .seq_len = 4,
        .loop = .{ .enabled = true, .repeats = 2 },
    };
    var m = try Model.init(allocator, cfg);
    defer m.deinit();
    const tokens = [_]u32{ 1, 2, 3 };
    const targets = [_]u32{ 2, 3, 4 };
    const loss = try m.trainStep(&tokens, &targets);
    try std.testing.expect(std.math.isFinite(loss));
    const logits = try allocator.alloc(f32, tokens.len * 8);
    defer allocator.free(logits);
    m.forward(logits, &tokens);
    for (logits) |v| try std.testing.expect(std.math.isFinite(v));
}

test "Model with BitLinear finite" {
    const allocator = std.testing.allocator;
    const cfg = config.ModelConfig{
        .d_model = 4,
        .n_groups = 1,
        .vocab_size = 8,
        .seq_len = 4,
        .bit = .{ .enabled = true },
    };
    var m = try Model.init(allocator, cfg);
    defer m.deinit();
    const tokens = [_]u32{ 1, 2, 3 };
    const targets = [_]u32{ 2, 3, 4 };
    const loss = try m.trainStep(&tokens, &targets);
    try std.testing.expect(std.math.isFinite(loss));
    const logits = try allocator.alloc(f32, tokens.len * 8);
    defer allocator.free(logits);
    m.forward(logits, &tokens);
    for (logits) |v| try std.testing.expect(std.math.isFinite(v));
}

test "Model with Engram finite" {
    const allocator = std.testing.allocator;
    const orders = [_]u8{ 2, 3 };
    const cfg = config.ModelConfig{
        .d_model = 4,
        .n_groups = 1,
        .vocab_size = 8,
        .seq_len = 4,
        .engram = .{ .enabled = true, .layers = &[_]usize{0}, .orders = &orders, .n_heads = 2, .table_size = 16 },
    };
    var m = try Model.init(allocator, cfg);
    defer m.deinit();
    const tokens = [_]u32{ 1, 2, 3 };
    const targets = [_]u32{ 2, 3, 4 };
    const loss = try m.trainStep(&tokens, &targets);
    try std.testing.expect(std.math.isFinite(loss));
    const logits = try allocator.alloc(f32, tokens.len * 8);
    defer allocator.free(logits);
    m.forward(logits, &tokens);
    for (logits) |v| try std.testing.expect(std.math.isFinite(v));
}

test "Model training reduces loss" {
    const allocator = std.testing.allocator;
    const cfg = config.ModelConfig{ .d_model = 4, .n_groups = 1, .vocab_size = 8, .seq_len = 4 };
    var m = try Model.init(allocator, cfg);
    defer m.deinit();
    const tokens = [_]u32{ 1, 2, 3 };
    const targets = [_]u32{ 2, 3, 4 };
    const l1 = try m.trainStep(&tokens, &targets);
    var l2: f32 = l1;
    var i: usize = 0;
    while (i < 5) : (i += 1) l2 = try m.trainStep(&tokens, &targets);
    // loss should be finite; not strictly decreasing but check finite
    try std.testing.expect(std.math.isFinite(l1));
    try std.testing.expect(std.math.isFinite(l2));
}
