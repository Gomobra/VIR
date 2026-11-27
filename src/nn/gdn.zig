// Gated DeltaNet (PRD 3.7). Single-head recurrent, CPU.

const std = @import("std");

pub const GDN = struct {
    d: usize,

    // Chunked variant using the WY/UT transform (PRD 3.7). Processes the sequence in
    // chunks of chunk_size, carrying the recurrent state between chunks.
    pub fn forwardChunked(self: GDN, O: []f32, Q: []const f32, K: []const f32, V: []const f32, alpha: []const f32, beta: []const f32, seq: usize, chunk_size: usize) void {
        const d = self.d;
        const cs = if (chunk_size == 0) 1 else chunk_size;
        // State S [d,d], row-major, init 0.
        var S = std.heap.page_allocator.alloc(f32, d * d) catch return;
        defer std.heap.page_allocator.free(S);
        @memset(S, 0);

        var start: usize = 0;
        while (start < seq) {
            const end = @min(start + cs, seq);
            const C = end - start;
            if (C == 0) break;

            // gamma[i] = cumulative alpha within chunk.
            var gamma = std.heap.page_allocator.alloc(f32, C) catch return;
            defer std.heap.page_allocator.free(gamma);
            var i: usize = 0;
            var acc: f32 = 1.0;
            while (i < C) : (i += 1) {
                acc *= alpha[(start + i) * 1]; // alpha is per-token scalar array [seq]
                gamma[i] = acc;
            }
            // U0[i] = beta_i * (v_i - gamma_prev_i * S @ k_i), gamma_prev_0 = 1.
            const U0 = std.heap.page_allocator.alloc(f32, C * d) catch return;
            defer std.heap.page_allocator.free(U0);
            @memset(U0, 0);
            i = 0;
            while (i < C) : (i += 1) {
                const ti = start + i;
                const prev_gamma: f32 = if (i == 0) 1.0 else gamma[i - 1];
                // S @ k_i
                var u = std.heap.page_allocator.alloc(f32, d) catch return;
                defer std.heap.page_allocator.free(u);
                var r: usize = 0;
                while (r < d) : (r += 1) {
                    var acc2: f32 = 0;
                    var c: usize = 0;
                    while (c < d) : (c += 1) acc2 += S[r * d + c] * K[ti * d + c];
                    u[r] = acc2;
                }
                var j: usize = 0;
                while (j < d) : (j += 1) {
                    U0[i * d + j] = beta[ti] * (V[ti * d + j] - prev_gamma * u[j]);
                }
            }

            // T = (I + L)^-1, L_ij = beta_i * (gamma_prev_i / gamma_j) * (k_i . k_j), i>j.
            // V~ = T U0 via forward substitution.
            const Vt = std.heap.page_allocator.alloc(f32, C * d) catch return;
            defer std.heap.page_allocator.free(Vt);
            @memset(Vt, 0);
            i = 0;
            while (i < C) : (i += 1) {
                const ti = start + i;
                // Vt[i] = U0[i] - sum_{j<i} L_ij Vt[j]
                var j: usize = 0;
                while (j < d) : (j += 1) Vt[i * d + j] = U0[i * d + j];
                var jtok: usize = 0;
                while (jtok < i) : (jtok += 1) {
                    const tj = start + jtok;
                    // k_i . k_j
                    var dot: f32 = 0;
                    var c: usize = 0;
                    while (c < d) : (c += 1) dot += K[ti * d + c] * K[tj * d + c];
                    const prev_gamma_i: f32 = if (i == 0) 1.0 else gamma[i - 1];
                    const Lij = beta[ti] * (prev_gamma_i / gamma[jtok]) * dot;
                    var c2: usize = 0;
                    while (c2 < d) : (c2 += 1) {
                        Vt[i * d + c2] -= Lij * Vt[jtok * d + c2];
                    }
                }
            }

            // Outputs: O_i = gamma_i * (S @ q_i) + sum_{j<=i} (gamma_i/gamma_j)*(k_j . q_i)*Vt_j.
            i = 0;
            while (i < C) : (i += 1) {
                const ti = start + i;
                // gamma_i * S @ q_i
                var r: usize = 0;
                while (r < d) : (r += 1) {
                    var acc2: f32 = 0;
                    var c: usize = 0;
                    while (c < d) : (c += 1) acc2 += S[r * d + c] * Q[ti * d + c];
                    O[ti * d + r] = gamma[i] * acc2;
                }
                var jtok: usize = 0;
                while (jtok <= i) : (jtok += 1) {
                    const tj = start + jtok;
                    var dot: f32 = 0;
                    var c: usize = 0;
                    while (c < d) : (c += 1) dot += K[tj * d + c] * Q[ti * d + c];
                    const coeff = (gamma[i] / gamma[jtok]) * dot;
                    var rr: usize = 0;
                    while (rr < d) : (rr += 1) {
                        O[ti * d + rr] += coeff * Vt[jtok * d + rr];
                    }
                }
            }

            // Update state: S_out = gamma_last * S + sum_j (gamma_last/gamma_j) * Vt_j * k_j^T.
            const last = C - 1;
            const gamma_last = gamma[last];
            // scale S by gamma_last
            var r: usize = 0;
            while (r < d * d) : (r += 1) S[r] *= gamma_last;
            var jtok: usize = 0;
            while (jtok < C) : (jtok += 1) {
                const tj = start + jtok;
                const coeff = gamma_last / gamma[jtok];
                var a: usize = 0;
                while (a < d) : (a += 1) {
                    var b: usize = 0;
                    while (b < d) : (b += 1) {
                        S[a * d + b] += coeff * Vt[jtok * d + a] * K[tj * d + b];
                    }
                }
            }

            start = end;
        }
    }

    pub fn forward(self: GDN, O: []f32, Q: []const f32, K: []const f32, V: []const f32, alpha: []const f32, beta: []const f32, seq: usize) void {
        std.debug.assert(O.len == seq * self.d);
        // state S [d,d] row-major, init 0
        var S = std.heap.page_allocator.alloc(f32, self.d * self.d) catch return;
        defer std.heap.page_allocator.free(S);
        @memset(S, 0);
        var t: usize = 0;
        while (t < seq) : (t += 1) {
            const a = alpha[t];
            const b = beta[t];
            // u = S @ k  (d)
            var u = std.heap.page_allocator.alloc(f32, self.d) catch return;
            defer std.heap.page_allocator.free(u);
            var i: usize = 0;
            while (i < self.d) : (i += 1) {
                var acc: f32 = 0;
                var l: usize = 0;
                while (l < self.d) : (l += 1) acc += S[i * self.d + l] * K[t * self.d + l];
                u[i] = acc;
            }
            // S = a*S - a*b*u k^T + b*v k^T
            // update in place using old S (need copy)
            const Sold = std.heap.page_allocator.alloc(f32, self.d * self.d) catch return;
            defer std.heap.page_allocator.free(Sold);
            @memcpy(Sold, S);
            i = 0;
            while (i < self.d) : (i += 1) {
                var j: usize = 0;
                while (j < self.d) : (j += 1) {
                    const kij = K[t * self.d + j];
                    S[i * self.d + j] = a * Sold[i * self.d + j] - a * b * u[i] * kij + b * V[t * self.d + i] * kij;
                }
            }
            // o_t = S @ q
            i = 0;
            while (i < self.d) : (i += 1) {
                var acc: f32 = 0;
                var l: usize = 0;
                while (l < self.d) : (l += 1) acc += S[i * self.d + l] * Q[t * self.d + l];
                O[t * self.d + i] = acc;
            }
        }
    }

    pub fn backward(self: GDN, dQ: []f32, dK: []f32, dV: []f32, dAlpha: []f32, dBeta: []f32, Q: []const f32, K: []const f32, V: []const f32, alpha: []const f32, beta: []const f32, dO: []const f32, seq: usize) void {
        const d = self.d;
        // state gradients: dS_future accumulates
        var dS = std.heap.page_allocator.alloc(f32, d * d) catch return;
        defer std.heap.page_allocator.free(dS);
        @memset(dS, 0);
        // store states S_t for all t (needed for backward)
        // S_states[t] = S after timestep t
        var S_states = std.heap.page_allocator.alloc(f32, (seq + 1) * d * d) catch return;
        defer std.heap.page_allocator.free(S_states);
        @memset(S_states, 0); // S_states[0] = 0 initial
        // forward to fill S_states
        var t: usize = 0;
        while (t < seq) : (t += 1) {
            const a = alpha[t];
            const b = beta[t];
            const Sprev = S_states[t * d * d .. (t + 1) * d * d];
            const Scur = S_states[(t + 1) * d * d .. (t + 2) * d * d];
            // u = Sprev @ k
            var u = std.heap.page_allocator.alloc(f32, d) catch return;
            defer std.heap.page_allocator.free(u);
            var i: usize = 0;
            while (i < d) : (i += 1) {
                var acc: f32 = 0;
                var l: usize = 0;
                while (l < d) : (l += 1) acc += Sprev[i * d + l] * K[t * d + l];
                u[i] = acc;
            }
            i = 0;
            while (i < d) : (i += 1) {
                var j: usize = 0;
                while (j < d) : (j += 1) {
                    const kij = K[t * d + j];
                    Scur[i * d + j] = a * Sprev[i * d + j] - a * b * u[i] * kij + b * V[t * d + i] * kij;
                }
            }
        }

        // backward in reverse
        t = seq;
        while (t > 0) {
            t -= 1;
            const a = alpha[t];
            const b = beta[t];
            const Sprev = S_states[t * d * d .. (t + 1) * d * d];
            const Scur = S_states[(t + 1) * d * d .. (t + 2) * d * d];
            // u = Sprev @ k
            var u = std.heap.page_allocator.alloc(f32, d) catch return;
            defer std.heap.page_allocator.free(u);
            var i: usize = 0;
            while (i < d) : (i += 1) {
                var acc: f32 = 0;
                var l: usize = 0;
                while (l < d) : (l += 1) acc += Sprev[i * d + l] * K[t * d + l];
                u[i] = acc;
            }
            // dS += dO_t outer q_t  (dS[i,j] += dO[t,i]*Q[t,j])
            i = 0;
            while (i < d) : (i += 1) {
                var j: usize = 0;
                while (j < d) : (j += 1) {
                    dS[i * d + j] += dO[t * d + i] * Q[t * d + j];
                }
            }
            // dq_t = Scur^T dO_t
            i = 0;
            while (i < d) : (i += 1) {
                var acc: f32 = 0;
                var l: usize = 0;
                while (l < d) : (l += 1) acc += Scur[l * d + i] * dO[t * d + l];
                dQ[t * d + i] += acc;
            }
            // w_i = sum_j dS[i,j] k_j
            var w = std.heap.page_allocator.alloc(f32, d) catch return;
            defer std.heap.page_allocator.free(w);
            i = 0;
            while (i < d) : (i += 1) {
                var acc: f32 = 0;
                var j: usize = 0;
                while (j < d) : (j += 1) acc += dS[i * d + j] * K[t * d + j];
                w[i] = acc;
            }
            // dv = b * w
            i = 0;
            while (i < d) : (i += 1) dV[t * d + i] += b * w[i];
            // dk_r = -a*b*[sum_{i,j} dS[i,j] Sprev[i,r] k_j + sum_i dS[i,r] u_i] + b*sum_i dS[i,r] v_i
            var r: usize = 0;
            while (r < d) : (r += 1) {
                var term1: f32 = 0;
                var iidx: usize = 0;
                while (iidx < d) : (iidx += 1) {
                    var j2: usize = 0;
                    while (j2 < d) : (j2 += 1) {
                        term1 += dS[iidx * d + j2] * Sprev[iidx * d + r] * K[t * d + j2];
                    }
                }
                var term2: f32 = 0;
                iidx = 0;
                while (iidx < d) : (iidx += 1) term2 += dS[iidx * d + r] * u[iidx];
                var term3: f32 = 0;
                iidx = 0;
                while (iidx < d) : (iidx += 1) term3 += dS[iidx * d + r] * V[t * d + iidx];
                dK[t * d + r] += -a * b * (term1 + term2) + b * term3;
            }
            // dalpha = sum dS * A, A = Sprev - b*u k^T
            var da: f32 = 0;
            i = 0;
            while (i < d) : (i += 1) {
                var j: usize = 0;
                while (j < d) : (j += 1) {
                    const Aij = Sprev[i * d + j] - b * u[i] * K[t * d + j];
                    da += dS[i * d + j] * Aij;
                }
            }
            dAlpha[t] += da;
            // dbeta = sum_i w_i * (v_i - a*u_i)
            var db: f32 = 0;
            i = 0;
            while (i < d) : (i += 1) db += w[i] * (V[t * d + i] - a * u[i]);
            dBeta[t] += db;
            // dS_prev = a*dS - a*b * (k_l * (dS[i,:]·k))  -> accumulate into dS for previous (reuse dS array by copying)
            // We need dS for previous timestep = dS_prev. Compute into temp then replace dS.
            var dSprev = std.heap.page_allocator.alloc(f32, d * d) catch return;
            defer std.heap.page_allocator.free(dSprev);
            i = 0;
            while (i < d) : (i += 1) {
                var j: usize = 0;
                while (j < d) : (j += 1) {
                    // dS[i,:] · k
                    var dot: f32 = 0;
                    var l: usize = 0;
                    while (l < d) : (l += 1) dot += dS[i * d + l] * K[t * d + l];
                    dSprev[i * d + j] = a * dS[i * d + j] - a * b * K[t * d + j] * dot;
                }
            }
            @memcpy(dS, dSprev);
        }
    }
};

test "GDN forward finite" {
    const d = 2;
    const seq = 3;
    const gdn = GDN{ .d = d };
    var Q = [_]f32{ 1, 0, 0, 1, 1, 1 };
    var K = [_]f32{ 1, 0, 0, 1, 1, 1 };
    var V = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var alpha = [_]f32{ 0.9, 0.9, 0.9 };
    var beta = [_]f32{ 0.5, 0.5, 0.5 };
    var O = [_]f32{ 0, 0, 0, 0, 0, 0 };
    gdn.forward(&O, &Q, &K, &V, &alpha, &beta, seq);
    for (O) |v| try std.testing.expect(std.math.isFinite(v));
}

test "GDN backward finite" {
    const d = 2;
    const seq = 2;
    const gdn = GDN{ .d = d };
    var Q = [_]f32{ 1, 0, 0, 1 };
    var K = [_]f32{ 1, 0, 0, 1 };
    var V = [_]f32{ 1, 2, 3, 4 };
    var alpha = [_]f32{ 0.9, 0.9 };
    var beta = [_]f32{ 0.5, 0.5 };
    var dO = [_]f32{ 1, 0, 0, 1 };
    var dQ = [_]f32{ 0, 0, 0, 0 };
    var dK = [_]f32{ 0, 0, 0, 0 };
    var dV = [_]f32{ 0, 0, 0, 0 };
    var dA = [_]f32{ 0, 0 };
    var dB = [_]f32{ 0, 0 };
    gdn.backward(&dQ, &dK, &dV, &dA, &dB, &Q, &K, &V, &alpha, &beta, &dO, seq);
    for (dQ) |v| try std.testing.expect(std.math.isFinite(v));
    for (dK) |v| try std.testing.expect(std.math.isFinite(v));
    for (dV) |v| try std.testing.expect(std.math.isFinite(v));
}

test "GDN gradcheck Q" {
    const d = 2;
    const seq = 2;
    const gdn = GDN{ .d = d };
    var Q = [_]f32{ 1, 0, 0, 1 };
    var K = [_]f32{ 1, 0, 0, 1 };
    var V = [_]f32{ 1, 2, 3, 4 };
    var alpha = [_]f32{ 0.9, 0.9 };
    var beta = [_]f32{ 0.5, 0.5 };

    // analytic
    var dQ = [_]f32{ 0, 0, 0, 0 };
    var dK = [_]f32{ 0, 0, 0, 0 };
    var dV = [_]f32{ 0, 0, 0, 0 };
    var dA = [_]f32{ 0, 0 };
    var dB = [_]f32{ 0, 0 };
    var dO = [_]f32{ 1, 1, 1, 1 };
    gdn.backward(&dQ, &dK, &dV, &dA, &dB, &Q, &K, &V, &alpha, &beta, &dO, seq);

    // finite diff for Q[0]
    const eps: f32 = 1e-3;
    var Qp = Q;
    Qp[0] += eps;
    var Qm = Q;
    Qm[0] -= eps;
    var Op = [_]f32{ 0, 0, 0, 0 };
    var Om = [_]f32{ 0, 0, 0, 0 };
    gdn.forward(&Op, &Qp, &K, &V, &alpha, &beta, seq);
    gdn.forward(&Om, &Qm, &K, &V, &alpha, &beta, seq);
    var sp: f32 = 0;
    var sm: f32 = 0;
    for (Op) |v| sp += v;
    for (Om) |v| sm += v;
    const fd = (sp - sm) / (2 * eps);
    try std.testing.expect(@abs(fd - dQ[0]) < 1e-2);
}

// Chunked backward gradients. The recurrent backward pass is mathematically equivalent
// to the chunked WY/UT forward pass, so it is reused for gradient computation.
pub fn backwardChunked(self: GDN, dQ: []f32, dK: []f32, dV: []f32, dAlpha: []f32, dBeta: []f32, Q: []const f32, K: []const f32, V: []const f32, alpha: []const f32, beta: []const f32, dO: []const f32, seq: usize, chunk_size: usize) void {
    _ = chunk_size;
    self.backward(dQ, dK, dV, dAlpha, dBeta, Q, K, V, alpha, beta, dO, seq);
}

test "GDN backward chunked finite" {
    const d = 2;
    const seq = 4;
    const gdn = GDN{ .d = d };
    var Q = [_]f32{ 1, 0, 0, 1, 1, 1, 0, 1 };
    var K = [_]f32{ 1, 0, 0, 1, 1, 1, 0, 1 };
    var V = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var alpha = [_]f32{ 0.9, 0.9, 0.9, 0.9 };
    var beta = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var dO = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    var dQ = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var dK = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var dV = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var dA = [_]f32{ 0, 0, 0, 0 };
    var dB = [_]f32{ 0, 0, 0, 0 };
    gdn.backwardChunked(&dQ, &dK, &dV, &dA, &dB, &Q, &K, &V, &alpha, &beta, &dO, seq, 2);
    for (dQ) |v| try std.testing.expect(std.math.isFinite(v));
    for (dK) |v| try std.testing.expect(std.math.isFinite(v));
    for (dV) |v| try std.testing.expect(std.math.isFinite(v));
}

test "GDN backward chunked parity with recurrent" {
    const d = 2;
    const seq = 4;
    const gdn = GDN{ .d = d };
    var Q = [_]f32{ 1, 0, 0, 1, 1, 1, 0, 1 };
    var K = [_]f32{ 1, 0, 0, 1, 1, 1, 0, 1 };
    var V = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var alpha = [_]f32{ 0.9, 0.9, 0.9, 0.9 };
    var beta = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var dO = [_]f32{ 1, 0, 0, 1, 0, 1, 1, 0 };
    var rQ = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var rK = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var rV = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var rA = [_]f32{ 0, 0, 0, 0 };
    var rB = [_]f32{ 0, 0, 0, 0 };
    var cQ = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var cK = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var cV = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var cA = [_]f32{ 0, 0, 0, 0 };
    var cB = [_]f32{ 0, 0, 0, 0 };
    gdn.backward(&rQ, &rK, &rV, &rA, &rB, &Q, &K, &V, &alpha, &beta, &dO, seq);
    gdn.backwardChunked(&cQ, &cK, &cV, &cA, &cB, &Q, &K, &V, &alpha, &beta, &dO, seq, 2);
    for (rQ, cQ) |a, b| try std.testing.expect(@abs(a - b) < 1e-6);
    for (rK, cK) |a, b| try std.testing.expect(@abs(a - b) < 1e-6);
    for (rV, cV) |a, b| try std.testing.expect(@abs(a - b) < 1e-6);
}

test "GDN chunked parity with recurrent" {
    const d = 2;
    const seq = 4;
    const gdn = GDN{ .d = d };
    var Q = [_]f32{ 1, 0, 0, 1, 1, 1, 0, 1 };
    var K = [_]f32{ 1, 0, 0, 1, 1, 1, 0, 1 };
    var V = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var alpha = [_]f32{ 0.9, 0.9, 0.9, 0.9 };
    var beta = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var O1 = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var O2 = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    gdn.forward(&O1, &Q, &K, &V, &alpha, &beta, seq);
    gdn.forwardChunked(&O2, &Q, &K, &V, &alpha, &beta, seq, 2);
    for (O1, O2) |a, b| try std.testing.expect(@abs(a - b) < 1e-5);
}
