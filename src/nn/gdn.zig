// Gated DeltaNet (PRD 3.7). Single-head recurrent, CPU.

const std = @import("std");

pub const GDN = struct {
    d: usize,

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
            var Sold = std.heap.page_allocator.alloc(f32, self.d * self.d) catch return;
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
                var i2: usize = 0;
                while (i2 < d) : (i2 += 1) {
                    var j2: usize = 0;
                    while (j2 < d) : (j2 += 1) {
                        term1 += dS[i2 * d + j2] * Sprev[i2 * d + r] * K[t * d + j2];
                    }
                }
                var term2: f32 = 0;
                i2 = 0;
                while (i2 < d) : (i2 += 1) term2 += dS[i2 * d + r] * u[i2];
                var term3: f32 = 0;
                i2 = 0;
                while (i2 < d) : (i2 += 1) term3 += dS[i2 * d + r] * V[t * d + i2];
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
