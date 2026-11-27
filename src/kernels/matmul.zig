// GPU tiled matmul kernel (PRD 3.3): one thread per output element (reference), PTX sm_75.

pub fn threadIdX() u32 {
    return asm ("mov.u32 \t%[r], %tid.x;"
        : [r] "=r" (-> u32),
    );
}

export fn matmul(a: [*]const f32, b: [*]const f32, c: [*]f32, m: u32, n: u32, k: u32) callconv(.kernel) void {
    const tid = threadIdX();
    const total = m * n;
    if (tid >= total) return;
    const row = tid / n;
    const col = tid % n;
    var acc: f32 = 0;
    var t: u32 = 0;
    while (t < k) : (t += 1) {
        acc += a[row * k + t] * b[t * n + col];
    }
    c[row * n + col] = acc;
}

test "matmul ptx placeholder" {}
