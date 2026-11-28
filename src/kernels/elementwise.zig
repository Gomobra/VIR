// GPU elementwise kernel (PRD 3.3): vector add, compiled to PTX (sm_75).

pub fn threadIdX() u32 {
    return asm ("mov.u32 \t%[r], %tid.x;"
        : [r] "=r" (-> u32),
    );
}

pub fn blockDimX() u32 {
    return asm ("mov.u32 \t%[r], %blockDim.x;"
        : [r] "=r" (-> u32),
    );
}

pub fn gridDimX() u32 {
    return asm ("mov.u32 \t%[r], %gridDim.x;"
        : [r] "=r" (-> u32),
    );
}

export fn vecadd(a: [*]const f32, b: [*]const f32, c: [*]f32, n: u32) callconv(.kernel) void {
    var idx = threadIdX();
    const stride = blockDimX() * gridDimX();
    while (idx < n) : (idx += stride) {
        c[idx] = a[idx] + b[idx];
    }
}

test "vecadd ptx placeholder" {}
