// GPU elementwise kernel (PRD 3.3): vector add, compiled to PTX (sm_75).

pub fn threadIdX() u32 {
    return asm ("mov.u32 \t%[r], %tid.x;"
        : [r] "=r" (-> u32),
    );
}

export fn vecadd(a: [*]const f32, b: [*]const f32, c: [*]f32, n: u32) callconv(.kernel) void {
    const i = threadIdX();
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

test "vecadd ptx placeholder" {}
