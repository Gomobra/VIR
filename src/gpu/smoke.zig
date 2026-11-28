// GPU smoke test (PRD 3.3, M0): load embedded PTX and run vector add on device.

const std = @import("std");
const gpu = @import("gpu.zig");

const ptx = @embedFile("elementwise_ptx");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    if (!gpu.resultOk(gpu.cuInit(0))) return error.CuInitFailed;

    var count: c_int = 0;
    if (!gpu.resultOk(gpu.cuDeviceGetCount(&count))) return error.CuDeviceCountFailed;
    if (count <= 0) return error.NoCudaDevice;

    var dev: gpu.CUdevice = 0;
    if (!gpu.resultOk(gpu.cuDeviceGet(&dev, 0))) return error.CuDeviceGetFailed;

    var ctx: gpu.CUcontext = null;
    if (!gpu.resultOk(gpu.cuCtxCreate(&ctx, 0, dev))) return error.CuCtxCreateFailed;
    defer _ = gpu.cuCtxDestroy(ctx);

    const ptx_z = ptx ++ "\x00";
    var module: gpu.CUmodule = null;
    if (!gpu.resultOk(gpu.cuModuleLoadData(&module, @ptrCast(ptx_z.ptr)))) return error.CuModuleLoadFailed;
    defer _ = gpu.cuModuleUnload(module);

    var func: gpu.CUfunction = null;
    if (!gpu.resultOk(gpu.cuModuleGetFunction(&func, module, "vecadd"))) return error.CuFunctionNotFound;

    const n: u32 = 1024;
    const bytes: usize = @as(usize, n) * @sizeOf(f32);

    var host_a = try allocator.alloc(f32, n);
    defer allocator.free(host_a);
    var host_b = try allocator.alloc(f32, n);
    defer allocator.free(host_b);
    var host_c = try allocator.alloc(f32, n);
    defer allocator.free(host_c);

    for (host_a, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i));
    for (host_b, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) * 2.0;

    var d_a: u64 = 0;
    var d_b: u64 = 0;
    var d_c: u64 = 0;
    if (!gpu.resultOk(gpu.cuMemAlloc(&d_a, bytes))) return error.CuMemAllocFailed;
    defer _ = gpu.cuMemFree(d_a);
    if (!gpu.resultOk(gpu.cuMemAlloc(&d_b, bytes))) return error.CuMemAllocFailed;
    defer _ = gpu.cuMemFree(d_b);
    if (!gpu.resultOk(gpu.cuMemAlloc(&d_c, bytes))) return error.CuMemAllocFailed;
    defer _ = gpu.cuMemFree(d_c);

    if (!gpu.resultOk(gpu.cuMemcpyHtoD(d_a, @ptrCast(&host_a[0]), bytes))) return error.CuMemcpyHtoDFailed;
    if (!gpu.resultOk(gpu.cuMemcpyHtoD(d_b, @ptrCast(&host_b[0]), bytes))) return error.CuMemcpyHtoDFailed;

    var p_a = d_a;
    var p_b = d_b;
    var p_c = d_c;
    var p_n = n;
    var args = [_]*anyopaque{
        @ptrCast(&p_a),
        @ptrCast(&p_b),
        @ptrCast(&p_c),
        @ptrCast(&p_n),
    };

    const blocks = (n + 255) / 256;
    if (!gpu.resultOk(gpu.cuLaunchKernel(func, blocks, 1, 1, 256, 1, 1, 0, null, @ptrCast(&args), null))) return error.CuLaunchFailed;
    if (!gpu.resultOk(gpu.cuStreamSynchronize(null))) return error.CuStreamSyncFailed;
    if (!gpu.resultOk(gpu.cuMemcpyDtoH(@ptrCast(&host_c[0]), d_c, bytes))) return error.CuMemcpyDtoHFailed;

    for (host_c, 0..) |got, i| {
        const expected = @as(f32, @floatFromInt(i)) * 3.0;
        if (@abs(got - expected) > 1e-5) return error.GpuParityFailed;
    }

    std.debug.print("GPU smoke OK: {d} elements verified\n", .{n});
}
