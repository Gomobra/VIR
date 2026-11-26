// GPU backend (PRD 3.3): CUDA Driver API bindings (extern). Kernels in src/kernels.

const std = @import("std");

// Opaque handles
pub const CUdevice = i32;
pub const CUcontext = ?*anyopaque;
pub const CUmodule = ?*anyopaque;
pub const CUfunction = ?*anyopaque;
pub const CUstream = ?*anyopaque;
pub const CUresult = i32;

pub const CUDA_SUCCESS: CUresult = 0;

// Extern declarations (libcuda). Linking only required when called.
pub extern "c" fn cuInit(flags: u32) CUresult;
pub extern "c" fn cuDeviceGetCount(count: *c_int) CUresult;
pub extern "c" fn cuDeviceGet(dev: *CUdevice, ordinal: c_int) CUresult;
pub extern "c" fn cuCtxCreate(ctx: *CUcontext, flags: u32, dev: CUdevice) CUresult;
pub extern "c" fn cuCtxDestroy(ctx: CUcontext) CUresult;
pub extern "c" fn cuModuleLoadData(module: *CUmodule, image: *const anyopaque) CUresult;
pub extern "c" fn cuModuleUnload(module: CUmodule) CUresult;
pub extern "c" fn cuModuleGetFunction(func: *CUfunction, module: CUmodule, name: [*:0]const u8) CUresult;
pub extern "c" fn cuLaunchKernel(
    f: CUfunction,
    gridDimX: u32,
    gridDimY: u32,
    gridDimZ: u32,
    blockDimX: u32,
    blockDimY: u32,
    blockDimZ: u32,
    sharedMemBytes: u32,
    hStream: CUstream,
    kernelParams: ?*anyopaque,
    extra: ?*anyopaque,
) CUresult;

// Memory management
pub extern "c" fn cuMemAlloc(dptr: *u64, bytesize: usize) CUresult;
pub extern "c" fn cuMemFree(dptr: u64) CUresult;
pub extern "c" fn cuMemcpyHtoD(dstDevice: u64, srcHost: *const anyopaque, ByteCount: usize) CUresult;
pub extern "c" fn cuMemcpyDtoH(dstHost: *anyopaque, srcDevice: u64, ByteCount: usize) CUresult;
pub extern "c" fn cuStreamCreate(phStream: *CUstream, Flags: u32) CUresult;
pub extern "c" fn cuStreamDestroy(hStream: CUstream) CUresult;
pub extern "c" fn cuStreamSynchronize(hStream: CUstream) CUresult;

pub const DeviceBuffer = struct {
    ptr: u64,
    size: usize,
};

pub fn resultOk(r: CUresult) bool {
    return r == CUDA_SUCCESS;
}

test "resultOk" {
    try std.testing.expect(resultOk(0));
    try std.testing.expect(!resultOk(1));
}
