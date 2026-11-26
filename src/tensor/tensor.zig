// Tensor core module (PRD 3.1). Implementasi bertahap.

pub const DType = enum { f32, f16, i8, u8, i32, u16, u32, ternary2 };
pub const Device = enum { cpu, gpu };

pub const Tensor = struct {
    dtype: DType,
    device: Device,
    ndim: u8,
    shape: [5]usize,
    strides: [5]usize,
    offset: usize,
    owns: bool,
};

pub fn dtypeSize(dt: DType) usize {
    return switch (dt) {
        .f32 => 4,
        .f16 => 2,
        .i8, .u8, .ternary2 => 1,
        .i32, .u32 => 4,
        .u16 => 2,
    };
}
