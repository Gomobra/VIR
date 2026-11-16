//! Sisi host untuk kernel argmin terfusi (F7.1): bentuk peluncuran, pemeriksaan
//! kesiapan argumen, dan jalur cadangan matmul + argmax.
//!
//! Berkas ini sengaja tidak bergantung pada `burn-cubecl`/CUDA. Kernel CubeCL
//! tetap tinggal di [`crate::kernel`] di balik feature `fused`, tetapi bagian
//! yang bisa berjalan di CPU dipisah ke sini supaya:
//!
//! 1. `plan_launch` (chunk tuning) punya tes CPU — dulu tidak teruji karena
//!    seluruh modul kernel hanya dikompilasi dengan `--features fused`;
//! 2. `args_compatible` (jaga-jaga batas kernel: `dim`/`codes` nol, dtype,
//!    tensor bergradien) bisa diuji tanpa GPU;
//! 3. `fused_argmin_bias` memakai [`naive_argmin`] sebagai jalur cadangan, jadi
//!    jalur itulah yang dibandingkan dengan `argmin` naif pada tes.

use burn::prelude::*;
use burn::tensor::{DType, Shape};
use std::sync::OnceLock;

/// Baris `z` per cube. Cukup besar untuk mengamortisasi beban `e` antar-baris.
pub const ROWS_PER_CUBE: usize = 32;
/// Lebar warp; dipakai membagi thread menjadi (warp = baris, lane = kode).
pub const PLANE: usize = 32;
/// Thread per cube: satu warp per baris.
pub const THREADS_PER_CUBE: usize = ROWS_PER_CUBE * PLANE;
/// Kode `e` per blok. Satu beban shared dipakai semua baris pada cube.
pub const CODES_PER_TILE: usize = PLANE;
/// Elemen `d` per tile shared (D=384 -> 3 tile; shared ~16 KiB).
pub const D_TILE: usize = 128;
/// Leading dimension tile shared + padding, supaya beban tidak bentrok bank.
pub const E_TILE_STRIDE: usize = D_TILE + 1;
/// Skor "paling buruk" untuk lane tanpa kode valid; hanya perlu < skor nyata mana pun.
pub const LOWEST_SCORE: f32 = -1.0e30;
/// Batas baris per peluncuran: 2048 cube, di bawah batas grid backend mana pun.
pub const MAX_ROWS_PER_LAUNCH: usize = 1 << 16;
/// Variabel lingkungan untuk menimpa batas baris per peluncuran (hint autotune).
const ROWS_HINT_ENV: &str = "VIR_FUSED_ARGMAX_ROWS";

/// Rencana peluncuran (F7.1 chunk tuning) untuk `rows` baris.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchPlan {
    /// Baris per cube; tetap karena ukuran tile shared konstan.
    pub rows_per_cube: usize,
    /// Baris per peluncuran kernel; kelipatan `rows_per_cube`.
    pub rows_per_launch: usize,
    /// Banyak peluncuran; `ceil(rows / rows_per_launch)` (0 bila `rows == 0`).
    pub launches: usize,
}

/// Hint autotune: batas baris per peluncuran dari `VIR_FUSED_ARGMAX_ROWS`, di-cache
/// sepanjang proses supaya hasil profil dipakai ulang di panggilan berikutnya.
fn rows_hint() -> Option<usize> {
    static HINT: OnceLock<Option<usize>> = OnceLock::new();
    *HINT.get_or_init(|| {
        std::env::var(ROWS_HINT_ENV)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
    })
}

/// Pilih bentuk peluncuran dari jumlah baris.
pub fn plan_launch(rows: usize) -> LaunchPlan {
    let cap = rows_hint().unwrap_or(MAX_ROWS_PER_LAUNCH);
    let rows_per_launch = rows.min(cap).div_ceil(ROWS_PER_CUBE).max(1) * ROWS_PER_CUBE;
    LaunchPlan {
        rows_per_cube: ROWS_PER_CUBE,
        rows_per_launch,
        launches: rows.div_ceil(rows_per_launch),
    }
}

/// Bentuk, dtype, dan grad cocok untuk kernel? Device diperiksa terpisah.
///
/// `rows == 0` (batch kosong) tetap cocok: keluarannya `[0]`, tanpa kerja.
pub fn args_compatible(z: &Tensor<2>, e: &Tensor<2>, bias: &Tensor<1>) -> bool {
    let [_, dim] = z.dims();
    let [codes, e_dim] = e.dims();
    let [bias_len] = bias.dims();
    dim > 0
        && codes > 0
        && e_dim == dim
        && bias_len == codes
        && z.dtype() == DType::F32
        && e.dtype() == DType::F32
        && bias.dtype() == DType::F32
        && !z.is_require_grad()
        && !e.is_require_grad()
        && !bias.is_require_grad()
}

/// Bentuk keluaran: satu indeks per baris.
pub fn output_shape(z: &Shape) -> Shape {
    let rows = z.as_slice().first().copied().unwrap_or(0);
    Shape::new([rows])
}

/// dtype keluaran: `Tensor::<_, Int>` memakai `i32` (lihat `Tensor::from_ints`).
pub fn output_dtype() -> DType {
    DType::I32
}

/// Jalur cadangan non-CubeCL: `argmax_j (z · e_j + bias_j)` lewat matmul penuh.
///
/// Inilah pembanding benih (ground truth) kernel terfusi: kernel harus memberi
/// indeks yang sama dengan fungsi ini (dan dengan jarak Euclidean naif) pada
/// device apa pun.
pub fn naive_argmin(z: Tensor<2>, e: Tensor<2>, bias: Tensor<1>) -> Tensor<1, Int> {
    let rows = z.dims()[0];
    let codes = e.dims()[0];
    z.matmul(e.transpose())
        .add(bias.reshape([1, codes]))
        .argmax(1)
        .reshape([rows])
}

/// Device ini device CubeCL yang bisa menjalankan kernel?
pub fn is_cube_device(device: &Device) -> bool {
    #[cfg(feature = "cuda")]
    {
        matches!(device.as_dispatch(), burn::backend::DispatchDevice::Cube(_))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        false
    }
}
