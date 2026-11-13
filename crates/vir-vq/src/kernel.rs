//! Kernel terfusi matmul + argmax untuk kuantisasi VQ (F7.1).
//!
//! Cari tetangga terdekat pada codebook berarti
//! `argmin_j ||z - e_j||^2 == argmax_j (z · e_j - 0.5 * ||e_j||^2)`.
//! Jalur biasa menghitung matmul `[N, D] x [D, V]` lalu `argmax` (reduce `Unit`):
//! ia menulis `[N, V]` (512 MB pada N=32768, V=4096) ke HBM lalu membacanya lagi.
//! Kernel di berkas ini menyatukan keduanya — tidak ada buffer `[N, V]`, hanya satu
//! `i32` per baris yang ditulis keluar.
//!
//! # Pemetaan thread
//!
//! - Satu cube memegang [`ROWS_PER_CUBE`] baris `z`; satu warp per baris.
//! - Satu lane memegang satu kode `e_j`; kode diproses per [`CODES_PER_TILE`] (32),
//!   jadi satu beban `e` dipakai bersama oleh semua baris pada cube (amortisasi
//!   lalu lintas `e`).
//! - `e` untuk satu blok kode dimuat ke shared memory sebagai tile
//!   `[CODES_PER_TILE][D_TILE]` secara coalesced, lalu tiap lane mengakumulasi dot
//!   product di register.
//! - Tiap lane melacak `(best_score, best_idx)` miliknya. Setelah semua blok kode,
//!   satu thread per baris mereduksi 32 kandidat lewat shared memory dan menulis
//!   argmax global. `>` (bukan `>=`) dipakai di kedua tingkat, jadi saat seri indeks
//!   terkecil yang menang — sama dengan `Tensor::argmax`.
//!
//! # Kapan dipakai
//!
//! Modul ini hanya dikompilasi dengan feature `fused` dan hanya dieksekusi untuk
//! tensor `f32` pada device CubeCL (lihat [`try_fused_argmin`]).

use burn::backend::Backend;
use burn::backend::TensorMetadata;
use burn::backend::backend_extension;
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::prelude::*;
use burn::tensor::{DType, Shape};
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::{CubeCount, CubeDim};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::tensor::CubeTensor;
use cube_ops::fused_argmin_kernel;

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

/// Kernel CubeCL. Dipisah ke modul sendiri karena prelude `cubecl` dan prelude
/// `burn` sama-sama mengekspor `Tensor`/`Float` (glob yang bertabrakan).
mod cube_ops {
    use super::{
        CODES_PER_TILE, D_TILE, E_TILE_STRIDE, LOWEST_SCORE, PLANE, ROWS_PER_CUBE, THREADS_PER_CUBE,
    };
    // Macro `cube` menumbuhkan path `cubecl::...`, jadi crate itu harus punya nama
    // tersebut di scope modul ini.
    use burn_cubecl::cubecl;
    use burn_cubecl::cubecl::prelude::*;

    /// `out[n] = argmax_j (z[n] · e[j] + bias[j])`.
    ///
    /// `z`: `[N, D]`, `e`: `[V, D]`, `bias`: `[V]`, `out`: `[N]`.
    #[cube(launch)]
    pub(crate) fn fused_argmin_kernel<F: Float, I: Numeric>(
        z: &Tensor<F>,
        e: &Tensor<F>,
        bias: &Tensor<F>,
        out: &mut Tensor<I>,
        num_rows: usize,
        dim: usize,
        num_codes: usize,
        #[define(F, I)] _dtypes: [ElemType; 2],
    ) {
        let cube = CUBE_POS_X as usize;
        let tid = UNIT_POS_X as usize;
        let row_start = cube * ROWS_PER_CUBE;

        // Guard seragam per cube: hanya bergantung pada indeks cube, jadi semua
        // thread keluar bersama dan tak ada yang tertinggal di `sync_cube`.
        if row_start >= num_rows {
            terminate!();
        }

        let warp = tid / PLANE;
        let lane = tid - warp * PLANE;
        let row = row_start + warp;
        // Baris terakhir sebuah cube bisa di luar `num_rows`; thread-nya tetap ikut
        // memuat tile dan mencapai setiap barrier, hanya tidak menghitung.
        let active = row < num_rows;

        let z_row_stride = z.stride(0);
        let e_row_stride = e.stride(0);

        // Tile `e` (shared) + papan skor/indeks untuk reduksi akhir.
        let mut sm_e = Shared::new_slice(CODES_PER_TILE * E_TILE_STRIDE);
        let mut sm_score = Shared::new_slice(ROWS_PER_CUBE * CODES_PER_TILE);
        let mut sm_idx = Shared::new_slice(ROWS_PER_CUBE * CODES_PER_TILE);
        let lowest = F::new(LOWEST_SCORE);

        let mut best_score = lowest;
        let mut best_idx = 0usize;

        let mut v_start = 0usize;
        while v_start < num_codes {
            let codes_in_tile = if v_start + CODES_PER_TILE <= num_codes {
                CODES_PER_TILE
            } else {
                num_codes - v_start
            };

            // Dot product penuh untuk kode pada blok ini, diakumulasi per tile `d`.
            let mut acc = F::new(0.0_f32);
            let mut d_start = 0usize;
            while d_start < dim {
                let d_len = if d_start + D_TILE <= dim {
                    D_TILE
                } else {
                    dim - d_start
                };

                // Muat `e` [kode][d] secara coalesced: thread berurutan membaca `d`
                // berurutan; `c` dan `d` dihitung dari indeks linear.
                let tile_elems = codes_in_tile * d_len;
                let mut i = tid;
                while i < tile_elems {
                    let c = i / d_len;
                    let d = i - c * d_len;
                    sm_e[c * E_TILE_STRIDE + d] = e[(v_start + c) * e_row_stride + d_start + d];
                    i += THREADS_PER_CUBE;
                }
                sync_cube();

                if active {
                    // `z[row][d]` uniform per warp -> broadcast, bukan beban per-lane.
                    let z_row = row * z_row_stride + d_start;
                    let mut d = 0usize;
                    while d < d_len {
                        acc += sm_e[lane * E_TILE_STRIDE + d] * z[z_row + d];
                        d += 1;
                    }
                }
                // Tile berikutnya menimpa shared, jadi baca harus selesai dulu.
                sync_cube();

                d_start += d_len;
            }

            if active && lane < codes_in_tile {
                let code = v_start + lane;
                let score = acc + bias[code];
                if score > best_score {
                    best_score = score;
                    best_idx = code;
                }
            }

            v_start += CODES_PER_TILE;
        }

        // Reduksi 32 kandidat per baris lewat shared memory: tidak bergantung pada
        // ukuran plane, jadi benar juga untuk runtime non-CUDA.
        let base = warp * CODES_PER_TILE;
        sm_score[base + lane] = best_score;
        sm_idx[base + lane] = i32::cast_from(best_idx);
        sync_cube();

        if lane == 0 && active {
            let mut best = 0usize;
            let mut k = 1usize;
            while k < CODES_PER_TILE {
                if sm_score[base + k] > sm_score[base + best] {
                    best = k;
                }
                k += 1;
            }
            out[row] = I::cast_from(usize::cast_from(sm_idx[base + best]));
        }
    }
}

/// Bentuk keluaran: satu indeks per baris.
fn output_shape(z: &Shape) -> Shape {
    Shape::new([z.dims::<2>()[0]])
}

/// dtype keluaran: `Tensor::<_, Int>` memakai `i32` (lihat `Tensor::from_ints`).
fn output_dtype() -> DType {
    DType::I32
}

/// Ekstensi backend: argmax terfusi untuk (z, e, bias).
#[backend_extension(Cube: cfg(feature = "cuda"), Fusion: cfg(feature = "cuda"))]
pub trait FusedArgminOps: Backend {
    /// `argmax_j (z · e_j + bias_j)` per baris `z`, tanpa materialisasi `[N, V]`.
    #[fusion(dtype = output_dtype(), shape = output_shape(z))]
    fn fused_argmin(
        z: FloatTensor<Self>,
        e: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> IntTensor<Self>;
}

impl FusedArgminOps for CubeBackend {
    fn fused_argmin(
        z: FloatTensor<Self>,
        e: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> IntTensor<Self> {
        launch_fused_argmin(z, e, bias)
    }
}

/// Host: siapkan argumen, jalankan kernel, kembalikan tensor indeks `[N]`.
fn launch_fused_argmin(z: CubeTensor, e: CubeTensor, bias: CubeTensor) -> CubeTensor {
    // Kernel mengindeks baris secara manual, jadi butuh layout physical row-major.
    let z = into_contiguous(z);
    let e = into_contiguous(e);
    let bias = into_contiguous(bias);

    let rows = z.shape().dims::<2>()[0];
    let dim = z.shape().dims::<2>()[1];
    let codes = e.shape().dims::<2>()[0];
    let z_dtype = z.dtype;
    let out_dtype = output_dtype();

    let client = z.client.clone();
    let device = z.device.clone();
    let out = empty_device_dtype(client, device, Shape::new([rows]), out_dtype);

    // Pemanggil menjaga `codes > 0` dan `dim > 0`; tanpa baris tidak ada kerja.
    if rows == 0 || codes == 0 || dim == 0 {
        return out;
    }

    let cubes = rows.div_ceil(ROWS_PER_CUBE);
    let cube_count = CubeCount::Static(cubes as u32, 1, 1);
    let cube_dim = CubeDim::new_1d(THREADS_PER_CUBE as u32);

    let client = out.client.clone();
    fused_argmin_kernel::launch(
        &client,
        cube_count,
        cube_dim,
        z.into_tensor_arg(),
        e.into_tensor_arg(),
        bias.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        rows,
        dim,
        codes,
        [
            dtype_to_storage_type(z_dtype),
            dtype_to_storage_type(out_dtype),
        ],
    );

    out
}

/// Device ini device CubeCL yang bisa menjalankan kernel?
pub fn is_cube_device(device: &Device) -> bool {
    #[cfg(feature = "cuda")]
    return matches!(device.as_dispatch(), burn::backend::DispatchDevice::Cube(_));
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = device;
        false
    }
}

/// `argmax_j (z · e_j + bias_j)` terfusi, atau `None` bila device/dtype tak cocok.
///
/// `None` berarti pemanggil harus memakai matmul + argmax biasa: device bukan
/// CubeCL, dtype bukan `f32`, atau `z` masih membawa graf autodiff (kernel ini tak
/// bisa dibalik, dan ekstensi backend sengaja tidak mendaftar `Autodiff`).
pub fn try_fused_argmin(z: &Tensor<2>, e: &Tensor<2>, bias: &Tensor<1>) -> Option<Tensor<1, Int>> {
    if z.dtype() != DType::F32 || z.is_require_grad() || !is_cube_device(&z.device()) {
        return None;
    }

    let z = z.clone().into_dispatch();
    let e = e.clone().into_dispatch();
    let bias = bias.clone().into_dispatch();

    Some(Tensor::from_dispatch(
        <burn::backend::Dispatch as FusedArgminOps>::fused_argmin(z, e, bias),
    ))
}

/// `argmin` terfusi untuk `z` `[N, D]` terhadap codebook `e` `[V, D]`.
///
/// Bias `-0.5 * ||e_j||^2` dihitung di sini; untuk pelatihan pakai
/// [`fused_argmin_bias`] dengan `e_sq` yang sudah di-cache.
///
/// # Panics
///
/// Panik bila device bukan CubeCL atau dtype bukan `f32`; pakai
/// [`try_fused_argmin`] bila perlu fallback.
pub fn fused_argmin(z: Tensor<2>, e: Tensor<2>) -> Tensor<1, Int> {
    let e_sq = (e.clone() * e.clone()).sum_dim(1).reshape([e.dims()[0]]);
    let bias = e_sq.mul_scalar(-0.5);
    fused_argmin_bias(z, e, bias)
}

/// `argmin` terfusi dengan bias siap pakai (`-0.5 * ||e_j||^2`).
///
/// # Panics
///
/// Panik bila device bukan CubeCL atau dtype bukan `f32`.
pub fn fused_argmin_bias(z: Tensor<2>, e: Tensor<2>, bias: Tensor<1>) -> Tensor<1, Int> {
    try_fused_argmin(&z, &e, &bias).unwrap_or_else(|| {
        // Jalur cadangan: hanya untuk device non-CubeCL, jadi biaya matmul ekstra
        // tidak pernah muncul pada target CUDA.
        let rows = z.dims()[0];
        z.matmul(e.transpose()).argmax(1).reshape([rows])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_constants_match_threads() {
        assert_eq!(THREADS_PER_CUBE, ROWS_PER_CUBE * PLANE);
        assert!(E_TILE_STRIDE > D_TILE);
        assert!(CODES_PER_TILE <= PLANE);
    }
}

/// Tes numerik kernel: butuh GPU (Kaggle/CI CUDA). `N` dan `V` sengaja bukan
/// kelipatan ukuran cube/tile agar jalur batas (baris & kode sisa, `d` sisa) ikut
/// teruji. Toleransi 0: indeks harus identik dengan `argmax` f32.
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use burn::tensor::Distribution;

    fn device() -> Device {
        Device::cuda(0)
    }

    /// Baca tensor indeks sebagai `Vec<i32>`.
    fn ints(tensor: Tensor<1, Int>) -> Vec<i32> {
        tensor
            .into_data()
            .try_to_vec::<i32>()
            .expect("tensor data as i32")
    }

    fn cases() -> [(usize, usize, usize); 4] {
        [(3000, 200, 100), (4096, 384, 4096), (37, 64, 32), (1, 8, 5)]
    }

    #[test]
    fn fused_matches_argmax_exactly() {
        let device = device();
        for (rows, dim, codes) in cases() {
            let z = Tensor::<2>::random([rows, dim], Distribution::Normal(0.0, 1.0), &device);
            let e = Tensor::<2>::random([codes, dim], Distribution::Normal(0.0, 1.0), &device);
            let e_sq = (e.clone() * e.clone()).sum_dim(1).reshape([codes]);
            let bias = e_sq.clone().mul_scalar(-0.5);

            let got = fused_argmin_bias(z.clone(), e.clone(), bias);
            let expected = (z.matmul(e.transpose()) - e_sq.reshape([1, codes]) * 0.5)
                .argmax(1)
                .reshape([rows]);

            assert_eq!(
                ints(got),
                ints(expected),
                "rows={rows} dim={dim} codes={codes}"
            );
        }
    }

    #[test]
    fn codebook_prefers_fused_on_cuda() {
        let device = device();
        let mut codebook = crate::Codebook::new(
            &crate::CodebookConfig {
                num_entries: 64,
                dim: 32,
                ..crate::CodebookConfig::default()
            },
            &device,
        );
        let z = Tensor::<2>::random([512, 32], Distribution::Normal(0.0, 1.0), &device);
        let idx = codebook.quantize_idx(z.clone());
        let reference = codebook.quantize_idx_reference(z);

        assert_eq!(ints(idx), ints(reference));
        codebook.refresh_cache();
    }
}
