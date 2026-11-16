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
//! - Satu cube memegang [`ROWS_PER_CUBE`] baris `z`; satu warp per baris, satu
//!   lane memegang satu kode `e_j` (kode diproses per [`CODES_PER_TILE`] = 32,
//!   jadi satu beban `e` dipakai bersama semua baris pada cube).
//! - `e` satu blok kode dimuat ke shared memory sebagai tile
//!   `[CODES_PER_TILE][D_TILE]` secara coalesced; tiap lane mengakumulasi dot
//!   product di register.
//! - Tiap lane melacak `(best_score, best_idx)`; setelah semua blok kode, satu
//!   thread per baris mereduksi 32 kandidat lewat shared memory. `>` (bukan
//!   `>=`) di kedua tingkat, jadi saat seri indeks terkecil menang — sama dengan
//!   `Tensor::argmax`.
//!
//! # Kapan dipakai
//!
//! Hanya dikompilasi dengan feature `fused`, hanya jalan untuk `f32` pada device
//! CubeCL (lihat [`try_fused_argmin`]).
//!
//! # Tuning (F7.1)
//!
//! Bentuk peluncuran dipilih dari jumlah baris oleh
//! [`plan_launch`](crate::fused_plan::plan_launch) di [`crate::fused_plan`]
//! (murni sisi-host, bisa diuji di CPU). Profil dapat menimpa batas baris lewat
//! `VIR_FUSED_ARGMAX_ROWS`; kernel hanya menulis `[N]` `i32`, jadi chunking
//! tidak menambah memori puncak.

use burn::backend::Backend;
use burn::backend::TensorMetadata;
use burn::backend::backend_extension;
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::prelude::*;
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::{CubeCount, CubeDim};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::{empty_device_dtype, zeros_client};
use burn_cubecl::tensor::CubeTensor;
use cube_ops::fused_argmin_kernel;

pub use crate::fused_plan::{
    CODES_PER_TILE, D_TILE, E_TILE_STRIDE, LOWEST_SCORE, LaunchPlan, MAX_ROWS_PER_LAUNCH, PLANE,
    ROWS_PER_CUBE, THREADS_PER_CUBE, args_compatible, is_cube_device, naive_argmin, output_dtype,
    output_shape, plan_launch,
};

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

    /// `out[n] = argmax_j (z[n] · e[j] + bias[j])` untuk baris
    /// `row_offset <= n < row_offset + num_rows`.
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
        row_offset: usize,
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
                    let z_row = (row_offset + row) * z_row_stride + d_start;
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
            out[row_offset + row] = I::cast_from(usize::cast_from(sm_idx[base + best]));
        }
    }
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

/// Host: siapkan argumen, jalankan kernel (per chunk baris), kembalikan `[N]`.
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
    let out = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([rows]),
        out_dtype,
    );

    // Pemanggil menjaga `codes > 0` dan `dim > 0`; tanpa baris tidak ada kerja.
    if rows == 0 {
        return out;
    }
    if codes == 0 || dim == 0 {
        // Jaga-jaga: kernel akan membaca `e`/`z` di luar batas bila salah satunya nol.
        return zeros_client(client, device, Shape::new([rows]), out_dtype);
    }

    let plan = plan_launch(rows);
    let mut start = 0usize;
    while start < rows {
        let end = (start + plan.rows_per_launch).min(rows);
        let chunk_rows = end - start;
        let cubes = chunk_rows.div_ceil(ROWS_PER_CUBE);
        fused_argmin_kernel::launch(
            &client,
            CubeCount::Static(cubes as u32, 1, 1),
            CubeDim::new_1d(THREADS_PER_CUBE as u32),
            z.clone().into_tensor_arg(),
            e.clone().into_tensor_arg(),
            bias.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            chunk_rows,
            dim,
            codes,
            start,
            [
                dtype_to_storage_type(z_dtype),
                dtype_to_storage_type(out_dtype),
            ],
        );
        start = end;
    }

    out
}

/// `argmax_j (z · e_j + bias_j)` terfusi, atau `None` bila tak dapat dijalankan.
///
/// `None` berarti pemanggil harus memakai matmul + argmax biasa. Dua alasan:
///
/// 1. Device bukan CubeCL, atau salah satu tensor bukan `f32`, atau `z` masih
///    membawa graf autodiff (kernel ini tak bisa dibalik, dan ekstensi backend
///    sengaja tidak mendaftar `Autodiff`).
/// 2. Bentuk tak cocok (`dim` `z` ≠ `dim` `e`, `len(bias)` ≠ `len(e)`, atau `dim`/
///    `codes` nol) atau ketiga tensor berada di device berbeda. Ini jaga-jaga agar
///    kernel tak pernah membaca memori di luar batas.
pub fn try_fused_argmin(z: &Tensor<2>, e: &Tensor<2>, bias: &Tensor<1>) -> Option<Tensor<1, Int>> {
    let same_device = z.device() == e.device() && z.device() == bias.device();
    if !same_device || !args_compatible(z, e, bias) || !is_cube_device(&z.device()) {
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
/// [`fused_argmin_bias`] dengan `e_sq` yang sudah di-cache. Jatuh ke matmul +
/// argmax biasa bila device bukan CubeCL atau dtype bukan `f32`.
pub fn fused_argmin(z: Tensor<2>, e: Tensor<2>) -> Tensor<1, Int> {
    let e_sq = (e.clone() * e.clone()).sum_dim(1).reshape([e.dims()[0]]);
    let bias = e_sq.mul_scalar(-0.5);
    fused_argmin_bias(z, e, bias)
}

/// `argmin` terfusi dengan bias siap pakai (`-0.5 * ||e_j||^2`).
///
/// Bila [`try_fused_argmin`] menolak (device bukan CubeCL, dtype bukan `f32`,
/// bentuk tak cocok), jatuh ke matmul + `argmax` biasa — hasilnya identik.
pub fn fused_argmin_bias(z: Tensor<2>, e: Tensor<2>, bias: Tensor<1>) -> Tensor<1, Int> {
    try_fused_argmin(&z, &e, &bias).unwrap_or_else(|| {
        // Jalur cadangan (non-CubeCL): biaya matmul ekstra tidak pernah muncul
        // pada target CUDA. `bias` wajib ikut, kalau tidak hasilnya bukan
        // `argmax_j (z·e_j + bias_j)`. Lihat `fused_plan::naive_argmin`.
        naive_argmin(z, e, bias)
    })
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

    /// Chunk tuning (`rows > MAX_ROWS_PER_LAUNCH`) harus identik dengan `argmax`:
    /// menguji `row_offset` dan batas antar-peluncuran.
    #[test]
    fn fused_chunked_rows_match_argmax() {
        let device = device();
        let rows = MAX_ROWS_PER_LAUNCH + 1000;
        let (dim, codes) = (48, 96);
        let z = Tensor::<2>::random([rows, dim], Distribution::Normal(0.0, 1.0), &device);
        let e = Tensor::<2>::random([codes, dim], Distribution::Normal(0.0, 1.0), &device);
        let e_sq = (e.clone() * e.clone()).sum_dim(1).reshape([codes]);
        let bias = e_sq.clone().mul_scalar(-0.5);

        let got = fused_argmin_bias(z.clone(), e.clone(), bias);
        let expected = (z.matmul(e.transpose()) - e_sq.reshape([1, codes]) * 0.5)
            .argmax(1)
            .reshape([rows]);

        assert_eq!(ints(got), ints(expected));
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
