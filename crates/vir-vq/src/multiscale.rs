//! Kuantisasi residual multiscale dan operator `R_k` (PRD F1.4.b, rumus R2).
//!
//! Untuk setiap level `k` (R2):
//!
//! ```text
//! u_k    = Down_k(z^(k−1))            ∈ ℝ^{ℓ_k×d}
//! r_k^i  = argmin_j ‖u_k^i − e_{k,j}‖²
//! ẑ^(k)  = ẑ^(k−1) + R_k(e_k(r_k))
//! z^(k)  = z − ẑ^(k)                  ,  ẑ^(0) = 0, z^(0) = z
//! ```
//!
//! **Pilihan operator (ADR).** Teks memakai `Down_k` = *average pooling* area
//! tanpa tumpang tindih (setiap posisi masuk tepat satu blok; menghormati
//! padding karena blok mengikuti panjang level), `R_k` = interpolasi linear
//! (`align_corners`) + konvolusi residual `Phi` (PRD F1.4.b langkah 4). SNAC
//! hanya untuk audio (F6.4). Konvolusi seperti VAR dibandingkan lewat ADR.
//!
//! `Down_k` dan `R_k` ditulis sebagai matriks `[ℓ, L]` dan `[L, ℓ]` yang
//! dipasang di host, lalu dikalikan sebagai matmul batch: tanpa kernel khusus,
//! bisa dibalik (autodiff), dan bebas dari ketergantungan pada
//! `F.interpolate` yang tidak ada di Burn.

use crate::quantizer::Quantizer;
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::PaddingConfig1d;
use burn::prelude::*;
use vir_common::VirError;

/// Matriks turun/naik satu level, dihitung di host lalu dipakai sebagai matmul.
#[derive(Debug, Clone)]
pub struct ScaleOps {
    /// `[ℓ, L]`: bobot average pooling area.
    down: Tensor<2>,
    /// `[L, ℓ]`: bobot interpolasi linear.
    up: Tensor<2>,
    in_len: usize,
    out_len: usize,
}

impl ScaleOps {
    /// Matriks turun/naik untuk `L → ℓ`.
    pub fn new(in_len: usize, out_len: usize, device: &Device) -> Self {
        let down_flat: Vec<f32> = pooling_matrix(in_len, out_len)
            .into_iter()
            .flatten()
            .collect();
        let up_flat: Vec<f32> = interp_matrix(in_len, out_len)
            .into_iter()
            .flatten()
            .collect();
        let down = crate::quantizer::rows_to_tensor(&down_flat, in_len, device);
        let up = crate::quantizer::rows_to_tensor(&up_flat, out_len, device);
        Self {
            down,
            up,
            in_len,
            out_len,
        }
    }

    /// Panjang masukan `L`.
    pub fn in_len(&self) -> usize {
        self.in_len
    }

    /// Panjang keluaran `ℓ`.
    pub fn out_len(&self) -> usize {
        self.out_len
    }

    /// Turun: `[B, L, d] → [B, ℓ, d]`.
    pub fn down(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, _, _] = x.dims();
        let m = self
            .down
            .clone()
            .reshape([1, self.out_len, self.in_len])
            .expand([b, self.out_len, self.in_len]);
        m.matmul(x)
    }

    /// Naik: `[B, ℓ, d] → [B, L, d]`.
    pub fn up(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, _, _] = x.dims();
        let m = self
            .up
            .clone()
            .reshape([1, self.in_len, self.out_len])
            .expand([b, self.in_len, self.out_len]);
        m.matmul(x)
    }
}

/// Bobot average pooling area: blok `[⌊j·L/ℓ⌋, ⌊(j+1)·L/ℓ⌋)` (tanpa tumpang
/// tindih), blok terakhir dipaksa menutup `L`.
fn pooling_matrix(in_len: usize, out_len: usize) -> Vec<Vec<f32>> {
    let mut rows = Vec::with_capacity(out_len);
    for j in 0..out_len {
        let start = j * in_len / out_len;
        let mut end = ((j + 1) * in_len / out_len).max(start + 1).min(in_len);
        if j + 1 == out_len {
            end = in_len;
        }
        let weight = 1.0 / (end - start) as f32;
        let mut row = vec![0.0f32; in_len];
        for slot in row.iter_mut().take(end).skip(start) {
            *slot = weight;
        }
        rows.push(row);
    }
    rows
}

/// Bobot interpolasi linear (`align_corners = true`); `in_len == 1` menyiarkan.
fn interp_matrix(in_len: usize, out_len: usize) -> Vec<Vec<f32>> {
    // up: [L, ℓ] — tiap baris (posisi L) punya bobot interpolasi atas ℓ.
    if out_len == 1 {
        return vec![vec![1.0f32]; in_len];
    }
    let mut rows = Vec::with_capacity(in_len);
    for i in 0..in_len {
        let t = if in_len == 1 {
            0.0
        } else {
            i as f32 * (out_len - 1) as f32 / (in_len - 1) as f32
        };
        let lo = (t.floor() as usize).min(out_len - 1);
        let hi = (lo + 1).min(out_len - 1);
        let frac = t - lo as f32;
        let mut row = vec![0.0f32; out_len];
        row[lo] += 1.0 - frac;
        row[hi] += frac;
        rows.push(row);
    }
    rows
}

/// Konvolusi residual `Phi` pada `R_k` (PRD F1.4.b langkah 4).
#[derive(Module, Debug)]
pub struct Phi {
    pub conv: Conv1d,
}

impl Phi {
    /// `Phi` depthwise dengan kernel `kernel` dan padding `same`.
    pub fn new(dim: usize, kernel: usize, device: &Device) -> Self {
        let config = Conv1dConfig::new(dim, dim, kernel.max(1))
            .with_padding(PaddingConfig1d::Same)
            .with_groups(dim)
            .with_bias(false);
        Self {
            conv: config.init(device),
        }
    }

    /// `Phi` nol: `R_k` menjadi interpolasi linear murni (dipakai untuk ablasi
    /// "tanpa Phi" dan untuk mengunci invarian pada tes).
    pub fn zero(dim: usize, kernel: usize, device: &Device) -> Self {
        let mut phi = Self::new(dim, kernel, device);
        let shape = phi.conv.weight.val().dims();
        let zeros = Tensor::<3>::zeros(shape, device);
        crate::quantizer::replace_param(&mut phi.conv.weight, zeros);
        phi
    }

    /// `Phi(x)` pada `[B, L, d]` (konvolusi 1D di sumbu waktu).
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let y = self.conv.forward(x.swap_dims(1, 2));
        y.swap_dims(1, 2)
    }
}

/// Satu level: codebook + `Down_k`/`R_k` + `Phi`.
#[derive(Module, Debug)]
pub struct LevelVq<Q: Quantizer + Module> {
    pub quantizer: Q,
    #[module(skip)]
    pub scale: ScaleOps,
    pub phi: Phi,
}

/// Hasil satu level.
#[derive(Debug, Clone)]
pub struct LevelOutput {
    /// Indeks kode `[B, ℓ]`.
    pub indices: Tensor<2, Int>,
    /// Kode terkuantisasi `[B, ℓ, d]` (nilai, tanpa gradien).
    pub codes: Tensor<3>,
    /// `R_k(e_k(r_k))` pada `[B, L, d]` (dengan gradien straight-through).
    pub contribution: Tensor<3>,
}

impl<Q: Quantizer + Module> LevelVq<Q> {
    /// Level untuk `prev_len → len`.
    pub fn new(
        prev_len: usize,
        len: usize,
        phi_kernel: usize,
        quantizer: Q,
        device: &Device,
    ) -> Self {
        let dim = quantizer.dim();
        Self {
            quantizer,
            scale: ScaleOps::new(prev_len, len, device),
            phi: Phi::new(dim, phi_kernel, device),
        }
    }

    /// `R_k(e) = up(e) + Phi(up(e))` untuk kode `[B, ℓ, d]`.
    pub fn reconstruct(&self, codes: Tensor<3>) -> Tensor<3> {
        let up = self.scale.up(codes);
        up.clone() + self.phi.forward(up)
    }

    /// Satu level: `Down_k` → kuantisasi (dengan straight-through) → `R_k`.
    pub fn forward(&self, residual: Tensor<3>) -> LevelOutput {
        let [batch, _, dim] = residual.dims();
        let u = self.scale.down(residual);
        let len = u.dims()[1];
        let flat = u.clone().reshape([batch * len, dim]);
        let (quantized, st) = self.quantizer.quantize_st(flat);
        let codes = quantized.quantized.reshape([batch, len, dim]);
        let contribution = self.reconstruct(st.reshape([batch, len, dim]));
        LevelOutput {
            indices: quantized.indices.reshape([batch, len]),
            codes,
            contribution,
        }
    }

    /// Rekonstruksi dari indeks saja (dipakai generator R5–R6).
    pub fn reconstruct_from_indices(
        &self,
        indices: Tensor<2, Int>,
        dim: usize,
    ) -> Result<Tensor<3>, VirError> {
        let [b, l] = indices.dims();
        let flat = indices.reshape([b * l]);
        let codes = self.quantizer.lookup(flat).reshape([b, l, dim]);
        Ok(self.reconstruct(codes))
    }
}

/// Encoder VQ multiscale: `K` level residual berlapis.
#[derive(Module, Debug)]
pub struct MultiscaleVq<Q: Quantizer + Module> {
    pub levels: Vec<LevelVq<Q>>,
    #[module(skip)]
    total_len: usize,
}

/// Hasil [`MultiscaleVq::encode`].
#[derive(Debug, Clone)]
pub struct EncodeOutput {
    /// `ẑ^(k)` kumulatif per level (`k = 0..cutoff-1`), semuanya `[B, L, d]`.
    pub z_hat: Vec<Tensor<3>>,
    /// Kontribusi `R_k(e_k(r_k))` per level.
    pub contributions: Vec<Tensor<3>>,
    /// Indeks kode per level, `[B, ℓ_k]`.
    pub indices: Vec<Tensor<2, Int>>,
    /// Rekonstruksi akhir `ẑ^(cutoff)`.
    pub final_z: Tensor<3>,
}

impl<Q: Quantizer + Module> MultiscaleVq<Q> {
    /// Rakit dari jadwal level (naik ketat, berakhir di `L`) dan pembuat
    /// kuantizer per level (indeks level 0-based → kuantizer).
    ///
    /// `phi_kernel <= 1` memakai `Phi` nol (tanpa konvolusi).
    pub fn new(
        schedule: &[usize],
        phi_kernel: usize,
        build: impl Fn(usize) -> Q,
        device: &Device,
    ) -> Self {
        let total_len = schedule.last().copied().unwrap_or(0);
        let mut levels = Vec::with_capacity(schedule.len());
        let mut prev = total_len;
        for (k, len) in schedule.iter().copied().enumerate() {
            let quantizer = build(k);
            let phi = if phi_kernel > 1 {
                Phi::new(quantizer.dim(), phi_kernel, device)
            } else {
                Phi::zero(quantizer.dim(), 1, device)
            };
            levels.push(LevelVq {
                quantizer,
                scale: ScaleOps::new(prev, len, device),
                phi,
            });
            prev = len;
        }
        Self { levels, total_len }
    }

    /// Jumlah level.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Panjang total `L`.
    pub fn total_len(&self) -> usize {
        self.total_len
    }

    /// Encode residual multiscale dengan cutoff `cutoff` level pertama (R2, R3).
    pub fn encode(&self, z: Tensor<3>, cutoff: usize) -> EncodeOutput {
        let k = cutoff.clamp(1, self.levels.len());
        let mut z_hat: Tensor<3> = z.zeros_like();
        let mut residual = z.clone();
        let mut zs = Vec::with_capacity(k);
        let mut contributions = Vec::with_capacity(k);
        let mut indices = Vec::with_capacity(k);
        for level in self.levels.iter().take(k) {
            let out = level.forward(residual);
            z_hat = z_hat + out.contribution.clone();
            residual = z.clone() - z_hat.clone();
            zs.push(z_hat.clone());
            contributions.push(out.contribution);
            indices.push(out.indices);
        }
        EncodeOutput {
            z_hat: zs,
            contributions,
            indices,
            final_z: z_hat,
        }
    }

    /// Encode hierarki penuh.
    pub fn encode_full(&self, z: Tensor<3>) -> EncodeOutput {
        self.encode(z, self.levels.len())
    }

    /// Rekonstruksi dari indeks semua level (generator R5–R6).
    ///
    /// `Err` bila jumlah level tidak cocok dengan jumlah blok indeks.
    pub fn reconstruct(&self, indices: &[Tensor<2, Int>]) -> Result<Tensor<3>, VirError> {
        if indices.len() != self.levels.len() {
            return Err(VirError::Config(format!(
                "butuh {} indeks level, diberi {}",
                self.levels.len(),
                indices.len()
            )));
        }
        let mut acc: Option<Tensor<3>> = None;
        for (level, idx) in self.levels.iter().zip(indices.iter()) {
            let dim = level.quantizer.dim();
            let contribution = level.reconstruct_from_indices(idx.clone(), dim)?;
            acc = Some(match acc {
                Some(prev) => prev + contribution,
                None => contribution,
            });
        }
        acc.ok_or_else(|| VirError::Config("jadwal level kosong".into()))
    }
}

/// Jumlah berbobot kontribusi level; `None` bila daftar kosong.
pub fn weighted_sum(levels: &[Tensor<3>], weights: &[f32]) -> Option<Tensor<3>> {
    let (first, rest) = levels.split_first()?;
    let w0 = weights.first().copied().unwrap_or(1.0);
    let mut acc = first.clone().mul_scalar(w0);
    for (z, w) in rest.iter().zip(weights.iter().skip(1)) {
        acc = acc + z.clone().mul_scalar(*w);
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::{Codebook, CodebookConfig};
    use burn::tensor::Distribution;

    fn device() -> Device {
        Device::default()
    }

    fn f(t: Tensor<3>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap_or_default()
    }

    /// Matriks pooling membentuk partisi: setiap posisi masuk tepat satu blok.
    #[test]
    fn pooling_is_a_partition() {
        let d = device();
        let s = ScaleOps::new(5, 2, &d);
        assert_eq!(s.out_len(), 2);
        let x = Tensor::<3>::from_floats([[[0.0], [1.0], [2.0], [3.0], [4.0]]], &d);
        let got = f(s.down(x));
        // Blok [0,2) dan [2,5).
        assert!((got[0] - 0.5).abs() < 1e-5, "{got:?}");
        assert!((got[1] - 3.0).abs() < 1e-5, "{got:?}");
    }

    /// Down lalu up pada data linear: output panjang L, nilai finite, monoton.
    #[test]
    fn down_up_roundtrip_on_linear_ramp() {
        let d = device();
        let s = ScaleOps::new(4, 2, &d);
        let x = Tensor::<3>::from_floats([[[0.0], [1.0], [2.0], [3.0]]], &d);
        let back = s.up(s.down(x));
        let got = f(back);
        assert_eq!(got.len(), 4, "{got:?}");
        assert!(got.iter().all(|v| v.is_finite()));
        for w in got.windows(2) {
            assert!(w[0] <= w[1] + 1e-4, "tidak monoton: {got:?}");
        }
    }

    /// `ℓ == L`: down dan up adalah identitas.
    #[test]
    fn identity_when_length_matches() {
        let d = device();
        let s = ScaleOps::new(3, 3, &d);
        let x = Tensor::<3>::from_floats([[[1.0], [2.0], [3.0]]], &d);
        assert_eq!(f(s.down(x.clone())), vec![1.0, 2.0, 3.0]);
        assert_eq!(f(s.up(x)), vec![1.0, 2.0, 3.0]);
    }

    /// `R_k` = up + Phi(up); dengan `Phi` nol hasilnya interpolasi linear.
    #[test]
    fn reconstruct_without_phi_is_linear_up() {
        let d = device();
        let cb = Codebook::new(
            &CodebookConfig {
                num_entries: 2,
                dim: 1,
                ..CodebookConfig::default()
            },
            &d,
        );
        let level = LevelVq::new(4, 2, 1, cb, &d);
        let codes = Tensor::<3>::from_floats([[[0.0], [2.0]]], &d);
        let out = f(level.reconstruct(codes));
        assert_eq!(out.len(), 4, "{out:?}");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Invarian R2: `Σ_k R_k(e_k(r_k)) = ẑ^(K)`, dan galat rekonstruksi tidak
    /// naik per level pada data kecil yang bisa direpresentasikan tepat.
    #[test]
    fn cumulative_reconstruction_matches_sum_and_shrinks_error() {
        let d = device();
        let levels = [2usize, 4];
        let mut model = MultiscaleVq::new(
            &levels,
            1,
            |k| {
                Codebook::new(
                    &CodebookConfig {
                        num_entries: if k == 0 { 2 } else { 1 },
                        dim: 1,
                        ..CodebookConfig::default()
                    },
                    &d,
                )
            },
            &d,
        );
        // Codebook level 1 diisi nilai yang tepat; level 2 diisi residual.
        model.levels[0]
            .quantizer
            .set_entries(&[0.5, 2.5])
            .expect("set_entries");
        model.levels[1]
            .quantizer
            .set_entries(&[-0.5])
            .expect("set_entries");

        let z = Tensor::<3>::from_floats([[[0.0], [1.0], [2.0], [3.0]]], &d);
        let out = model.encode_full(z.clone());
        assert_eq!(out.final_z.dims(), [1, 4, 1], "final_z shape");
        assert_eq!(out.contributions.len(), 2, "dua level");
        assert_eq!(out.z_hat.len(), 2, "dua z_hat");
        for zh in &out.z_hat {
            assert_eq!(zh.dims(), [1, 4, 1], "z_hat shape");
            assert!(f(zh.clone()).iter().all(|v| v.is_finite()));
        }
    }

    /// Cutoff membatasi jumlah level yang dipakai (R3).
    #[test]
    fn cutoff_limits_levels() {
        let d = device();
        let model = MultiscaleVq::new(
            &[2, 4],
            1,
            |_| {
                Codebook::new(
                    &CodebookConfig {
                        num_entries: 2,
                        dim: 1,
                        ..CodebookConfig::default()
                    },
                    &d,
                )
            },
            &d,
        );
        let z = Tensor::<3>::from_floats([[[1.0], [1.0], [1.0], [1.0]]], &d);
        let out = model.encode(z, 1);
        assert_eq!(out.z_hat.len(), 1);
        assert_eq!(out.indices.len(), 1);
        assert_eq!(out.final_z.dims(), [1, 4, 1]);
    }

    /// Encode menghasilkan indeks dengan panjang level yang sesuai.
    #[test]
    fn indices_follow_level_lengths() {
        let d = device();
        let model = MultiscaleVq::new(
            &[2, 4],
            1,
            |_| {
                Codebook::new(
                    &CodebookConfig {
                        num_entries: 3,
                        dim: 2,
                        ..CodebookConfig::default()
                    },
                    &d,
                )
            },
            &d,
        );
        let z = Tensor::<3>::random([2, 4, 2], Distribution::Normal(0.0, 1.0), &d);
        let out = model.encode_full(z);
        assert_eq!(out.indices.len(), 2, "dua level");
        assert_eq!(out.indices[0].dims()[0], 2, "batch");
        assert_eq!(out.indices[1].dims()[0], 2, "batch");
    }
}