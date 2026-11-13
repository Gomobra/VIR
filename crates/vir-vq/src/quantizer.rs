//! Trait kuantizer bersama untuk VQ-EMA (F1.4.a) dan FSQ (F1.4.f).
//!
//! PRD F1.4.f mewajibkan kuantizer alternatif berada "di belakang trait
//! `Quantizer` yang sama dengan `Codebook`", sehingga loop residual multiscale
//! ([`crate::multiscale`]) tidak perlu tahu kuantizer mana yang dipakai.
//!
//! Satu-satunya perbedaan yang terlihat dari luar:
//!
//! | | VQ-EMA | FSQ / RFSQ / iFSQ |
//! |---|---|---|
//! | codebook | dipelajari lewat EMA | implisit `∏ L_i` |
//! | loss komitmen | ya (`β·MSE`) | tidak ada |
//! | pembaruan state | EMA + restart kode mati | tidak ada |
//!
//! Karena itu [`Quantizer::update`] ada di trait: pemanggil (loop latihan) selalu
//! memanggilnya, dan implementasi FSQ memang tidak melakukan apa pun di sana.

use burn::module::Param;
use burn::prelude::*;
use burn::tensor::TensorData;
use rand::RngCore;

/// Hasil kuantisasi satu batch `u ∈ ℝ^{N×d}`.
#[derive(Debug, Clone)]
pub struct Quantized {
    /// Indeks kode per posisi, `[N]`.
    pub indices: Tensor<1, Int>,
    /// Kode terkuantisasi pada ruang laten penuh, `[N, d]`.
    pub quantized: Tensor<2>,
    /// Kode mentah pada ruang kode (setelah proyeksi ke `n` dimensi untuk FSQ),
    /// `[N, c]`. Untuk codebook, `codes == quantized`.
    pub codes: Tensor<2>,
}

/// Statistik satu codebook: utilisasi dan perplexity (NCM Gambar 16).
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizerStats {
    /// Ukuran codebook `|V|`.
    pub codes: usize,
    /// Jumlah entri yang terpakai minimal sekali.
    pub used: usize,
    /// `used / codes`.
    pub utilization: f32,
    /// `exp(H)` dengan `H` entropi histogram kode terhadap `|V|`.
    pub perplexity: f32,
    /// Histogram mentah (kosong bila data indeks tak dapat dibaca).
    pub counts: Vec<u64>,
}

impl QuantizerStats {
    /// Statistik dari histogram kode.
    pub fn from_counts(counts: &[u64]) -> Self {
        let codes = counts.len();
        let total: u64 = counts.iter().sum();
        let used = counts.iter().filter(|c| **c > 0).count();
        if total == 0 {
            return Self {
                codes,
                used,
                utilization: 0.0,
                perplexity: 0.0,
                counts: counts.to_vec(),
            };
        }
        let mut entropy = 0.0f64;
        for c in counts {
            if *c > 0 {
                let p = *c as f64 / total as f64;
                entropy -= p * p.ln();
            }
        }
        Self {
            codes,
            used,
            utilization: used as f32 / codes as f32,
            perplexity: entropy.exp() as f32,
            counts: counts.to_vec(),
        }
    }

    /// Statistik kosong (dipakai saat histogram tak dapat dihitung).
    pub fn empty(codes: usize) -> Self {
        Self {
            codes,
            used: 0,
            utilization: 0.0,
            perplexity: 0.0,
            counts: Vec::new(),
        }
    }

    /// Utilisasi dianggap sehat bila ≥ 50% entri terpakai.
    ///
    /// Pembanding arah NCM Tabel 1: utilisasi rata-rata 61,9–90,1%.
    pub fn is_healthy(&self) -> bool {
        self.utilization >= 0.5
    }

    /// Ringkasan JSON satu baris (untuk TUI / artefak evaluasi F1.4.e).
    pub fn to_json(&self) -> String {
        format!(
            "{{\"codes\":{},\"used\":{},\"utilization\":{:.4},\"perplexity\":{:.2}}}",
            self.codes, self.used, self.utilization, self.perplexity
        )
    }
}

/// Kuantizer diskret: `u ∈ ℝ^{N×d}` → indeks kode + kode terkuantisasi.
pub trait Quantizer {
    /// Ukuran codebook efektif `|V|`.
    fn num_codes(&self) -> usize;

    /// Dimensi laten penuh `d` yang diterima dan dikembalikan kuantizer.
    fn dim(&self) -> usize;

    /// Kuantisasi `u` (`[N, d]`) menjadi indeks dan kode.
    fn quantize(&self, u: Tensor<2>) -> Quantized;

    /// Kode terkuantisasi dari indeks `[N]` (dipakai generator R5–R6).
    fn lookup(&self, indices: Tensor<1, Int>) -> Tensor<2>;

    /// `quantize` + straight-through estimator: nilai keluaran sama dengan
    /// [`Quantizer::quantize`] tetapi gradien mengalir ke `u` (NCM §3.3).
    fn quantize_st(&self, u: Tensor<2>) -> (Quantized, Tensor<2>) {
        let q = self.quantize(u.clone());
        let st = u.clone() + (q.quantized.clone() - u).detach();
        (q, st)
    }

    /// Straight-through estimator saja (tanpa indeks).
    fn straight_through(&self, u: Tensor<2>) -> Tensor<2> {
        self.quantize_st(u).1
    }

    /// Perbarui state terpelajar dari batch: EMA + restart kode mati untuk
    /// [`crate::codebook::Codebook`]; tidak ada state untuk FSQ.
    fn update(&mut self, indices: &Tensor<1, Int>, u: &Tensor<2>, rng: &mut dyn RngCore);

    /// Statistik histogram kode dari indeks batch (F1.4.a dan F1.4.e).
    fn stats(&self, indices: &Tensor<1, Int>) -> QuantizerStats;
}

/// Ganti nilai parameter tanpa mengganti `ParamId` (agar optimizer dan
/// pemuatan checkpoint tetap cocok).
pub fn replace_param<const D: usize>(param: &mut Param<Tensor<D>>, value: Tensor<D>) {
    let current = core::mem::replace(param, Param::from_tensor(value.clone()));
    *param = current.map(|_| value);
}

/// Tensor `[n]` dari data host.
pub fn vec_to_tensor(values: &[f32], device: &Device) -> Tensor<1> {
    Tensor::<1>::from_floats(TensorData::new(values.to_vec(), [values.len()]), device)
}

/// Tensor `[n]` bulat dari data host.
pub fn vec_to_int_tensor(values: &[i64], device: &Device) -> Tensor<1, Int> {
    Tensor::<1, Int>::from_ints(TensorData::new(values.to_vec(), [values.len()]), device)
}

/// Tensor `[rows, cols]` dari data baris-major di host.
pub fn rows_to_tensor(rows: &[f32], cols: usize, device: &Device) -> Tensor<2> {
    let cols = cols.max(1);
    let n = rows.len() / cols;
    Tensor::<2>::from_floats(TensorData::new(rows.to_vec(), [n, cols]), device)
}

/// Tensor `[rows, cols]` bulat dari data baris-major di host.
pub fn rows_to_int_tensor(rows: &[i64], cols: usize, device: &Device) -> Tensor<2, Int> {
    let cols = cols.max(1);
    let n = rows.len() / cols;
    Tensor::<2, Int>::from_ints(TensorData::new(rows.to_vec(), [n, cols]), device)
}

/// Baca indeks `[N]` sebagai `Vec<i64>`; kosong bila konversi gagal.
pub fn indices_to_vec(indices: &Tensor<1, Int>) -> Vec<i64> {
    let data = indices.to_data();
    if let Ok(values) = data.try_to_vec::<i64>() {
        return values;
    }
    data.try_to_vec::<i32>()
        .map(|values| values.into_iter().map(i64::from).collect())
        .unwrap_or_default()
}

/// Histogram kode dari indeks batch; nol bila konversi data gagal.
pub fn histogram(indices: &Tensor<1, Int>, codes: usize) -> Vec<u64> {
    let mut counts = vec![0u64; codes];
    for value in indices_to_vec(indices) {
        if value >= 0 {
            let idx = value as usize;
            if idx < codes {
                counts[idx] += 1;
            }
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Device {
        Device::default()
    }

    #[test]
    fn stats_from_counts_matches_definition() {
        // Histogram [4, 4, 2] pada 3 entri: H = -(0.4 ln0.4 * 2 + 0.2 ln0.2).
        let stats = QuantizerStats::from_counts(&[4, 4, 2]);
        assert_eq!(stats.codes, 3);
        assert_eq!(stats.used, 3);
        assert!((stats.utilization - 1.0).abs() < 1e-6);
        let expected = (-(0.4f64 * 0.4f64.ln()) * 2.0 - 0.2f64 * 0.2f64.ln()).exp();
        assert!((stats.perplexity as f64 - expected).abs() < 1e-4);
    }

    #[test]
    fn stats_empty_batch_is_zero() {
        let stats = QuantizerStats::from_counts(&[0, 0, 0]);
        assert_eq!(stats.used, 0);
        assert_eq!(stats.utilization, 0.0);
        assert_eq!(stats.perplexity, 0.0);
        assert!(!stats.is_healthy());
    }

    #[test]
    fn histogram_counts_only_seen_codes() {
        let device = device();
        let idx = Tensor::<1, Int>::from_ints([0i64, 2, 2, 5], &device);
        assert_eq!(histogram(&idx, 4), vec![1, 0, 2, 0]);
    }

    #[test]
    fn histogram_ignores_out_of_range() {
        let device = device();
        let idx = Tensor::<1, Int>::from_ints([-1i64, 9], &device);
        assert_eq!(histogram(&idx, 4), vec![0, 0, 0, 0]);
    }

    #[test]
    fn json_is_parseable_shape() {
        let stats = QuantizerStats::from_counts(&[1, 1]);
        assert!(stats.to_json().contains("\"codes\":2"));
    }
}