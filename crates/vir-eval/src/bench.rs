//! F4.5 / F5.4 — konfigurasi eval, laporan RULER (akurasi vs panjang, panjang
//! efektif), dan tabel metrik antar checkpoint. Tidak menjalankan model: pemanggil
//! menyediakan fungsi jawaban sehingga suite dapat dipakai dengan model apa pun.

use std::collections::BTreeMap;

use crate::ruler::{RulerCategory, RulerSample, effective_context_length, gen_ruler, match_score};

/// Deret panjang konteks baku RULER (4K..128K); VIR menambah 1M/10M di F5.5.
pub const DEFAULT_LENGTHS: [usize; 6] = [4_096, 8_192, 16_384, 32_768, 65_536, 131_072];

/// Ambang kualitatif baku RULER (kinerja Llama2-7B pada 4K), dipakai bawaan VIR.
pub const DEFAULT_THRESHOLD: f64 = 0.856;

/// Konfigurasi suite eval: versi, seed, panjang, jumlah contoh, kategori, ambang.
#[derive(Clone, Debug)]
pub struct EvalConfig {
    pub suite_version: String,
    pub seed: u64,
    pub lengths: Vec<usize>,
    pub samples_per_length: usize,
    pub categories: Vec<RulerCategory>,
    pub threshold: f64,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            suite_version: "vir-ruler-v1".to_string(),
            seed: 0x5EED,
            lengths: DEFAULT_LENGTHS.to_vec(),
            samples_per_length: 500,
            categories: RulerCategory::ALL.to_vec(),
            threshold: DEFAULT_THRESHOLD,
        }
    }
}

/// Satu titik kurva: panjang konteks dan skor rata-rata (0..=1) pada panjang itu.
#[derive(Clone, Copy, Debug)]
pub struct LengthScore {
    pub length: usize,
    pub score: f64,
}

/// Bobot rata-rata terhadap panjang, mensimulasikan distribusi panjang pemakaian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weighting {
    /// Rata-rata biasa.
    Uniform,
    /// Bobot naik terhadap panjang (pemakaian condong ke konteks panjang).
    Ascending,
    /// Bobot turun terhadap panjang (pemakaian condong ke konteks pendek).
    Descending,
}

/// Laporan RULER: skor per kategori dan agregat, plus turunan (panjang efektif).
#[derive(Clone, Debug)]
pub struct RulerReport {
    pub version: String,
    pub threshold: f64,
    pub per_category: BTreeMap<String, Vec<LengthScore>>,
    pub overall: Vec<LengthScore>,
}

impl RulerReport {
    /// Panjang efektif agregat: terbesar dengan skor rata-rata di atas ambang.
    pub fn effective_context_length(&self) -> Option<usize> {
        let lengths: Vec<usize> = self.overall.iter().map(|p| p.length).collect();
        let scores: Vec<f64> = self.overall.iter().map(|p| p.score).collect();
        effective_context_length(&lengths, &scores, self.threshold)
    }

    /// Skor agregat pada panjang tertentu, bila ada.
    pub fn score_at(&self, length: usize) -> Option<f64> {
        self.overall
            .iter()
            .find(|p| p.length == length)
            .map(|p| p.score)
    }

    /// Rata-rata berbobot terhadap panjang sesuai `mode`.
    pub fn weighted_score(&self, mode: Weighting) -> f64 {
        let n = self.overall.len();
        if n == 0 {
            return 0.0;
        }
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (i, point) in self.overall.iter().enumerate() {
            let w = match mode {
                Weighting::Uniform => 1.0,
                Weighting::Ascending => (i + 1) as f64,
                Weighting::Descending => (n - i) as f64,
            };
            num += w * point.score;
            den += w;
        }
        if den == 0.0 { 0.0 } else { num / den }
    }

    /// Tabel markdown: baris panjang, kolom kategori + rata-rata.
    pub fn render_table(&self) -> String {
        let mut table = MetricTable::new(&["Panjang", "Rata-rata"]);
        for name in self.per_category.keys() {
            table.add_column(name);
        }
        for (i, point) in self.overall.iter().enumerate() {
            let mut cells = vec![point.length.to_string(), fmt_num(point.score)];
            for series in self.per_category.values() {
                let value = series.get(i).map(|p| p.score).unwrap_or(0.0);
                cells.push(fmt_num(value));
            }
            table.push_row(&cells);
        }
        table.render()
    }
}

/// Jalankan suite RULER: untuk tiap (kategori, panjang), rata-rata skor jawaban
/// dari `answer` atas `samples_per_length` contoh deterministik.
pub fn run_ruler_suite<F>(config: &EvalConfig, mut answer: F) -> RulerReport
where
    F: FnMut(&RulerSample) -> Vec<u32>,
{
    let mut per_category: BTreeMap<String, Vec<LengthScore>> = BTreeMap::new();
    let mut overall_sum = vec![0.0f64; config.lengths.len()];

    for (ci, &category) in config.categories.iter().enumerate() {
        let mut series = Vec::with_capacity(config.lengths.len());
        for (li, &length) in config.lengths.iter().enumerate() {
            let mut sum = 0.0f64;
            for i in 0..config.samples_per_length {
                let seed = sample_seed(config.seed, length, ci, i);
                let sample = gen_ruler(seed, length, category);
                sum += match_score(&sample.answer, &answer(&sample));
            }
            let mean = sample_mean(sum, config.samples_per_length);
            series.push(LengthScore {
                length,
                score: mean,
            });
            overall_sum[li] += mean;
        }
        per_category.insert(category.name().to_string(), series);
    }

    let divisor = config.categories.len().max(1) as f64;
    let overall = config
        .lengths
        .iter()
        .enumerate()
        .map(|(li, &length)| LengthScore {
            length,
            score: overall_sum[li] / divisor,
        })
        .collect();

    RulerReport {
        version: config.suite_version.clone(),
        threshold: config.threshold,
        per_category,
        overall,
    }
}

/// Rata-rata aman terhadap pembagian nol.
fn sample_mean(sum: f64, n: usize) -> f64 {
    if n == 0 { 0.0 } else { sum / n as f64 }
}

/// Seed deterministik per (panjang, kategori, contoh), tercampur dari seed dasar.
fn sample_seed(base: u64, length: usize, category_idx: usize, sample_idx: usize) -> u64 {
    base ^ (length as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ ((category_idx as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        ^ ((sample_idx as u64 + 1).wrapping_mul(0x94D0_49BB_1331_11EB))
}

/// Kurva pass@k untuk beberapa `k` sekaligus (F4.5).
pub fn pass_at_k_curve(n: usize, correct: usize, ks: &[usize]) -> Vec<(usize, f64)> {
    ks.iter()
        .map(|&k| (k, crate::pass_at_k(n, correct, k)))
        .collect()
}

/// Tabel metrik bertipe teks: header + baris sel, dirender sebagai markdown.
#[derive(Clone, Debug, Default)]
pub struct MetricTable {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl MetricTable {
    /// Tabel dengan kolom awal.
    pub fn new(headers: &[&str]) -> Self {
        Self {
            headers: headers.iter().map(|h| h.to_string()).collect(),
            rows: Vec::new(),
        }
    }

    /// Tambah kolom di akhir (setiap baris akan diisi nilai kosong bila kurang).
    pub fn add_column(&mut self, name: &str) {
        self.headers.push(name.to_string());
    }

    /// Tambah satu baris sel.
    pub fn push_row(&mut self, cells: &[String]) {
        let mut row: Vec<String> = cells.to_vec();
        row.resize(self.headers.len(), String::new());
        self.rows.push(row);
    }

    /// Jumlah baris data.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Tabel kosong (belum ada baris).
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Render tabel markdown (header, garis pemisah, baris).
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("| ");
        out.push_str(&self.headers.join(" | "));
        out.push_str(" |\n");
        out.push_str("| ");
        out.push_str(&vec!["---"; self.headers.len()].join(" | "));
        out.push_str(" |\n");
        for row in &self.rows {
            out.push_str("| ");
            out.push_str(&row.join(" | "));
            out.push_str(" |\n");
        }
        out
    }
}

/// Format angka metrik dengan empat desimal.
pub fn fmt_num(x: f64) -> String {
    format!("{x:.4}")
}
