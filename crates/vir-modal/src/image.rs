//! Kuantizer FSQ untuk jalur kode gambar (F6.3).
//!
//! FSQ (arXiv 2309.15505): proyeksi ke `n` dimensi, bulatkan tiap dimensi ke
//! `L_i` nilai. Tanpa codebook yang dipelajari dan tanpa *codebook collapse*.
//! Penurunan/peningkatan antar level ada di [`crate::multiscale`], dan decoder
//! piksel di [`crate::decode`].

use vir_common::VirError;

use crate::connector::ModalTokenizer;

/// FSQ: proyeksi ke `n` dimensi, bulatkan tiap dimensi ke `L_i` nilai.
#[derive(Debug, Clone)]
pub struct FsqQuantizer {
    pub levels: Vec<u32>,
    /// Bobot proyeksi `[n, d]` row-major; nol = proyeksi mati (pilih
    /// [`FsqQuantizer::identity`] untuk uji rekonstruksi).
    pub proj: Vec<f32>,
    pub n: usize,
    pub d: usize,
}

impl FsqQuantizer {
    pub fn new(levels: Vec<u32>, d: usize) -> Self {
        let n = levels.len();
        Self {
            levels,
            proj: vec![0.0; n * d],
            n,
            d,
        }
    }

    /// Proyeksi identitas (`n == d`) — rekonstruksi dapat diuji langsung.
    pub fn identity(levels: Vec<u32>) -> Self {
        let n = levels.len();
        let mut q = Self::new(levels, n);
        for i in 0..n {
            q.proj[i * n + i] = 1.0;
        }
        q
    }

    pub fn validate(&self) -> Result<(), VirError> {
        if self.levels.is_empty() {
            return Err(VirError::Config("levels FSQ kosong".into()));
        }
        if self.d == 0 {
            return Err(VirError::Config("dimensi laten FSQ 0".into()));
        }
        if self.proj.len() != self.n * self.d {
            return Err(VirError::Config(format!(
                "proj {} != {}×{}",
                self.proj.len(),
                self.n,
                self.d
            )));
        }
        for (j, level) in self.levels.iter().enumerate() {
            if *level < 2 {
                return Err(VirError::Config(format!("level {j} = {level} < 2")));
            }
        }
        Ok(())
    }

    /// Ukuran codebook implisit `∏ L_i`.
    pub fn codebook_size(&self) -> u32 {
        self.levels.iter().product()
    }

    /// Indeks integer satu baris kode (untuk head klasifikasi gabungan).
    pub fn code_index(&self, codes: &[u32]) -> Result<u32, VirError> {
        if codes.len() != self.n {
            return Err(VirError::Config(format!(
                "kode {} != n {}",
                codes.len(),
                self.n
            )));
        }
        let mut index = 0u32;
        for (j, code) in codes.iter().enumerate() {
            if *code >= self.levels[j] {
                return Err(VirError::Config(format!(
                    "kode {code} di luar level {}",
                    self.levels[j]
                )));
            }
            index = index * self.levels[j] + code;
        }
        Ok(index)
    }

    /// Kebalikan [`FsqQuantizer::code_index`].
    pub fn index_codes(&self, mut index: u32) -> Result<Vec<u32>, VirError> {
        if index >= self.codebook_size() {
            return Err(VirError::Config(format!(
                "indeks {index} di luar codebook {}",
                self.codebook_size()
            )));
        }
        let mut codes = vec![0u32; self.n];
        for j in (0..self.n).rev() {
            codes[j] = index % self.levels[j];
            index /= self.levels[j];
        }
        Ok(codes)
    }

    /// Nilai kuat untuk dimensi `j` (setengah lebar setelah pembatasan).
    fn half_l(&self, j: usize) -> f32 {
        self.levels[j] as f32 * (1.0 + FSQ_EPS) / 2.0
    }

    fn offset(&self, j: usize) -> f32 {
        if self.levels[j] % 2 == 0 { 0.5 } else { 0.0 }
    }

    /// Basis kode bulat: `L/2` (genap) atau `(L−1)/2` (ganjil) — sama dengan
    /// `half_width` pada implementasi FSQ acuan, sehingga kode ↔ nilai utuh.
    fn level_base(&self, j: usize) -> f32 {
        (self.levels[j] / 2) as f32
    }

    /// `bound(v)` untuk dimensi `j`: `tanh` yang memetakan ℝ ke rentang kode
    /// (dipakai juga oleh sisi tensor agar batasnya identik).
    pub fn bound(&self, j: usize, v: f32) -> f32 {
        let half = self.half_l(j);
        let offset = self.offset(j);
        let shift = (offset / half).atanh();
        (v + shift).tanh() * half - offset
    }

    /// `round(bound(v))` untuk dimensi `j` (dijepit ke rentang kode).
    fn round_level(&self, j: usize, v: f32) -> f32 {
        let lo = -self.level_base(j);
        let hi = self.levels[j] as f32 - 1.0 - self.level_base(j);
        self.bound(j, v).round().clamp(lo, hi)
    }

    /// Nilai kuat dari sebuah kode (kebalikan pembulatan).
    fn level_value(&self, j: usize, code: u32) -> Result<f32, VirError> {
        let level = self.levels[j];
        if code >= level {
            return Err(VirError::Config(format!(
                "kode {code} di luar level {level}"
            )));
        }
        Ok(code as f32 - self.level_base(j))
    }

    /// Quantize `z [N, d]` → (kode datar `[N·n]`, kuantisasi `[N, d]`).
    pub fn quantize(&self, z: &[f32]) -> Result<(Vec<u32>, Vec<f32>), VirError> {
        self.validate()?;
        let rows = self.rows(z)?;
        let mut codes = Vec::with_capacity(rows * self.n);
        let mut q = vec![0.0f32; rows * self.d];
        for (row, src) in z.chunks_exact(self.d).enumerate() {
            let dst = &mut q[row * self.d..(row + 1) * self.d];
            for j in 0..self.n {
                let weights = &self.proj[j * self.d..(j + 1) * self.d];
                let projected = dot(weights, src);
                let rounded = self.round_level(j, projected);
                codes.push((rounded + self.level_base(j)) as u32);
                for (slot, weight) in dst.iter_mut().zip(weights.iter()) {
                    *slot += rounded * weight;
                }
            }
        }
        Ok((codes, q))
    }

    /// Kode → vektor kuat `[positions, d]` (sisi sintesis `projᵀ`).
    pub fn dequantize(&self, codes: &[u32]) -> Result<Vec<f32>, VirError> {
        self.validate()?;
        if codes.len() % self.n != 0 {
            return Err(VirError::Config(format!(
                "kode {} bukan kelipatan n {}",
                codes.len(),
                self.n
            )));
        }
        let rows = codes.len() / self.n;
        let mut out = vec![0.0f32; rows * self.d];
        for (row, dst) in out.chunks_exact_mut(self.d).enumerate() {
            for j in 0..self.n {
                let value = self.level_value(j, codes[row * self.n + j])?;
                let weights = &self.proj[j * self.d..(j + 1) * self.d];
                for (slot, weight) in dst.iter_mut().zip(weights.iter()) {
                    *slot += value * weight;
                }
            }
        }
        Ok(out)
    }

    /// Quantize lalu dequantize (rekonstruksi langsung satu tahap).
    pub fn roundtrip(&self, z: &[f32]) -> Result<(Vec<u32>, Vec<f32>), VirError> {
        let (codes, _) = self.quantize(z)?;
        let q = self.dequantize(&codes)?;
        Ok((codes, q))
    }

    fn rows(&self, z: &[f32]) -> Result<usize, VirError> {
        if self.d == 0 || z.len() % self.d != 0 {
            return Err(VirError::Config(format!(
                "panjang z {} bukan kelipatan d {}",
                z.len(),
                self.d
            )));
        }
        Ok(z.len() / self.d)
    }
}

/// `ε` pembatas FSQ (kode acuan).
pub const FSQ_EPS: f32 = 1e-3;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

impl ModalTokenizer for FsqQuantizer {
    fn tokenize(&self, data: &[f32]) -> Result<Vec<u32>, VirError> {
        let (codes, _) = self.quantize(data)?;
        let rows = self.rows(data)?;
        let mut out = Vec::with_capacity(rows);
        for row in 0..rows {
            out.push(self.code_index(&codes[row * self.n..(row + 1) * self.n])?);
        }
        Ok(out)
    }

    fn decode(&self, codes: &[u32]) -> Result<Vec<f32>, VirError> {
        let mut flat = Vec::with_capacity(codes.len() * self.n);
        for code in codes {
            flat.extend_from_slice(&self.index_codes(*code)?);
        }
        self.dequantize(&flat)
    }
}
