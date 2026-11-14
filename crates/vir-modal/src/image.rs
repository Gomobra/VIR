//! Kuantizer FSQ untuk jalur kode gambar (F6.3).
//!
//! FSQ (arXiv 2309.15505): proyeksi ke `n` dimensi, bulatkan tiap dimensi ke
//! `L_i` nilai. Tanpa codebook yang dipelajari dan tanpa *codebook collapse*.
//! Penurunan/peningkatan antar level ada di [`crate::multiscale`], dan decoder
//! piksel di [`crate::decode`].

use vir_common::VirError;

use crate::connector::ModalTokenizer;
use crate::pool;

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

/// Tokenizer gambar multiscale FSQ: level 0 kasar → level terakhir halus.
#[derive(Debug, Clone)]
pub struct MultiScaleTokenizer {
    pub fsq: FsqQuantizer,
    /// Jumlah posisi per level, kasar → halus (naik 4× = 2×2 patch).
    pub lengths: Vec<usize>,
}

impl MultiScaleTokenizer {
    pub fn new(fsq: FsqQuantizer, lengths: Vec<usize>) -> Result<Self, VirError> {
        fsq.validate()?;
        if lengths.is_empty() {
            return Err(VirError::Config("daftar level kosong".into()));
        }
        for pair in lengths.windows(2) {
            if pair[0] >= pair[1] {
                return Err(VirError::Config(format!(
                    "panjang level harus naik: {:?}",
                    lengths
                )));
            }
        }
        Ok(Self { fsq, lengths })
    }

    /// Panjang level geometris untuk `grid` posisi halus dengan `levels` level
    /// (tiap level 4× lebih halus; level terakhir = `grid`).
    pub fn grid_lengths(grid: usize, levels: usize) -> Result<Vec<usize>, VirError> {
        if grid == 0 || levels == 0 {
            return Err(VirError::Config("grid/level 0".into()));
        }
        if grid < levels {
            return Err(VirError::Config(format!(
                "grid {grid} terlalu kecil untuk {levels} level"
            )));
        }
        let finest_exp = (levels - 1) as i32;
        let mut lengths = Vec::with_capacity(levels);
        for k in 0..levels {
            let exp = k as i32 - finest_exp;
            let raw = grid as f64 * 4f64.powi(exp);
            lengths.push((raw.round() as usize).clamp(1, grid));
        }
        for k in 1..lengths.len() {
            lengths[k] = lengths[k].max(lengths[k - 1] + 1);
        }
        let last = lengths.len() - 1;
        lengths[last] = grid;
        Ok(lengths)
    }

    /// Jumlah posisi level paling halus.
    pub fn finest(&self) -> usize {
        self.lengths[self.lengths.len() - 1]
    }

    /// Turun `n → m` posisi dengan average pooling area (per dimensi).
    pub fn down(&self, x: &[f32], n: usize, m: usize) -> Result<Vec<f32>, VirError> {
        pool::avg_pool_rows(x, n, m, self.fsq.d)
    }

    /// Naik `m → n` posisi dengan nearest neighbor (per dimensi).
    pub fn up(&self, x: &[f32], m: usize, n: usize) -> Result<Vec<f32>, VirError> {
        pool::nearest_up_rows(x, m, n, self.fsq.d)
    }

    /// Encode latent halus `[finest, d]` → satu `Vec<u32>` kode per level.
    pub fn encode(&self, z: &[f32]) -> Result<Vec<Vec<u32>>, VirError> {
        let finest = self.finest();
        let d = self.fsq.d;
        if z.len() != finest * d {
            return Err(VirError::Config(format!(
                "panjang z {} != {finest}×{d}",
                z.len()
            )));
        }
        let mut residual = z.to_vec();
        let mut z_hat = vec![0.0f32; z.len()];
        let mut levels = Vec::with_capacity(self.lengths.len());
        for len in &self.lengths {
            let coarse = self.down(&residual, finest, *len)?;
            let (codes, _) = self.fsq.quantize(&coarse)?;
            let q = self.fsq.dequantize(&codes)?;
            let up = self.up(&q, *len, finest)?;
            pool::add_assign(&mut z_hat, &up)?;
            for (res, (target, estimate)) in residual.iter_mut().zip(z.iter().zip(z_hat.iter())) {
                *res = target - estimate;
            }
            levels.push(codes);
        }
        Ok(levels)
    }

    /// Decode kode berlevel → latent halus `[finest, d]`.
    pub fn decode(&self, levels: &[Vec<u32>]) -> Result<Vec<f32>, VirError> {
        if levels.len() != self.lengths.len() {
            return Err(VirError::Config(format!(
                "level {} != panjang level {}",
                levels.len(),
                self.lengths.len()
            )));
        }
        let finest = self.finest();
        let d = self.fsq.d;
        let mut out = vec![0.0f32; finest * d];
        for (codes, len) in levels.iter().zip(self.lengths.iter()) {
            if codes.len() != len * self.fsq.n {
                return Err(VirError::Config(format!(
                    "kode level {} = {} != {}×{}",
                    len,
                    codes.len(),
                    len,
                    self.fsq.n
                )));
            }
            let q = self.fsq.dequantize(codes)?;
            let up = self.up(&q, *len, finest)?;
            for (acc, value) in out.iter_mut().zip(up.iter()) {
                *acc += value;
            }
        }
        Ok(out)
    }
}
