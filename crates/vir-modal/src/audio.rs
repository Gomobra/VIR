//! Residual VQ untuk jalur kode audio (F6.4).
//!
//! [`Rvq`] adalah kernel kuantisasi: `K` level codebook, tiap level
//! mengkuantisasi residual dari level sebelumnya, dan rekonstruksi adalah
//! jumlahan semua level. Konfigurasi codec (laju, hop, bitrate) ada di
//! [`crate::snac`]; tokenizer gambar multiscale memakai operator turun/naik
//! yang sama ([`crate::pool`]) agar kedua jalur konsisten.

use vir_common::VirError;

use crate::pool;

/// Residual VQ: K level codebook, tiap level kuantisasi residual.
#[derive(Debug, Clone)]
pub struct Rvq {
    /// `[level][entry][dim]`; level 0 = paling kasar.
    pub codebooks: Vec<Vec<Vec<f32>>>,
    pub dims: Vec<usize>,
}

impl Rvq {
    /// Codebook nol sebanyak `levels` level (bobot dimuat dari checkpoint).
    pub fn new(codebook_size: usize, dim: usize, levels: usize) -> Self {
        Self {
            codebooks: vec![vec![vec![0.0; dim]; codebook_size]; levels],
            dims: vec![dim; levels],
        }
    }

    /// Jumlah level (sama dengan `codebooks.len()`).
    pub fn levels(&self) -> usize {
        self.codebooks.len()
    }

    /// Dimensi satu entri codebook (level pertama), `0` bila kosong.
    pub fn dim(&self) -> usize {
        self.dims.first().copied().unwrap_or(0)
    }

    /// Ukuran codebook (entri per level), `0` bila kosong.
    pub fn codebook_size(&self) -> usize {
        self.codebooks.first().map_or(0, Vec::len)
    }

    pub fn validate(&self) -> Result<(), VirError> {
        let dim = self.dim();
        if self.codebooks.is_empty() {
            return Err(VirError::Config("codebook RVQ kosong".into()));
        }
        if dim == 0 {
            return Err(VirError::Config("dimensi codebook RVQ 0".into()));
        }
        if self.dims.iter().any(|d| *d != dim) {
            return Err(VirError::Config("dimensi antar level beda".into()));
        }
        for (level, entries) in self.codebooks.iter().enumerate() {
            if entries.is_empty() {
                return Err(VirError::Config(format!("codebook level {level} kosong")));
            }
            if entries.iter().any(|e| e.len() != dim) {
                return Err(VirError::Config(format!(
                    "entri codebook level {level} bukan berdimensi {dim}"
                )));
            }
        }
        Ok(())
    }

    /// Cari entri terdekat satu baris pada `level` (kuadrat terkecil).
    pub fn encode_level(
        &self,
        level: usize,
        z: &[f32],
        n: usize,
        dim: usize,
    ) -> Result<Vec<u32>, VirError> {
        self.validate()?;
        if dim != self.dim() {
            return Err(VirError::Config(format!(
                "dim {dim} != dimensi codebook {}",
                self.dim()
            )));
        }
        if n * dim != z.len() {
            return Err(VirError::Config(format!(
                "panjang z {} != {n}×{dim}",
                z.len()
            )));
        }
        let entries = self
            .codebooks
            .get(level)
            .ok_or(VirError::Config(format!("level {level} di luar codebook")))?;
        let mut codes = vec![0u32; n];
        for (i, code) in codes.iter_mut().enumerate() {
            let row = &z[i * dim..(i + 1) * dim];
            *code = nearest_entry(entries, row) as u32;
        }
        Ok(codes)
    }

    /// Kode satu level → vektor kuat `[n, dim]`.
    pub fn decode_level(
        &self,
        level: usize,
        codes: &[u32],
        dim: usize,
    ) -> Result<Vec<f32>, VirError> {
        self.validate()?;
        if dim != self.dim() {
            return Err(VirError::Config(format!(
                "dim {dim} != dimensi codebook {}",
                self.dim()
            )));
        }
        let entries = self
            .codebooks
            .get(level)
            .ok_or(VirError::Config(format!("level {level} di luar codebook")))?;
        let mut out = vec![0.0f32; codes.len() * dim];
        for (i, code) in codes.iter().enumerate() {
            let Some(entry) = entries.get(*code as usize) else {
                return Err(VirError::Config(format!(
                    "kode {code} di luar codebook level {level}"
                )));
            };
            out[i * dim..(i + 1) * dim].copy_from_slice(entry);
        }
        Ok(out)
    }

    /// Encode `z [n, dim]` → satu `Vec<u32>` kode per level (residual).
    pub fn encode(&self, z: &[f32], n: usize, dim: usize) -> Result<Vec<Vec<u32>>, VirError> {
        self.validate()?;
        let levels = self.levels();
        let mut residual = z.to_vec();
        let mut z_hat = vec![0.0f32; z.len()];
        let mut codes = Vec::with_capacity(levels);
        for level in 0..levels {
            let level_codes = self.encode_level(level, &residual, n, dim)?;
            let q = self.decode_level(level, &level_codes, dim)?;
            pool::add_assign(&mut z_hat, &q)?;
            pool::sub_assign(&mut residual, &q)?;
            codes.push(level_codes);
        }
        Ok(codes)
    }

    /// Decode kode `[K][n]` → `z [n, dim]` (jumlahan semua level).
    pub fn decode(&self, codes: &[Vec<u32>], n: usize, dim: usize) -> Result<Vec<f32>, VirError> {
        self.validate()?;
        let mut out = vec![0.0f32; n * dim];
        for (level, level_codes) in codes.iter().enumerate() {
            if level_codes.len() != n {
                return Err(VirError::Config(format!(
                    "level {level} punya {} kode != {n}",
                    level_codes.len()
                )));
            }
            let q = self.decode_level(level, level_codes, dim)?;
            pool::add_assign(&mut out, &q)?;
        }
        Ok(out)
    }

    /// Bitrate `Σ_k laju_k × log2|V|` untuk laju per level (Hz).
    pub fn bitrate(&self, rates: &[f32]) -> Result<f32, VirError> {
        self.validate()?;
        if rates.len() != self.levels() {
            return Err(VirError::Config(format!(
                "laju {} != level {}",
                rates.len(),
                self.levels()
            )));
        }
        let bits = (self.codebook_size() as f32).log2();
        Ok(rates.iter().sum::<f32>() * bits)
    }
}

fn nearest_entry(entries: &[Vec<f32>], row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::MAX;
    for (j, entry) in entries.iter().enumerate() {
        let d: f32 = row
            .iter()
            .zip(entry.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if d < best_d {
            best_d = d;
            best = j;
        }
    }
    best
}

