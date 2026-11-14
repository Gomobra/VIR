//! Tokenizer gambar multiscale FSQ (F6.3): level 0 kasar → level halus.
//!
//! Tiap level `k` memakai `Down_k` = average pooling area dan `R_k` = nearest
//! neighbor (ADR-009 §2), sama seperti `vir-vq::multiscale`: residual
//! diturunkan, dikuantisasi FSQ, dinaikkan kembali, lalu dikurangkan dari
//! residual. Rekonstruksi adalah jumlahan semua level.

use vir_common::VirError;

use crate::image::FsqQuantizer;
use crate::pool;

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

