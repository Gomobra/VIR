//! Tokenizer gambar multiscale FSQ (F6.3).

use vir_common::VirError;

/// FSQ: proyeksi ke `n` dimensi, bulatkan tiap dimensi ke `L_i` nilai.
#[derive(Debug, Clone)]
pub struct FsqQuantizer {
    pub levels: Vec<u32>,
    pub proj: Vec<f32>, // [n, d]
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

    pub fn codebook_size(&self) -> u32 {
        self.levels.iter().product()
    }

    /// Quantize z [N, d] → (codes [N, n], quantized [N, d]).
    pub fn quantize(&self, z: &[f32]) -> Result<(Vec<u32>, Vec<f32>), VirError> {
        let n = z.len() / self.d;
        if n * self.d != z.len() {
            return Err(VirError::Config("panjang z tidak kelipatan d".into()));
        }
        let mut codes = Vec::with_capacity(n * self.n);
        let mut q = vec![0.0f32; n * self.d];
        for row in 0..n {
            // proyeksi
            let mut proj = vec![0.0f32; self.n];
            for (j, pj) in proj.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for k in 0..self.d {
                    acc += self.proj[j * self.d + k] * z[row * self.d + k];
                }
                *pj = acc;
            }
            // bound + round
            for (j, p) in proj.iter().enumerate() {
                let l = self.levels[j];
                let half_l = (l as f32 - 1.0) * (1.0 + 1e-3) / 2.0;
                let offset = if l % 2 == 0 { 0.5 } else { 0.0 };
                let shift = (offset / half_l).atanh();
                let bounded = (p + shift).tanh() * half_l - offset;
                let rounded = bounded.round().clamp(-(half_l - offset), half_l - offset);
                let code = (rounded + half_l - offset) as u32;
                codes.push(code);
                for k in 0..self.d {
                    q[row * self.d + k] += rounded * self.proj[j * self.d + k];
                }
            }
        }
        Ok((codes, q))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsq_codebook_size() {
        let q = FsqQuantizer::new(vec![8, 5, 5, 5, 5], 4);
        assert_eq!(q.codebook_size(), 8 * 5 * 5 * 5 * 5);
    }

    #[test]
    fn fsq_quantize_shape() {
        let q = FsqQuantizer::new(vec![5, 5], 4);
        let z = vec![0.1f32; 8];
        let (codes, qz) = q.quantize(&z).unwrap();
        assert_eq!(codes.len(), 2 * 2);
        assert_eq!(qz.len(), 8);
    }
}
