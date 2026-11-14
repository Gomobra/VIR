//! Codec audio residual (SNAC-style RVQ) untuk F6.4.

use vir_common::VirError;

/// Residual VQ: K level codebook, tiap level kuantisasi residual.
#[derive(Debug, Clone)]
pub struct Rvq {
    pub codebooks: Vec<Vec<Vec<f32>>>, // [level][entry][dim]
    pub dims: Vec<usize>,
}

impl Rvq {
    pub fn new(codebook_size: usize, dim: usize, levels: usize) -> Self {
        Self {
            codebooks: vec![vec![vec![0.0; dim]; codebook_size]; levels],
            dims: vec![dim; levels],
        }
    }

    /// Encode z [N, dim] → codes [K][N] (indeks per level).
    pub fn encode(&self, z: &[f32], n: usize, dim: usize) -> Result<Vec<Vec<u32>>, VirError> {
        if n * dim != z.len() {
            return Err(VirError::Config("panjang z tidak kelipatan dim".into()));
        }
        let k = self.codebooks.len();
        let mut codes = vec![vec![0u32; n]; k];
        let mut residual = z.to_vec();
        for level in 0..k {
            let cb = &self.codebooks[level];
            for i in 0..n {
                let row = &residual[i * dim..(i + 1) * dim];
                let mut best = 0usize;
                let mut best_d = f32::MAX;
                for (j, e) in cb.iter().enumerate() {
                    let d: f32 = row
                        .iter()
                        .zip(e.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    if d < best_d {
                        best_d = d;
                        best = j;
                    }
                }
                codes[level][i] = best as u32;
                // update residual
                for t in 0..dim {
                    residual[i * dim + t] -= cb[best][t];
                }
            }
        }
        Ok(codes)
    }

    /// Decode codes [K][N] → z [N, dim].
    pub fn decode(&self, codes: &[Vec<u32>], n: usize, dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n * dim];
        for (level, level_codes) in codes.iter().enumerate() {
            let cb = &self.codebooks[level];
            for i in 0..n {
                let e = &cb[level_codes[i] as usize];
                for t in 0..dim {
                    out[i * dim + t] += e[t];
                }
            }
        }
        out
    }

    /// Bitrate: Σ_k rate_k × log2|V|.
    pub fn bitrate(&self, rates: &[f32]) -> f32 {
        let bits = (self.codebooks[0].len() as f32).log2();
        rates.iter().sum::<f32>() * bits
    }
}

/// SNAC 24 kHz: stride [4,2,1] → rates 12/23/47 Hz, codebook 4096.
pub fn snac24k() -> Rvq {
    Rvq::new(4096, 8, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rvq_roundtrip_shape() {
        let rvq = Rvq::new(16, 4, 2);
        let z = vec![0.1f32; 8];
        let codes = rvq.encode(&z, 2, 4).unwrap();
        assert_eq!(codes.len(), 2);
        assert_eq!(codes[0].len(), 2);
        let dec = rvq.decode(&codes, 2, 4);
        assert_eq!(dec.len(), 8);
    }

    #[test]
    fn snac24k_bitrate() {
        let rvq = snac24k();
        // (12+23+47) * log2(4096) = 82 * 12 = 984 bps
        let bps = rvq.bitrate(&[12.0, 23.0, 47.0]);
        assert!((bps - 984.0).abs() < 1e-3);
    }
}
