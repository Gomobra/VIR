//! Konektor encoder beku → prefix token (F6.1/F6.2).

use vir_common::VirError;

/// MLP konektor sederhana (plain f32, row-major). `d_in → h → d_out`.
#[derive(Debug, Clone)]
pub struct Connector {
    pub w1: Vec<f32>, // [h, d_in]
    pub b1: Vec<f32>, // [h]
    pub w2: Vec<f32>, // [d_out, h]
    pub b2: Vec<f32>, // [d_out]
    pub d_in: usize,
    pub h: usize,
    pub d_out: usize,
}

impl Connector {
    pub fn new(d_in: usize, h: usize, d_out: usize) -> Self {
        Self {
            w1: vec![0.0; h * d_in],
            b1: vec![0.0; h],
            w2: vec![0.0; d_out * h],
            b2: vec![0.0; d_out],
            d_in,
            h,
            d_out,
        }
    }

    /// Forward: x [n, d_in] → [n, d_out] (GELU).
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, VirError> {
        let n = x.len() / self.d_in;
        if n * self.d_in != x.len() {
            return Err(VirError::Config("panjang x tidak kelipatan d_in".into()));
        }
        let mut h = vec![0.0f32; n * self.h];
        for i in 0..n {
            for j in 0..self.h {
                let mut acc = self.b1[j];
                for k in 0..self.d_in {
                    acc += self.w1[j * self.d_in + k] * x[i * self.d_in + k];
                }
                h[i * self.h + j] = gelu(acc);
            }
        }
        let mut out = vec![0.0f32; n * self.d_out];
        for i in 0..n {
            for j in 0..self.d_out {
                let mut acc = self.b2[j];
                for k in 0..self.h {
                    acc += self.w2[j * self.h + k] * h[i * self.h + k];
                }
                out[i * self.d_out + j] = acc;
            }
        }
        Ok(out)
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978846 * (x + 0.044715 * x * x * x)).tanh())
}

/// Trait tokenizer modalitas (gambar/audio) → kode diskret.
pub trait ModalTokenizer {
    fn tokenize(&self, data: &[f32]) -> Result<Vec<u32>, VirError>;
    fn decode(&self, codes: &[u32]) -> Result<Vec<f32>, VirError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connector_forward_shape() {
        let c = Connector::new(4, 8, 4);
        let x = vec![0.1f32; 8];
        let out = c.forward(&x).unwrap();
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn connector_rejects_bad_len() {
        let c = Connector::new(4, 8, 4);
        assert!(c.forward(&[0.1, 0.2, 0.3]).is_err());
    }

    #[test]
    fn gelu_zero() {
        assert!(gelu(0.0).abs() < 1e-6);
    }
}
