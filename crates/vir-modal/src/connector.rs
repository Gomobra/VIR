//! Lapisan linear dan konektor encoder beku → prefix token (F6.1/F6.2).
//!
//! Jalur fitur (ADR-009 §1): encoder pretrained **beku** menghasilkan satu
//! vektor per posisi (patch gambar / frame audio), lalu **konektor terlatih**
//! memetakannya ke `d_model` baris prefix. Konektor adalah MLP dua lapis
//! `d_in → h → d_out` dengan GELU; hanya lapisan ini yang dilatih, bobot
//! encoder tidak pernah disentuh gradien.
//!
//! Semua matriks disimpan row-major `f32` (tanpa tensor, tanpa Burn) supaya
//! crate ini tetap ringan di Termux; implementor tensor boleh memakai
//! [`Connector`] sebagai spesifikasi bobot yang sama.

use vir_common::VirError;

/// Lapisan linear `d_in → d_out`; bobot `[d_out, d_in]` row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl Linear {
    /// Bobot nol (tempat menempel checkpoint lewat `safetensors`).
    pub fn zeros(in_dim: usize, out_dim: usize) -> Result<Self, VirError> {
        if in_dim == 0 || out_dim == 0 {
            return Err(VirError::Config("dimensi lapisan linear harus > 0".into()));
        }
        Ok(Self {
            w: vec![0.0; in_dim * out_dim],
            b: vec![0.0; out_dim],
            in_dim,
            out_dim,
        })
    }

    /// Inisialisasi deterministik (uniform ±√(6/fan_in)) — hanya untuk tes dan
    /// inisialisasi awal sebelum checkpoint dimuat.
    pub fn randomized(seed: u64, in_dim: usize, out_dim: usize) -> Result<Self, VirError> {
        let mut layer = Self::zeros(in_dim, out_dim)?;
        let limit = (6.0 / in_dim as f32).sqrt();
        let mut state = seed;
        for slot in layer.w.iter_mut() {
            let (unit, next) = splitmix_unit(state);
            state = next;
            *slot = (unit * 2.0 - 1.0) * limit;
        }
        Ok(layer)
    }

    /// `x [n, in_dim] → [n, out_dim]`.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, VirError> {
        if x.len() % self.in_dim != 0 {
            return Err(VirError::Config(format!(
                "panjang x {} bukan kelipatan d_in {}",
                x.len(),
                self.in_dim
            )));
        }
        let n = x.len() / self.in_dim;
        let mut out = vec![0.0f32; n * self.out_dim];
        for (i, row) in out.chunks_exact_mut(self.out_dim).enumerate() {
            let src = &x[i * self.in_dim..(i + 1) * self.in_dim];
            for (j, slot) in row.iter_mut().enumerate() {
                let w = &self.w[j * self.in_dim..(j + 1) * self.in_dim];
                *slot = self.b[j] + dot(w, src);
            }
        }
        Ok(out)
    }
}

/// MLP konektor `d_in → h → d_out` (GELU di antara).
#[derive(Debug, Clone, PartialEq)]
pub struct Connector {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl Connector {
    /// Bobot nol; `d_in`/`h`/`d_out` harus > 0.
    pub fn new(d_in: usize, h: usize, d_out: usize) -> Result<Self, VirError> {
        Ok(Self {
            fc1: Linear::zeros(d_in, h)?,
            fc2: Linear::zeros(h, d_out)?,
        })
    }

    /// Bobot acak deterministik (kandidat awal sebelum latih konektor).
    pub fn randomized(seed: u64, d_in: usize, h: usize, d_out: usize) -> Result<Self, VirError> {
        Ok(Self {
            fc1: Linear::randomized(seed, d_in, h)?,
            fc2: Linear::randomized(seed ^ 0x9E37_79B9_7F4A_7C15, h, d_out)?,
        })
    }

    pub const fn d_in(&self) -> usize {
        self.fc1.in_dim
    }

    pub const fn hidden(&self) -> usize {
        self.fc1.out_dim
    }

    pub const fn d_out(&self) -> usize {
        self.fc2.out_dim
    }

    /// Forward: `x [n, d_in] → [n, d_out]`.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, VirError> {
        let hidden = self.fc1.forward(x)?;
        let activated: Vec<f32> = hidden.iter().map(|v| gelu(*v)).collect();
        self.fc2.forward(&activated)
    }

    /// Terapkan konektor lalu pecah menjadi baris `[tokens, d_out]`.
    pub fn forward_rows(&self, x: &[f32]) -> Result<(Vec<f32>, usize), VirError> {
        let out = self.forward(x)?;
        let tokens = if self.d_out() == 0 {
            0
        } else {
            out.len() / self.d_out()
        };
        Ok((out, tokens))
    }
}

/// GELU (aproksimasi tanh, sama dengan `vir-nn`).
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// splitmix64 → `(unit ∈ [0,1), state berikutnya)`; tanpa dependensi RNG.
fn splitmix_unit(mut state: u64) -> (f32, u64) {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let unit = (z >> 40) as f32 / (1u32 << 24) as f32;
    (unit, state)
}

/// Trait tokenizer modalitas (gambar/audio) → kode diskret.
pub trait ModalTokenizer {
    fn tokenize(&self, data: &[f32]) -> Result<Vec<u32>, VirError>;
    fn decode(&self, codes: &[u32]) -> Result<Vec<f32>, VirError>;
}

