//! Decoder gambar (F6.3): kode diskret berlevel → piksel RGB.
//!
//! Trait [`ImageDecoder`] adalah batas antara jalur kode dan tensor: crate ini
//! menyediakan satu implementasi nyata ([`LinearImageDecoder`]) yang memakai
//! [`MultiScaleTokenizer`] untuk dekuantisasi FSQ lalu satu lapisan linear
//! `d → p·p·c` sebagai kepala piksel. Decoder penuh (transposed conv, VQGAN)
//! mengimplementasikan trait yang sama dengan bobot `safetensors`.

use vir_common::VirError;

use crate::connector::Linear;
use crate::image::MultiScaleTokenizer;

/// Decoder gambar: kode per level (level 0 = kasar) → piksel.
pub trait ImageDecoder {
    /// `levels[k]` adalah kode datar level `k` (`positions_k · n` nilai).
    /// Keluaran: piksel RGB HWC row-major, rentang `[0, 1]`.
    fn decode(
        &self,
        levels: &[Vec<u32>],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, VirError>;
}

/// Decoder FSQ + kepala linear `d → p·p·c` (kebalikan encoder patch linear).
#[derive(Debug, Clone)]
pub struct LinearImageDecoder {
    pub tokenizer: MultiScaleTokenizer,
    head: Linear,
    patch: usize,
    channels: usize,
}

impl LinearImageDecoder {
    pub fn new(
        tokenizer: MultiScaleTokenizer,
        patch: usize,
        channels: usize,
    ) -> Result<Self, VirError> {
        if patch == 0 || channels == 0 {
            return Err(VirError::Config("patch/kanal decoder 0".into()));
        }
        let out_dim = patch
            .checked_mul(patch)
            .and_then(|v| v.checked_mul(channels))
            .ok_or(VirError::Config(
                "dimensi patch decoder terlalu besar".into(),
            ))?;
        let head = Linear::zeros(tokenizer.fsq.d, out_dim)?;
        Ok(Self {
            tokenizer,
            head,
            patch,
            channels,
        })
    }

    /// Kepala identitas (hanya mungkin bila `d == p·p·c`) — untuk uji.
    pub fn identity(
        tokenizer: MultiScaleTokenizer,
        patch: usize,
        channels: usize,
    ) -> Result<Self, VirError> {
        let mut decoder = Self::new(tokenizer, patch, channels)?;
        let d = decoder.tokenizer.fsq.d;
        let out_dim = decoder.head.out_dim;
        if d != out_dim {
            return Err(VirError::Config(format!(
                "kepala identitas butuh d == p²c ({d} != {out_dim})"
            )));
        }
        decoder.head = Linear {
            w: identity_weights(d),
            b: vec![0.0; out_dim],
            in_dim: d,
            out_dim,
        };
        Ok(decoder)
    }

    /// Ganti kepala piksel (bobot terlatih).
    pub fn with_head(mut self, head: Linear) -> Result<Self, VirError> {
        let want = self.tokenizer.fsq.d;
        if head.in_dim != want {
            return Err(VirError::Config(format!(
                "kepala d_in {} != dimensi laten {want}",
                head.in_dim
            )));
        }
        self.head = head;
        Ok(self)
    }

    pub const fn patch(&self) -> usize {
        self.patch
    }

    pub const fn channels(&self) -> usize {
        self.channels
    }

    pub const fn head(&self) -> &Linear {
        &self.head
    }

    /// Grid patch `(gh, gw)`; menolak dimensi yang tidak habis dibagi patch.
    pub fn grid(&self, height: usize, width: usize) -> Result<(usize, usize), VirError> {
        if height == 0 || width == 0 {
            return Err(VirError::Config("dimensi gambar 0".into()));
        }
        if height % self.patch != 0 || width % self.patch != 0 {
            return Err(VirError::Config(format!(
                "dimensi {height}x{width} tidak habis dibagi patch {}",
                self.patch
            )));
        }
        Ok((height / self.patch, width / self.patch))
    }

    /// Latent halus `[grid, d]` → raster `[height·width·channels]`.
    pub fn decode_latents(
        &self,
        latent: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, VirError> {
        let (gh, gw) = self.grid(height, width)?;
        let d = self.tokenizer.fsq.d;
        let grid = gh * gw;
        if latent.len() != grid * d {
            return Err(VirError::Config(format!(
                "latent {} != {grid}×{d}",
                latent.len()
            )));
        }
        let patches = self.head.forward(latent)?;
        let volume = self.patch * self.patch * self.channels;
        if patches.len() != grid * volume {
            return Err(VirError::Config(format!(
                "kepala menghasilkan {} != {grid}×{volume}",
                patches.len()
            )));
        }
        let mut out = vec![0.0f32; height * width * self.channels];
        for gy in 0..gh {
            for gx in 0..gw {
                let src = (gy * gw + gx) * volume;
                let patch = &patches[src..src + volume];
                self.scatter(&mut out, height, width, gy, gx, patch);
            }
        }
        Ok(out)
    }

    /// Tempel satu patch `p×p×c` ke raster pada grid `(gy, gx)` (dijepit `[0,1]`).
    fn scatter(
        &self,
        out: &mut [f32],
        height: usize,
        width: usize,
        gy: usize,
        gx: usize,
        patch: &[f32],
    ) {
        let plane = height * width;
        for py in 0..self.patch {
            for px in 0..self.patch {
                for (c, value) in patch[(py * self.patch + px) * self.channels..]
                    .iter()
                    .take(self.channels)
                    .enumerate()
                {
                    let y = (gy * self.patch + py).min(height - 1);
                    let x = (gx * self.patch + px).min(width - 1);
                    out[c * plane + y * width + x] = value.clamp(0.0, 1.0);
                }
            }
        }
    }
}

impl ImageDecoder for LinearImageDecoder {
    fn decode(
        &self,
        levels: &[Vec<u32>],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, VirError> {
        let (gh, gw) = self.grid(height, width)?;
        if gh * gw != self.tokenizer.finest() {
            return Err(VirError::Config(format!(
                "kode level halus {} != grid {}",
                self.tokenizer.finest(),
                gh * gw
            )));
        }
        let latent = self.tokenizer.decode(levels)?;
        self.decode_latents(&latent, height, width)
    }
}

fn identity_weights(n: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n * n];
    for i in 0..n {
        w[i * n + i] = 1.0;
    }
    w
}

