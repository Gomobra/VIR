//! Encoder gambar beku (F6.1): patchify → konektor → prefix `N` token.
//!
//! `N = (H/p)·(W/p)` (PRD F6.1 langkah 2): 224/16 → 196, 384/16 → 576 token
//! prefix. Encoder visi penuh (keluarga SigLIP) mengimplementasikan
//! [`crate::encoder::ModalEncoder`] dengan bobot `safetensors`; baseline di
//! sini adalah patch linear + konektor, dan anggaran tokennya identik.

use vir_common::VirError;

use crate::asset::ModalAsset;
use crate::connector::Connector;
use crate::encoder::{EncoderSpec, ModalEncoder, Prefix};
use crate::kind::ModalKind;

/// Encoder gambar beku (F6.1): patchify `p×p×c` → konektor → prefix `N` token.
///
/// `N = (H/p)·(W/p)` (PRD F6.1 langkah 2): 224/16 → 196, 384/16 → 576.
#[derive(Debug, Clone)]
pub struct ImageEncoder {
    spec: EncoderSpec,
    connector: Connector,
    patch: usize,
    channels: usize,
}

impl ImageEncoder {
    /// Encoder dengan konektor nol (bobot dimuat dari checkpoint).
    pub fn new(spec: EncoderSpec, channels: usize, d_model: usize) -> Result<Self, VirError> {
        if spec.kind != ModalKind::Image {
            return Err(VirError::Config(format!(
                "spec {} bukan encoder gambar",
                spec.kind
            )));
        }
        spec.validate()?;
        if channels == 0 {
            return Err(VirError::Config("jumlah kanal gambar 0".into()));
        }
        let patch = spec.patch as usize;
        let d_in = patch
            .checked_mul(patch)
            .and_then(|v| v.checked_mul(channels))
            .ok_or(VirError::Config("dimensi patch terlalu besar".into()))?;
        Ok(Self {
            connector: Connector::new(d_in, d_model, d_model)?,
            spec,
            patch,
            channels,
        })
    }

    /// Encoder dengan konektor acak deterministik (uji/awal sebelum latih).
    pub fn randomized(
        seed: u64,
        spec: EncoderSpec,
        channels: usize,
        d_model: usize,
    ) -> Result<Self, VirError> {
        let encoder = Self::new(spec, channels, d_model)?;
        let d_in = encoder.connector.d_in();
        Ok(Self {
            connector: Connector::randomized(seed, d_in, d_model, d_model)?,
            ..encoder
        })
    }

    /// Ganti konektor (mis. bobot terlatih), dengan pemeriksaan dimensi.
    pub fn with_connector(mut self, connector: Connector) -> Result<Self, VirError> {
        let want = self.patch * self.patch * self.channels;
        if connector.d_in() != want {
            return Err(VirError::Config(format!(
                "konektor d_in {} != volume patch {want}",
                connector.d_in()
            )));
        }
        self.connector = connector;
        Ok(self)
    }

    pub const fn connector(&self) -> &Connector {
        &self.connector
    }

    pub const fn patch(&self) -> usize {
        self.patch
    }

    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Grid patch `(gh, gw)` untuk `height × width`; menolak dimensi tak habis
    /// dibagi patch (PRD F6.1: 224/16 → 14×14).
    pub fn patch_grid(&self, height: usize, width: usize) -> Result<(usize, usize), VirError> {
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

    /// `N = (H/p)·(W/p)` token prefix untuk satu gambar.
    pub fn patch_tokens(&self, height: usize, width: usize) -> Result<usize, VirError> {
        let (gh, gw) = self.patch_grid(height, width)?;
        Ok(gh * gw)
    }

    /// Panjang piksel HWC yang diharapkan untuk dimensi ini.
    pub fn raw_len_of(&self, height: usize, width: usize) -> Result<usize, VirError> {
        self.patch_grid(height, width)?;
        height
            .checked_mul(width)
            .and_then(|v| v.checked_mul(self.channels))
            .ok_or(VirError::Config("dimensi piksel terlalu besar".into()))
    }

    /// Patchify HWC → `[N, p·p·c]` (blok `p×p` berurutan baris).
    pub fn patchify(
        &self,
        pixels: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, VirError> {
        let (gh, gw) = self.patch_grid(height, width)?;
        let want = height
            .checked_mul(width)
            .and_then(|v| v.checked_mul(self.channels))
            .ok_or(VirError::Config("dimensi piksel terlalu besar".into()))?;
        if pixels.len() != want {
            return Err(VirError::Config(format!(
                "piksel {} != {height}x{width}x{}",
                pixels.len(),
                self.channels
            )));
        }
        let plane = height * width;
        let volume = self.patch * self.patch * self.channels;
        let mut out = vec![0.0f32; gh * gw * volume];
        for gy in 0..gh {
            for gx in 0..gw {
                let dst = (gy * gw + gx) * volume;
                for py in 0..self.patch {
                    for px in 0..self.patch {
                        for c in 0..self.channels {
                            let y = gy * self.patch + py;
                            let x = gx * self.patch + px;
                            let src = c * plane + y * width + x;
                            let slot = dst + (py * self.patch + px) * self.channels + c;
                            out[slot] = pixels[src];
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Prefix gambar: piksel HWC → `N = (H/p)(W/p)` baris `d_model`.
    pub fn encode_image(
        &self,
        pixels: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Prefix, VirError> {
        let patches = self.patchify(pixels, height, width)?;
        let (rows, tokens) = self.connector.forward_rows(&patches)?;
        Ok(Prefix::new(rows, tokens, self.connector.d_out()))
    }
}

impl ModalEncoder for ImageEncoder {
    type Output = Prefix;

    fn spec(&self) -> &EncoderSpec {
        &self.spec
    }

    fn token_budget(&self, asset: &ModalAsset) -> Result<usize, VirError> {
        if asset.kind != ModalKind::Image {
            return Err(VirError::Config(format!(
                "encoder gambar menolak aset {}",
                asset.kind
            )));
        }
        asset.validate()?;
        self.patch_tokens(asset.height as usize, asset.width as usize)
    }

    fn raw_len(&self, asset: &ModalAsset) -> Result<usize, VirError> {
        if asset.kind != ModalKind::Image {
            return Err(VirError::Config(format!(
                "encoder gambar menolak aset {}",
                asset.kind
            )));
        }
        asset.validate()?;
        asset
            .height
            .checked_mul(asset.width)
            .and_then(|v| v.checked_mul(self.channels as u32))
            .map(|v| v as usize)
            .ok_or(VirError::Config("dimensi piksel terlalu besar".into()))
    }

    fn encode(&self, asset: &ModalAsset, data: &[f32]) -> Result<Self::Output, VirError> {
        if data.len() != self.raw_len(asset)? {
            return Err(VirError::Config(format!(
                "piksel {} != yang diharapkan {}",
                data.len(),
                self.raw_len(asset)?
            )));
        }
        self.encode_image(data, asset.height as usize, asset.width as usize)
    }
}

