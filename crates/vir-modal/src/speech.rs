//! Encoder ucapan beku (F6.2): log-mel Whisper-style + konektor → prefix.
//!
//! Anggaran token (PRD F6.2 langkah 2): `frame = durasi × sample_rate / hop`,
//! lalu dibagi faktor stride encoder. Whisper-style (hop 160, stride 2,
//! 16 kHz) → 50 posisi/detik, jadi 10 detik = **500** token prefix
//! (ADR-009 §3). Encodernya beku; hanya konektor yang dilatih.

use vir_common::VirError;

use crate::asset::ModalAsset;
use crate::budget;
use crate::connector::Connector;
use crate::encoder::{EncoderSpec, ModalEncoder, Prefix};
use crate::kind::ModalKind;
use crate::mel::{self, MelSpec};

/// Encoder ucapan beku: PCM → log-mel → (avg-pool stride) → konektor.
#[derive(Debug, Clone)]
pub struct AudioEncoder {
    spec: EncoderSpec,
    mel: MelSpec,
    connector: Connector,
}

impl AudioEncoder {
    /// Encoder dengan konektor nol (bobot dimuat dari checkpoint).
    pub fn new(spec: EncoderSpec, d_model: usize) -> Result<Self, VirError> {
        if spec.kind != ModalKind::Audio {
            return Err(VirError::Config(format!(
                "spec {} bukan encoder audio",
                spec.kind
            )));
        }
        spec.validate()?;
        if spec.stride == 0 {
            return Err(VirError::Config("stride encoder audio 0".into()));
        }
        let mel = MelSpec::for_encoder(spec.sample_rate, spec.patch as usize)?;
        let connector = Connector::new(mel.n_mels, d_model, d_model)?;
        Ok(Self {
            spec,
            mel,
            connector,
        })
    }

    /// Encoder dengan konektor acak deterministik (uji/awal sebelum latih).
    pub fn randomized(seed: u64, spec: EncoderSpec, d_model: usize) -> Result<Self, VirError> {
        let encoder = Self::new(spec, d_model)?;
        let connector = Connector::randomized(seed, encoder.mel.n_mels, d_model, d_model)?;
        Ok(Self {
            connector,
            ..encoder
        })
    }

    pub const fn spec(&self) -> &EncoderSpec {
        &self.spec
    }

    pub const fn mel(&self) -> &MelSpec {
        &self.mel
    }

    pub const fn connector(&self) -> &Connector {
        &self.connector
    }

    /// Posisi prefix per detik: `sample_rate / (hop × stride)` (50 pada Whisper).
    pub fn frames_per_sec(&self) -> f32 {
        self.spec.positions_per_sec()
    }

    /// Jumlah frame log-mel untuk `duration_ms` (sebelum stride).
    pub fn mel_frames(&self, duration_ms: u32) -> Result<usize, VirError> {
        if duration_ms == 0 {
            return Err(VirError::Config("durasi audio 0".into()));
        }
        let samples = self.raw_samples(duration_ms)?;
        Ok(self.mel.frames(samples))
    }

    /// Jumlah token prefix untuk `duration_ms` (frame dibagi stride).
    pub fn prefix_tokens(&self, duration_ms: u32) -> Result<usize, VirError> {
        self.spec.frames(duration_ms, self.spec.sample_rate)
    }

    /// Log-mel `[frames, n_mels]` dari PCM mono.
    pub fn featurize(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize), VirError> {
        if pcm.is_empty() {
            return Err(VirError::Config("PCM kosong".into()));
        }
        mel::log_mel(pcm, &self.mel)
    }

    /// Turunkan sumbu waktu dengan average pooling `stride` (Whisper: 2).
    fn pool_time(&self, feats: &[f32], frames: usize) -> Result<Vec<f32>, VirError> {
        let stride = self.spec.stride as usize;
        let bands = self.mel.n_mels;
        if frames == 0 || feats.len() != frames * bands {
            return Err(VirError::Config(format!(
                "fitur {} tidak cocok {frames}×{bands}",
                feats.len()
            )));
        }
        let out_frames = frames.div_ceil(stride);
        let mut out = vec![0.0f32; out_frames * bands];
        for (i, row) in out.chunks_exact_mut(bands).enumerate() {
            let start = i * stride;
            let end = (start + stride).min(frames);
            let count = (end - start).max(1) as f32;
            for (m, slot) in row.iter_mut().enumerate() {
                let mut acc = 0.0;
                for f in start..end {
                    acc += f64::from(feats[f * bands + m]);
                }
                *slot = (acc / f64::from(count)) as f32;
            }
        }
        Ok(out)
    }

    /// Prefix audio: PCM mono → `tokens = frame / stride` baris `d_model`.
    pub fn encode_audio(&self, pcm: &[f32]) -> Result<Prefix, VirError> {
        let (feats, frames) = self.featurize(pcm)?;
        let pooled = self.pool_time(&feats, frames)?;
        let (rows, tokens) = self.connector.forward_rows(&pooled)?;
        Ok(Prefix::new(rows, tokens, self.connector.d_out()))
    }

    fn raw_samples(&self, duration_ms: u32) -> Result<usize, VirError> {
        let samples = u64::from(duration_ms) * u64::from(self.spec.sample_rate) / 1_000;
        usize::try_from(samples).map_err(|_| VirError::Config("durasi audio terlalu besar".into()))
    }
}

impl ModalEncoder for AudioEncoder {
    type Output = Prefix;

    fn spec(&self) -> &EncoderSpec {
        &self.spec
    }

    fn token_budget(&self, asset: &ModalAsset) -> Result<usize, VirError> {
        if asset.kind != ModalKind::Audio {
            return Err(VirError::Config(format!(
                "encoder audio menolak aset {}",
                asset.kind
            )));
        }
        asset.validate()?;
        budget::prefix_budget(&self.spec, asset)
    }

    fn raw_len(&self, asset: &ModalAsset) -> Result<usize, VirError> {
        if asset.kind != ModalKind::Audio {
            return Err(VirError::Config(format!(
                "encoder audio menolak aset {}",
                asset.kind
            )));
        }
        asset.validate()?;
        if asset.sample_rate != self.spec.sample_rate {
            return Err(VirError::Config(format!(
                "sample rate aset {} != encoder {} (butuh resampler)",
                asset.sample_rate, self.spec.sample_rate
            )));
        }
        self.raw_samples(asset.duration_ms)
    }

    fn encode(&self, asset: &ModalAsset, data: &[f32]) -> Result<Self::Output, VirError> {
        if data.len() != self.raw_len(asset)? {
            return Err(VirError::Config(format!(
                "PCM {} != yang diharapkan {}",
                data.len(),
                self.raw_len(asset)?
            )));
        }
        self.encode_audio(data)
    }
}

