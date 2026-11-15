//! Codec audio residual SNAC (F6.4): konfigurasi laju, bitrate, dan kode.
//!
//! SNAC (arXiv 2410.14411) adalah RVQ multiskala: tiap tahap `i` menurunkan
//! residual dengan faktor `W_i` (average pooling area), mencari kode, lalu
//! menaikkannya kembali dengan `W_i` (nearest neighbor) — pola `Down_k`/`R_k`
//! NCM (F1.4.b). Bitrate wajib diperiksa di CI sebelum decoder dipakai
//! (ADR-009 §3):
//!
//! ```text
//! bps = Σ_k laju_k × log2|V|
//! 24 kHz, stride [4,2,1]   → 12/23/47 Hz   → 82 × 12 = 984 bps
//! 32 kHz, stride [8,4,2,1] → 10/21/42/83   → 156 × 12 = 1.872 bps
//! 44 kHz, stride [8,4,2,1] → 14/29/57/115  → 215 × 12 = 2.580 bps
//! ```

use vir_common::VirError;

use crate::audio::Rvq;
use crate::budget;
use crate::pool;

#[derive(Debug, Clone, PartialEq)]
pub struct SnacConfig {
    pub sample_rate: u32,
    /// Faktor turun kumulatif per level, kasar → halus; terakhir harus 1.
    pub strides: Vec<u32>,
    /// Hop level terhalus dalam sample (512 pada 24 kHz).
    pub hop: u32,
    pub codebook_bits: u32,
    pub codebook_size: u32,
    pub dim: usize,
}

impl SnacConfig {
    /// SNAC 24 kHz: stride `[4,2,1]`, hop 512, codebook 4096 (12 bit).
    pub fn snac24k() -> Self {
        Self {
            sample_rate: 24_000,
            strides: vec![4, 2, 1],
            hop: 512,
            codebook_bits: 12,
            codebook_size: 4_096,
            dim: 8,
        }
    }

    /// SNAC 32 kHz: stride `[8,4,2,1]`, hop 384 → 10/21/42/83 Hz.
    pub fn snac32k() -> Self {
        Self {
            sample_rate: 32_000,
            strides: vec![8, 4, 2, 1],
            hop: 384,
            ..Self::snac24k()
        }
    }

    /// SNAC 44.1 kHz: stride `[8,4,2,1]`, hop 384 → 14/29/57/115 Hz.
    pub fn snac44k() -> Self {
        Self {
            sample_rate: 44_100,
            strides: vec![8, 4, 2, 1],
            hop: 384,
            ..Self::snac24k()
        }
    }

    pub fn validate(&self) -> Result<(), VirError> {
        if self.sample_rate == 0 || self.hop == 0 {
            return Err(VirError::Config("sample rate/hop codec 0".into()));
        }
        if self.strides.is_empty() {
            return Err(VirError::Config("stride codec kosong".into()));
        }
        if self.strides.contains(&0) {
            return Err(VirError::Config("stride codec 0".into()));
        }
        if self.strides[self.strides.len() - 1] != 1 {
            return Err(VirError::Config(
                "stride terakhir (level terhalus) harus 1".into(),
            ));
        }
        if self.strides.windows(2).any(|w| w[0] <= w[1]) {
            return Err(VirError::Config(
                "stride harus menurun kasar → halus".into(),
            ));
        }
        if self.codebook_size == 0 || self.dim == 0 {
            return Err(VirError::Config("codebook/dimensi codec 0".into()));
        }
        let bits = (self.codebook_size as f64).log2();
        if (bits - f64::from(self.codebook_bits)).abs() > 1e-6 {
            return Err(VirError::Config(format!(
                "codebook_bits {} tidak cocok |V|={}",
                self.codebook_bits, self.codebook_size
            )));
        }
        Ok(())
    }

    /// Laju per level (Hz), dibulatkan seperti deklarasi PRD (12/23/47 …).
    pub fn rates(&self) -> Result<Vec<f32>, VirError> {
        self.validate()?;
        Ok(self
            .strides
            .iter()
            .map(|s| {
                let hz = f64::from(self.sample_rate) / (f64::from(self.hop) * f64::from(*s));
                hz.round().max(1.0) as f32
            })
            .collect())
    }

    /// Bitrate efektif `Σ_k laju_k × bit_per_kode`.
    pub fn bitrate_bps(&self) -> Result<f32, VirError> {
        let rates = self.rates()?;
        Ok(budget::audio_bitrate(&rates, self.codebook_bits))
    }

    /// Jumlah kode per level untuk `duration_ms` (level terhalus dulu).
    pub fn frames(&self, duration_ms: u32) -> Result<Vec<usize>, VirError> {
        self.validate()?;
        if duration_ms == 0 {
            return Err(VirError::Config("durasi audio 0".into()));
        }
        let samples = u64::from(duration_ms) * u64::from(self.sample_rate) / 1_000;
        let finest = (samples / u64::from(self.hop)).max(1) as usize;
        Ok(self
            .strides
            .iter()
            .map(|s| (finest / *s as usize).max(1))
            .collect())
    }

    /// Total kode untuk `duration_ms` (82/detik pada 24 kHz → 10 s = 820).
    pub fn total_codes(&self, duration_ms: u32) -> Result<usize, VirError> {
        let rates = self.rates()?;
        budget::audio_codes(duration_ms, &rates)
    }

    /// Pemeriksaan wajib CI: laju turunan harus sama dengan deklarasi PRD dan
    /// bitrate harus cocok (toleransi 1 bps) sebelum decoder dipakai.
    pub fn check_bitrate(&self) -> Result<f32, VirError> {
        self.validate()?;
        let rates = self.rates()?;
        let bps = budget::audio_bitrate(&rates, self.codebook_bits);
        let expected = reference_rates(self.sample_rate).ok_or(VirError::Config(format!(
            "tidak ada referensi laju untuk {} Hz",
            self.sample_rate
        )))?;
        if rates != expected {
            return Err(VirError::Config(format!(
                "laju {rates:?} != referensi {expected:?} pada {} Hz",
                self.sample_rate
            )));
        }
        let reference_bps = budget::audio_bitrate(&expected, self.codebook_bits);
        if (bps - reference_bps).abs() > 1.0 {
            return Err(VirError::Config(format!(
                "bitrate {bps} bps != referensi {reference_bps} bps"
            )));
        }
        Ok(bps)
    }
}

/// Laju referensi PRD F6.4 per sample rate.
fn reference_rates(sample_rate: u32) -> Option<Vec<f32>> {
    match sample_rate {
        24_000 => Some(vec![12.0, 23.0, 47.0]),
        32_000 => Some(vec![10.0, 21.0, 42.0, 83.0]),
        44_100 => Some(vec![14.0, 29.0, 57.0, 115.0]),
        _ => None,
    }
}

/// Codec SNAC: RVQ multiskala + pemeriksaan bitrate.
#[derive(Debug, Clone)]
pub struct SnacCodec {
    pub cfg: SnacConfig,
    pub rvq: Rvq,
}

impl SnacCodec {
    pub fn new(cfg: SnacConfig) -> Result<Self, VirError> {
        cfg.validate()?;
        let rvq = Rvq::new(cfg.codebook_size as usize, cfg.dim, cfg.strides.len());
        rvq.validate()?;
        Ok(Self { cfg, rvq })
    }

    /// SNAC 24 kHz (default F6.4).
    pub fn snac24k() -> Result<Self, VirError> {
        Self::new(SnacConfig::snac24k())
    }

    pub fn bitrate_bps(&self) -> Result<f32, VirError> {
        self.rvq.bitrate(&self.cfg.rates()?)
    }

    /// Pemeriksaan bitrate wajib (ADR-009 §3).
    pub fn check_bitrate(&self) -> Result<f32, VirError> {
        let declared = self.cfg.check_bitrate()?;
        let computed = self.bitrate_bps()?;
        if (declared - computed).abs() > 1.0 {
            return Err(VirError::Config(format!(
                "bitrate deklarasi {declared} != hitung {computed}"
            )));
        }
        Ok(computed)
    }

    pub fn frames(&self, duration_ms: u32) -> Result<Vec<usize>, VirError> {
        self.cfg.frames(duration_ms)
    }

    pub fn codes(&self, duration_ms: u32) -> Result<usize, VirError> {
        self.cfg.total_codes(duration_ms)
    }

    /// Encode latent halus `[finest, dim]` → kode per level (kasar → halus).
    pub fn encode(&self, latents: &[f32]) -> Result<Vec<Vec<u32>>, VirError> {
        self.cfg.validate()?;
        let dim = self.cfg.dim;
        let finest = self.finest_len(latents.len(), dim)?;
        let mut residual = latents.to_vec();
        let mut z_hat = vec![0.0f32; latents.len()];
        let mut out = Vec::with_capacity(self.cfg.strides.len());
        for (level, stride) in self.cfg.strides.iter().enumerate() {
            let len = (finest / *stride as usize).max(1);
            let coarse = pool::avg_pool_rows(&residual, finest, len, dim)?;
            let codes = self.rvq.encode_level(level, &coarse, len, dim)?;
            let q = self.rvq.decode_level(level, &codes, dim)?;
            let up = pool::nearest_up_rows(&q, len, finest, dim)?;
            pool::add_assign(&mut z_hat, &up)?;
            pool::sub_assign(&mut residual, &up)?;
            out.push(codes);
        }
        Ok(out)
    }

    /// Decode kode per level → latent halus `[finest, dim]` (jumlahan level).
    ///
    /// Latent ini masih domain codec; gelombang akhir dibentuk oleh *vocoder*
    /// (transposed conv) yang memakai bobot `safetensors` di sisi tensor.
    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>, VirError> {
        self.cfg.validate()?;
        let dim = self.cfg.dim;
        if codes.len() != self.cfg.strides.len() {
            return Err(VirError::Config(format!(
                "level {} != stride {}",
                codes.len(),
                self.cfg.strides.len()
            )));
        }
        let finest = self.finest_from_codes(codes)?;
        let mut out = vec![0.0f32; finest * dim];
        for (level, (level_codes, stride)) in codes.iter().zip(self.cfg.strides.iter()).enumerate()
        {
            let want = (finest / *stride as usize).max(1);
            if level_codes.len() != want {
                return Err(VirError::Config(format!(
                    "level {level} punya {} kode != {want}",
                    level_codes.len()
                )));
            }
            let q = self.rvq.decode_level(level, level_codes, dim)?;
            let up = pool::nearest_up_rows(&q, want, finest, dim)?;
            pool::add_assign(&mut out, &up)?;
        }
        Ok(out)
    }

    fn finest_len(&self, elements: usize, dim: usize) -> Result<usize, VirError> {
        if dim == 0 || elements == 0 || elements % dim != 0 {
            return Err(VirError::Config(format!(
                "panjang latent {elements} bukan kelipatan {dim}"
            )));
        }
        Ok(elements / dim)
    }

    /// Jumlah **baris** level terhalus: panjang `Vec<u32>` level terakhir
    /// adalah jumlah posisi (bukan elemen), sama dengan keluaran
    /// [`SnacCodec::encode`] dan masukan `decode_level`.
    fn finest_from_codes(&self, codes: &[Vec<u32>]) -> Result<usize, VirError> {
        let finest_level = codes.len() - 1;
        if self.cfg.strides[finest_level] != 1 {
            return Err(VirError::Config("level terhalus harus berstride 1".into()));
        }
        let count = codes[finest_level].len();
        if count == 0 {
            return Err(VirError::Config(
                "kode level terhalus kosong (tidak bisa menentukan panjang)".into(),
            ));
        }
        Ok(count)
    }
}
