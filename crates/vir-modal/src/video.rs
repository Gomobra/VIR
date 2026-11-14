//! Video (F6.5) — **pemahaman saja** (ADR-009 §4): sampler frame → encoder
//! gambar per frame → token frame berurutan dengan penanda waktu.
//!
//! Tidak ada encoder 3D/temporal: setiap frame masuk encoder gambar F6.1
//! ([`ImageEncoder`]) dan prefix-nya disusun berurutan. PRD F6.5 memakai
//! `P = frame × N_frame`, jadi anggaran token dihitung dari jumlah frame hasil
//! sampling, bukan dari durasi video.

use vir_common::VirError;

use crate::asset::ModalAsset;
use crate::budget;
use crate::encoder::Prefix;
use crate::kind::ModalKind;
use crate::vision::ImageEncoder;

/// Cara memilih frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleMode {
    /// Satu frame tiap `stride_s` detik.
    Fixed { stride_s: u32 },
    /// Frame saat selisih rata-rata antar frame mencapai `threshold`.
    SceneChange { threshold: f32 },
}

/// Hasil sampling: detik awal tiap frame terpilih.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameSelection {
    pub times: Vec<u32>,
}

impl FrameSelection {
    pub fn new(times: Vec<u32>) -> Self {
        Self { times }
    }

    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Token prefix total: `frames × token per frame` (ADR-009 §4).
    pub fn tokens(&self, per_frame: usize) -> Result<usize, VirError> {
        budget::video_tokens(self.times.len() as u32, per_frame)
    }
}

/// Sampling laju tetap: satu frame tiap `stride_s` detik, mulai detik 0.
pub fn sample_fixed(duration_s: u32, fps: u32, stride_s: u32) -> Result<Vec<u32>, VirError> {
    if fps == 0 {
        return Err(VirError::Config("fps video 0".into()));
    }
    if stride_s == 0 {
        return Err(VirError::Config("stride sampling 0".into()));
    }
    let total = u64::from(duration_s) * u64::from(fps);
    let step = u64::from(stride_s) * u64::from(fps);
    let mut times = Vec::new();
    let mut t = 0u64;
    while t < total {
        times.push((t / u64::from(fps)) as u32);
        t += step;
    }
    if times.is_empty() {
        times.push(0);
    }
    Ok(times)
}

/// Sampling berbasis perubahan adegan: selalu ambil frame 0, lalu setiap frame
/// dengan selisih rata-rata ≥ `threshold` (`frame_diffs[i]` = selisih frame
/// `i−1` → `i`).
pub fn sample_scene_change(frame_diffs: &[f32], threshold: f32) -> Result<Vec<u32>, VirError> {
    if frame_diffs.is_empty() {
        return Err(VirError::Config("daftar selisih frame kosong".into()));
    }
    if !threshold.is_finite() {
        return Err(VirError::Config("threshold sampling tidak hingga".into()));
    }
    let mut out = vec![0u32];
    for (i, diff) in frame_diffs.iter().enumerate() {
        if i > 0 && *diff >= threshold {
            out.push(i as u32);
        }
    }
    Ok(out)
}

/// Sampling sesuai [`SampleMode`] untuk satu aset video.
pub fn select_frames(
    asset: &ModalAsset,
    mode: SampleMode,
    frame_diffs: &[f32],
) -> Result<FrameSelection, VirError> {
    if asset.kind != ModalKind::Video {
        return Err(VirError::Config(format!(
            "bukan aset video: {}",
            asset.kind
        )));
    }
    asset.validate()?;
    match mode {
        SampleMode::Fixed { stride_s } => {
            let duration_s = asset.duration_ms.div_ceil(1_000);
            let times = sample_fixed(duration_s, u32::from(asset.fps), stride_s)?;
            Ok(FrameSelection::new(times))
        }
        SampleMode::SceneChange { threshold } => {
            let times = sample_scene_change(frame_diffs, threshold)?;
            Ok(FrameSelection::new(times))
        }
    }
}

/// Anggaran token prefix video untuk `frames` frame.
pub fn video_token_budget(
    asset: &ModalAsset,
    per_frame: usize,
    frames: usize,
) -> Result<usize, VirError> {
    if asset.kind != ModalKind::Video {
        return Err(VirError::Config(format!(
            "bukan aset video: {}",
            asset.kind
        )));
    }
    budget::video_tokens(frames as u32, per_frame)
}

/// Prefix per frame: `frames[i]` adalah piksel HWC satu frame, `times[i]` detik
/// awalnya (dipakai penanda waktu `<t=…>` di sisi perakit urutan).
pub fn encode_frames(
    encoder: &ImageEncoder,
    height: usize,
    width: usize,
    frames: &[&[f32]],
    times: &[u32],
) -> Result<Vec<Prefix>, VirError> {
    if frames.is_empty() {
        return Err(VirError::Config("video tanpa frame".into()));
    }
    if frames.len() != times.len() {
        return Err(VirError::Config(format!(
            "frame {} != waktu {}",
            frames.len(),
            times.len()
        )));
    }
    let want = encoder.raw_len_of(height, width)?;
    let mut out = Vec::with_capacity(frames.len());
    for (frame, time) in frames.iter().zip(times.iter()) {
        if frame.len() != want {
            return Err(VirError::Config(format!(
                "frame t={time}s punya {} piksel != {want}",
                frame.len()
            )));
        }
        out.push(encoder.encode_image(frame, height, width)?);
    }
    Ok(out)
}

