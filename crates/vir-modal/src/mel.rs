//! Front-end log-mel (Whisper-style) untuk encoder ucapan F6.2.
//!
//! Rantai: PCM mono → jendela Hann `n_fft` (zero-pad) → FFT radix-2 → daya →
//! filterbank mel → `log`. Ini front-end yang sama dengan Whisper (hop 160,
//! `n_fft` 400 di-*zero-pad* ke 512, 80 mels pada 16 kHz) tanpa dependensi
//! FFT eksternal.
//!
//! Keluaran `[frames, n_mels]` menjadi masukan konektor audio; jumlah frame
//! adalah `frames = durasi × sample_rate / hop`, dan prefix memakai
//! `frames / stride` (ADR-009 §3).

use vir_common::VirError;

/// Lantai nilai sebelum log (hindari `log(0)`).
pub const MEL_FLOOR: f32 = 1e-10;

/// Parameter front-end mel.
#[derive(Debug, Clone, PartialEq)]
pub struct MelSpec {
    pub sample_rate: u32,
    pub n_mels: usize,
    /// Ukuran jendela dalam sample (Whisper: 400).
    pub n_fft: usize,
    /// Lompatan antar jendela dalam sample (Whisper: 160 → 100 frame/s).
    pub hop: usize,
    pub fmin: f32,
    pub fmax: f32,
}

impl MelSpec {
    /// Front-end Whisper 16 kHz: 80 mels, `n_fft` 400, hop 160.
    pub fn whisper() -> Self {
        Self {
            sample_rate: 16_000,
            n_mels: 80,
            n_fft: 400,
            hop: 160,
            fmin: 0.0,
            fmax: 8_000.0,
        }
    }

    /// Turunkan dari encoder: `n_fft = 5/2 · hop` (400 pada hop 160), 80 mels.
    pub fn for_encoder(sample_rate: u32, hop: usize) -> Result<Self, VirError> {
        if sample_rate == 0 || hop == 0 {
            return Err(VirError::Config("sample rate/hop mel 0".into()));
        }
        Ok(Self {
            sample_rate,
            n_mels: 80,
            n_fft: hop * 5 / 2,
            hop,
            fmin: 0.0,
            fmax: sample_rate as f32 / 2.0,
        })
    }

    pub fn validate(&self) -> Result<(), VirError> {
        if self.sample_rate == 0 {
            return Err(VirError::Config("sample rate mel 0".into()));
        }
        if self.n_mels == 0 || self.hop == 0 || self.n_fft == 0 {
            return Err(VirError::Config("parameter mel 0".into()));
        }
        if self.n_fft < self.hop {
            return Err(VirError::Config("n_fft < hop".into()));
        }
        if self.fmax <= self.fmin {
            return Err(VirError::Config("fmax <= fmin".into()));
        }
        Ok(())
    }

    /// Ukuran FFT setelah pembulatan ke pangkat dua (Whisper: 400 → 512).
    pub fn fft_size(&self) -> usize {
        self.n_fft.next_power_of_two()
    }

    /// Jumlah jendela untuk `samples` sample PCM (jendela terakhir di-nol-kan).
    pub fn frames(&self, samples: usize) -> usize {
        if samples == 0 {
            return 0;
        }
        samples.div_ceil(self.hop).max(1)
    }

    /// Filterbank mel `[n_mels, n_fft/2 + 1]` (di-pad ke ukuran FFT).
    pub fn filters(&self) -> Result<Vec<Vec<f32>>, VirError> {
        self.validate()?;
        let bins = self.n_fft / 2 + 1;
        let width = self.fft_size() / 2 + 1;
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10f32.powf(mel / 2595.0) - 1.0);
        let mel_min = hz_to_mel(self.fmin.max(0.0));
        let mel_max = hz_to_mel(self.fmax.min(self.sample_rate as f32 / 2.0));
        let points: Vec<f32> = (0..=self.n_mels + 1)
            .map(|i| {
                let mel = mel_min + (mel_max - mel_min) * i as f32 / (self.n_mels + 1) as f32;
                mel_to_hz(mel)
            })
            .collect();
        let bin_hz = self.sample_rate as f32 / self.n_fft as f32;
        let mut out = Vec::with_capacity(self.n_mels);
        for m in 0..self.n_mels {
            let (left, center, right) = (points[m], points[m + 1], points[m + 2]);
            let mut row = vec![0.0f32; width];
            for (bin, slot) in row.iter_mut().enumerate().take(bins) {
                let hz = bin as f32 * bin_hz;
                let up = (hz - left) / (center - left).max(1e-6);
                let down = (right - hz) / (right - center).max(1e-6);
                *slot = up.min(down).max(0.0);
            }
            let norm = 2.0 / (right - left).max(1e-6);
            for slot in row.iter_mut() {
                *slot *= norm;
            }
            out.push(row);
        }
        Ok(out)
    }
}

/// Log-mel `[frames, n_mels]` dari PCM mono; frame pertama selalu ada.
pub fn log_mel(pcm: &[f32], spec: &MelSpec) -> Result<(Vec<f32>, usize), VirError> {
    spec.validate()?;
    if pcm.is_empty() {
        return Err(VirError::Config("PCM kosong".into()));
    }
    let filters = spec.filters()?;
    let window = hann(spec.n_fft);
    let size = spec.fft_size();
    let frames = spec.frames(pcm.len());
    let mut out = vec![0.0f32; frames * spec.n_mels];
    let mut re = vec![0.0f32; size];
    let mut im = vec![0.0f32; size];
    for (frame, row) in out.chunks_exact_mut(spec.n_mels).enumerate() {
        let start = frame * spec.hop;
        for slot in re.iter_mut().take(size) {
            *slot = 0.0;
        }
        for slot in im.iter_mut() {
            *slot = 0.0;
        }
        for (t, slot) in re.iter_mut().take(spec.n_fft).enumerate() {
            let sample = pcm.get(start + t).copied().unwrap_or(0.0);
            *slot = sample * window[t];
        }
        fft(&mut re, &mut im)?;
        for (band, weights) in row.iter_mut().zip(filters.iter()) {
            let mut power = 0.0f32;
            for (bin, weight) in weights.iter().enumerate() {
                if *weight == 0.0 {
                    continue;
                }
                power += weight * (re[bin] * re[bin] + im[bin] * im[bin]);
            }
            *band = (power + MEL_FLOOR).ln();
        }
    }
    Ok((out, frames))
}

/// Jendela Hann simetris sepanjang `n`.
pub fn hann(n: usize) -> Vec<f32> {
    if n < 2 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos())
        .collect()
}

/// FFT radix-2 di tempat (panjang harus pangkat dua).
pub fn fft(re: &mut [f32], im: &mut [f32]) -> Result<(), VirError> {
    let n = check_fft_len(re.len(), im.len())?;
    bit_reverse(re, im);
    let mut len = 2;
    while len <= n {
        butterflies(re, im, len);
        len *= 2;
    }
    Ok(())
}

fn check_fft_len(re_len: usize, im_len: usize) -> Result<usize, VirError> {
    if re_len != im_len || re_len < 2 || !re_len.is_power_of_two() {
        return Err(VirError::Config(
            "FFT butuh panjang pangkat dua >= 2".into(),
        ));
    }
    Ok(re_len)
}

fn bit_reverse(re: &mut [f32], im: &mut [f32]) {
    let bits = re.len().trailing_zeros();
    for i in 0..re.len() {
        let j = ((i as u64).reverse_bits() >> (64 - bits)) as usize;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
}

fn butterflies(re: &mut [f32], im: &mut [f32], len: usize) {
    let angle = -2.0 * std::f32::consts::PI / len as f32;
    let (wr, wi) = (angle.cos(), angle.sin());
    for start in (0..re.len()).step_by(len) {
        let (mut cr, mut ci) = (1.0f32, 0.0f32);
        for k in 0..len / 2 {
            let a = start + k;
            let b = a + len / 2;
            let tr = re[b] * cr - im[b] * ci;
            let ti = re[b] * ci + im[b] * cr;
            re[b] = re[a] - tr;
            im[b] = im[a] - ti;
            re[a] += tr;
            im[a] += ti;
            let ncr = cr * wr - ci * wi;
            ci = cr * wi + ci * wr;
            cr = ncr;
        }
    }
}

