//! Tes nyata F6.2 untuk front-end log-mel (`mel.rs`) dan encoder ucapan
//! (`speech.rs`): FFT radix-2, filterbank, frame `durasi × sr / hop`, prefix
//! `frame / stride` (Whisper 16 kHz hop 160 stride 2 → 50 posisi/detik,
//! 10 s = 500 token).
//!
//! Dijalankan job CI `vir-modal`. Fungsi murni (`Vec<f32>`) + FFT numerik.

use vir_common::VirError;
use vir_modal::asset::ModalAsset;
use vir_modal::encoder::{EncoderSpec, ModalEncoder};
use vir_modal::kind::ModalKind;
use vir_modal::mel::{MelSpec, fft, hann, log_mel};
use vir_modal::speech::AudioEncoder;

fn is_err<T>(r: Result<T, VirError>) -> bool {
    r.is_err()
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-3
}

// ---------------------------------------------------------------------------
// mel.rs
// ---------------------------------------------------------------------------

#[test]
fn hann_window_values() {
    assert_eq!(hann(1), vec![1.0]);
    assert_eq!(hann(0), Vec::<f32>::new());
    let w = hann(4);
    assert_eq!(w.len(), 4);
    assert!(approx(w[0], 0.0));
    assert!(approx(w[1], 0.75));
    assert!(approx(w[2], 0.75));
    assert!(approx(w[3], 0.0));
    assert!(hann(8).iter().all(|v| *v >= -1e-6));
}

#[test]
fn fft_impulse_is_flat_and_constant_is_dc() {
    let mut re = vec![1.0f32, 0.0, 0.0, 0.0];
    let mut im = vec![0.0f32; 4];
    fft(&mut re, &mut im).expect("fft");
    for (r, i) in re.iter().zip(im.iter()) {
        assert!(approx(*r, 1.0), "re {r}");
        assert!(approx(*i, 0.0), "im {i}");
    }

    let mut re = vec![1.0f32; 4];
    let mut im = vec![0.0f32; 4];
    fft(&mut re, &mut im).expect("fft");
    assert!(approx(re[0], 4.0));
    for v in re[1..].iter().chain(im.iter()) {
        assert!(v.abs() < 1e-4, "bin bukan nol: {v}");
    }

    assert!(is_err(fft(&mut [0.0; 3], &mut [0.0; 3])));
    assert!(is_err(fft(&mut [0.0; 4], &mut [0.0; 5])));
    assert!(is_err(fft(&mut [0.0; 1], &mut [0.0; 1])));
}

#[test]
fn mel_spec_parameters_and_filters() {
    let spec = MelSpec::whisper();
    assert_eq!(
        (spec.sample_rate, spec.n_mels, spec.n_fft, spec.hop),
        (16_000, 80, 400, 160)
    );
    assert_eq!(spec.fft_size(), 512);
    assert!(spec.validate().is_ok());
    assert_eq!(spec.frames(16_000), 100);
    assert_eq!(spec.frames(1), 1);
    assert_eq!(spec.frames(0), 0);

    let derived = MelSpec::for_encoder(16_000, 160).expect("mel");
    assert_eq!((derived.n_fft, derived.n_mels), (400, 80));
    assert_eq!(derived.fmax, 8_000.0);
    assert!(is_err(MelSpec::for_encoder(0, 160)));
    assert!(is_err(MelSpec::for_encoder(16_000, 0)));

    let filters = spec.filters().expect("filters");
    assert_eq!(filters.len(), 80);
    assert!(filters.iter().all(|row| row.len() == 257));
    assert!(filters.iter().all(|row| row.iter().all(|v| *v >= 0.0)));
    assert!(filters.iter().any(|row| row.iter().any(|v| *v > 0.0)));

    let mut bad = MelSpec::whisper();
    bad.sample_rate = 0;
    assert!(is_err(bad.validate()));
    let mut inverted = MelSpec::whisper();
    inverted.fmin = 8_000.0;
    inverted.fmax = 0.0;
    assert!(is_err(inverted.validate()));
}

#[test]
fn log_mel_shape_finite_and_deterministic() {
    let spec = MelSpec::whisper();
    let pcm = vec![0.25f32; 16_000];
    let (feats, frames) = log_mel(&pcm, &spec).expect("mel");
    assert_eq!(frames, 100);
    assert_eq!(feats.len(), frames * spec.n_mels);
    assert!(feats.iter().all(|v| v.is_finite()));
    assert!(feats.iter().all(|v| *v > -30.0));

    let (again, again_frames) = log_mel(&pcm, &spec).expect("mel");
    assert_eq!(again_frames, frames);
    assert_eq!(feats, again);

    // Sinyal kosong ditolak; frame pertama selalu ada untuk input pendek.
    assert!(is_err(log_mel(&[], &spec)));
    let (short, short_frames) = log_mel(&[0.5f32; 10], &spec).expect("mel");
    assert_eq!(short_frames, 1);
    assert_eq!(short.len(), spec.n_mels);
}

// ---------------------------------------------------------------------------
// speech.rs
// ---------------------------------------------------------------------------

#[test]
fn speech_token_budget_matches_prd() {
    let enc = AudioEncoder::new(EncoderSpec::whisper(), 8).expect("encoder");
    assert_eq!(enc.spec().kind, ModalKind::Audio);
    assert_eq!(enc.frames_per_sec(), 50.0);
    assert_eq!(enc.prefix_tokens(10_000).ok(), Some(500));
    assert_eq!(enc.prefix_tokens(2_000).ok(), Some(100));
    assert_eq!(enc.mel_frames(10_000).ok(), Some(1_000));
    assert!(is_err(enc.mel_frames(0)));
    assert!(is_err(enc.prefix_tokens(0)));

    let asset = ModalAsset::audio("a.wav", 10_000, 16_000, 1);
    assert_eq!(enc.token_budget(&asset).ok(), Some(500));
    assert_eq!(enc.raw_len(&asset).ok(), Some(160_000));
    assert!(enc.is_frozen());
    assert!(is_err(
        enc.raw_len(&ModalAsset::audio("b.wav", 10_000, 8_000, 1))
    ));
    assert!(is_err(enc.token_budget(&ModalAsset::image("i.png", 8, 8))));
}

#[test]
fn speech_featurize_and_prefix() {
    let enc = AudioEncoder::randomized(11, EncoderSpec::whisper(), 4).expect("encoder");
    let pcm = vec![0.1f32; 16_000];
    let (feats, frames) = enc.featurize(&pcm).expect("featurize");
    assert_eq!((frames, feats.len()), (100, 100 * 80));

    let prefix = enc.encode_audio(&pcm).expect("encode");
    assert_eq!(prefix.tokens, 50);
    assert_eq!(prefix.d_model, 4);
    assert_eq!(prefix.rows.len(), 50 * 4);
    assert!(prefix.rows.iter().all(|v| v.is_finite()));
    assert!(is_err(enc.encode_audio(&[])));

    // 1 detik = 50 token; jumlah baris = token × d_model.
    let asset = ModalAsset::audio("a.wav", 1_000, 16_000, 1);
    let one_second = vec![0.0f32; 16_000];
    let prefix = enc.encode(&asset, &one_second).expect("encode");
    assert_eq!(prefix.tokens, 50);
    assert_eq!(prefix.rows.len(), 50 * 4);
    assert!(is_err(enc.encode(&asset, &vec![0.0; 15_999])));
}

#[test]
fn speech_encoder_is_deterministic_with_seed() {
    let pcm = vec![0.2f32; 3_200];
    let a = AudioEncoder::randomized(5, EncoderSpec::whisper(), 4).expect("encoder");
    let b = AudioEncoder::randomized(5, EncoderSpec::whisper(), 4).expect("encoder");
    let c = AudioEncoder::randomized(6, EncoderSpec::whisper(), 4).expect("encoder");
    assert_eq!(
        a.encode_audio(&pcm).expect("a").rows,
        b.encode_audio(&pcm).expect("b").rows
    );
    assert_ne!(
        a.encode_audio(&pcm).expect("a").rows,
        c.encode_audio(&pcm).expect("c").rows
    );
}

#[test]
fn speech_encoder_rejects_bad_specs() {
    assert!(is_err(AudioEncoder::new(EncoderSpec::siglip(16), 8)));
    assert!(is_err(AudioEncoder::new(
        EncoderSpec::speech("w", "r", 160, 0, 16_000),
        8
    )));
    assert!(is_err(AudioEncoder::new(
        EncoderSpec::speech("w", "r", 160, 2, 0),
        8
    )));
    let enc = AudioEncoder::new(EncoderSpec::speech("w", "r", 160, 2, 16_000), 8).expect("encoder");
    assert_eq!(enc.spec().describe(), "w/r patch=160 stride=2 frozen=true");
    assert!(approx(enc.mel().n_fft as f32, 400.0));
}
