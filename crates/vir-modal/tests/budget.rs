//! Tes nyata F6.1–F6.4 untuk anggaran token (`budget.rs`) dan spesifikasi
//! encoder (`encoder.rs`): 224/16 → 196, 384/16 → 576, audio 50 posisi/detik,
//! 10 s = 500 token, SNAC 24 kHz = 82 kode/detik dan 984 bps.
//!
//! Dijalankan job CI `vir-modal`. Semua fungsi murni (integer/float).

use vir_common::VirError;
use vir_modal::asset::ModalAsset;
use vir_modal::budget::{
    DEFAULT_PATCH, SNAC_CODEBOOK_BITS, SNAC24K_RATES, SPEECH_POSITIONS_PER_SEC, audio_bitrate,
    audio_codes, audio_understanding_tokens, image_tokens, prefix_budget, snac24k_bitrate,
    snac24k_codes, video_tokens,
};
use vir_modal::encoder::{EncoderSpec, ModalEncoder, budget_of, default_patch, spec_budget};
use vir_modal::kind::ModalKind;
use vir_modal::vision::ImageEncoder;

fn is_err<T>(r: Result<T, VirError>) -> bool {
    r.is_err()
}

#[test]
fn image_tokens_match_prd_numbers() {
    assert_eq!(DEFAULT_PATCH, 16);
    assert_eq!(default_patch(), 16);
    assert_eq!(image_tokens(224, 224, 16).ok(), Some(196));
    assert_eq!(image_tokens(384, 384, 16).ok(), Some(576));
    assert_eq!(image_tokens(32, 32, 4).ok(), Some(64));
    assert!(is_err(image_tokens(224, 224, 0)));
    assert!(is_err(image_tokens(225, 224, 16)));
    assert!(is_err(image_tokens(224, 225, 16)));
}

#[test]
fn speech_tokens_match_prd_numbers() {
    assert_eq!(SPEECH_POSITIONS_PER_SEC, 50.0);
    assert_eq!(audio_understanding_tokens(10_000, 50.0).ok(), Some(500));
    assert_eq!(audio_understanding_tokens(2_000, 50.0).ok(), Some(100));
    assert_eq!(audio_understanding_tokens(1_000, 100.0).ok(), Some(100));
    assert!(is_err(audio_understanding_tokens(0, 50.0)));
    assert!(is_err(audio_understanding_tokens(1_000, 0.0)));
    assert!(is_err(audio_understanding_tokens(1_000, -1.0)));
}

#[test]
fn audio_codes_sum_over_levels() {
    assert_eq!(audio_codes(10_000, &SNAC24K_RATES).ok(), Some(820));
    assert_eq!(audio_codes(1_000, &SNAC24K_RATES).ok(), Some(82));
    assert_eq!(audio_codes(10_000, &[12.0, 23.0, 47.0]).ok(), Some(820));
    assert!(is_err(audio_codes(0, &SNAC24K_RATES)));
    assert!(is_err(audio_codes(1_000, &[])));
    assert!(is_err(audio_codes(1_000, &[0.0])));
    assert!(is_err(audio_codes(1_000, &[-1.0])));
}

#[test]
fn snac24k_bitrate_is_984_bps() {
    assert_eq!(SNAC24K_RATES, [12.0, 23.0, 47.0]);
    assert_eq!(SNAC_CODEBOOK_BITS, 12);
    assert_eq!(snac24k_bitrate(), 984.0);
    assert_eq!(snac24k_codes(10_000).ok(), Some(820));
    // PRD F6.4: 32 kHz → 156 × 12 = 1.872 bps; 44 kHz → 215 × 12 = 2.580 bps.
    assert_eq!(audio_bitrate(&[10.0, 21.0, 42.0, 83.0], 12), 1_872.0);
    assert_eq!(audio_bitrate(&[14.0, 29.0, 57.0, 115.0], 12), 2_580.0);
    assert_eq!(audio_bitrate(&[], 12), 0.0);
}

#[test]
fn video_tokens_and_errors() {
    assert_eq!(video_tokens(2, 196).ok(), Some(392));
    assert_eq!(video_tokens(1, 576).ok(), Some(576));
    assert!(is_err(video_tokens(0, 196)));
}

#[test]
fn prefix_budget_for_vision_encoder() {
    let spec = EncoderSpec::siglip(DEFAULT_PATCH);
    assert_eq!(spec.kind, ModalKind::Image);
    assert!(spec.frozen);
    assert_eq!(
        prefix_budget(&spec, &ModalAsset::image("a.png", 224, 224)).ok(),
        Some(196)
    );
    assert_eq!(
        prefix_budget(&spec, &ModalAsset::image("b.png", 384, 384)).ok(),
        Some(576)
    );
    let coarse = EncoderSpec::vision("siglip", "r", 32);
    assert_eq!(
        prefix_budget(&coarse, &ModalAsset::image("c.png", 224, 224)).ok(),
        Some(49)
    );
    // Spec gambar menolak aset audio.
    assert!(is_err(prefix_budget(
        &spec,
        &ModalAsset::audio("a.wav", 10_000, 16_000, 1)
    )));
}

#[test]
fn prefix_budget_for_video_uses_sampled_frames() {
    let spec = EncoderSpec::siglip(DEFAULT_PATCH);
    let video = ModalAsset::video("v.mp4", 224, 224, 4_000, 25);
    assert_eq!(video.frames, 100);
    assert_eq!(prefix_budget(&spec, &video).ok(), Some(19_600));
    let short = ModalAsset::video("s.mp4", 224, 224, 1_000, 25);
    assert_eq!(short.frames, 25);
    assert_eq!(prefix_budget(&spec, &short).ok(), Some(4_900));
    assert!(is_err(prefix_budget(
        &spec,
        &ModalAsset::image("i.png", 224, 224)
    )));
}

#[test]
fn prefix_budget_for_audio_uses_hop_and_stride() {
    let spec = EncoderSpec::whisper();
    assert_eq!(spec.kind, ModalKind::Audio);
    assert_eq!(spec.positions_per_sec(), 50.0);
    let asset = ModalAsset::audio("a.wav", 10_000, 16_000, 1);
    assert_eq!(prefix_budget(&spec, &asset).ok(), Some(500));
    assert_eq!(
        prefix_budget(&spec, &ModalAsset::audio("b.wav", 1_000, 16_000, 1)).ok(),
        Some(50)
    );

    let mut hop0 = spec.clone();
    hop0.patch = 0;
    assert!(is_err(prefix_budget(&hop0, &asset)));

    let mut stride0 = spec.clone();
    stride0.stride = 0;
    assert_eq!(prefix_budget(&stride0, &asset).ok(), Some(1_000));

    assert!(is_err(prefix_budget(
        &spec,
        &ModalAsset::image("i.png", 8, 8)
    )));
}

#[test]
fn prefix_budget_rejects_text_and_video_specs() {
    let mut spec = EncoderSpec::siglip(DEFAULT_PATCH);
    let asset = ModalAsset::image("a.png", 224, 224);
    spec.kind = ModalKind::Video;
    assert!(is_err(prefix_budget(&spec, &asset)));
    spec.kind = ModalKind::Text;
    assert!(is_err(prefix_budget(&spec, &asset)));
}

#[test]
fn encoder_spec_validate_and_describe() {
    assert!(EncoderSpec::siglip(16).validate().is_ok());
    assert!(EncoderSpec::whisper().validate().is_ok());
    assert!(is_err(EncoderSpec::vision("siglip", "r", 0).validate()));
    assert!(is_err(EncoderSpec::speech("w", "r", 160, 2, 0).validate()));
    assert!(is_err(
        EncoderSpec::speech("w", "r", 0, 2, 16_000).validate()
    ));
    assert!(is_err(EncoderSpec::vision("", "r", 16).validate()));
    let mut video = EncoderSpec::siglip(16);
    video.kind = ModalKind::Video;
    assert!(is_err(video.validate()));
    assert_eq!(
        EncoderSpec::whisper().describe(),
        "whisper/large-v3-turbo patch=160 stride=2 frozen=true"
    );
    assert_eq!(
        EncoderSpec::whisper().frames(10_000, 16_000).ok(),
        Some(500)
    );
    assert_eq!(EncoderSpec::whisper().frames(2_000, 16_000).ok(), Some(100));
    assert!(is_err(EncoderSpec::whisper().frames(0, 16_000)));
    assert!(is_err(EncoderSpec::siglip(16).frames(10_000, 16_000)));
}

#[test]
fn spec_and_implementation_budget_agree() {
    let spec = EncoderSpec::siglip(DEFAULT_PATCH);
    let asset = ModalAsset::image("a.png", 224, 224);
    assert_eq!(spec_budget(&spec, &asset).ok(), Some(196));
    let encoder = ImageEncoder::new(EncoderSpec::siglip(DEFAULT_PATCH), 3, 8).expect("encoder");
    assert_eq!(budget_of(&encoder, &asset).ok(), Some(196));
    assert_eq!(encoder.token_budget(&asset).ok(), Some(196));
    assert_eq!(
        encoder.name(),
        "siglip/webli-en-patch16 patch=16 stride=1 frozen=true"
    );
}
