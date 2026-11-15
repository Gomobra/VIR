//! Tes nyata F6.5 (video = pemahaman) untuk sampler frame (`video.rs`): laju
//! tetap, perubahan adegan, anggaran token `frames × N_frame`, dan prefix per
//! frame lewat encoder gambar F6.1.
//!
//! Dijalankan job CI `vir-modal`. Fungsi murni (`Vec<u32>`/`Prefix`).

use vir_common::VirError;
use vir_modal::asset::ModalAsset;
use vir_modal::encoder::EncoderSpec;
use vir_modal::kind::ModalKind;
use vir_modal::video::{
    FrameSelection, SampleMode, encode_frames, sample_fixed, sample_scene_change, select_frames,
    video_token_budget,
};
use vir_modal::vision::ImageEncoder;

fn is_err<T>(r: Result<T, VirError>) -> bool {
    r.is_err()
}

fn encoder() -> ImageEncoder {
    ImageEncoder::randomized(3, EncoderSpec::siglip(4), 1, 4).expect("encoder")
}

#[test]
fn fixed_sampling_times() {
    assert_eq!(sample_fixed(10, 30, 2).expect("fixed"), vec![0, 2, 4, 6, 8]);
    assert_eq!(
        sample_fixed(10, 30, 1).expect("fixed"),
        (0..10).collect::<Vec<u32>>()
    );
    assert_eq!(sample_fixed(2, 10, 1).expect("fixed"), vec![0, 1]);
    assert_eq!(sample_fixed(1, 30, 5).expect("fixed"), vec![0]);
    // Durasi 0 tetap menghasilkan satu frame (frame pertama selalu ada).
    assert_eq!(sample_fixed(0, 30, 2).expect("fixed"), vec![0]);
    assert!(is_err(sample_fixed(10, 0, 2)));
    assert!(is_err(sample_fixed(10, 30, 0)));
}

#[test]
fn scene_change_sampling() {
    assert_eq!(
        sample_scene_change(&[0.1, 0.9, 0.05, 0.95], 0.5).expect("scene"),
        vec![0, 1, 3]
    );
    assert_eq!(sample_scene_change(&[0.9], 0.5).expect("scene"), vec![0]);
    assert_eq!(
        sample_scene_change(&[1.0, 1.0, 1.0], 0.5).expect("scene"),
        vec![0, 1, 2]
    );
    assert_eq!(sample_scene_change(&[0.4], 0.5).expect("scene"), vec![0]);
    assert!(is_err(sample_scene_change(&[], 0.5)));
    assert!(is_err(sample_scene_change(&[0.5], f32::NAN)));
    assert!(is_err(sample_scene_change(&[0.5], f32::INFINITY)));
}

#[test]
fn frame_selection_token_budget() {
    let sel = FrameSelection::new(vec![0, 2, 4]);
    assert_eq!(sel.len(), 3);
    assert!(!sel.is_empty());
    assert_eq!(sel.tokens(196).ok(), Some(588));
    assert!(FrameSelection::new(vec![]).tokens(196).is_err());
    assert!(FrameSelection::new(vec![]).is_empty());
}

#[test]
fn select_frames_for_video_asset() {
    let asset = ModalAsset::video("v.mp4", 224, 224, 4_000, 25);
    assert_eq!(asset.kind, ModalKind::Video);
    assert!(asset.validate().is_ok());
    assert_eq!(asset.frames, 100);

    let fixed = select_frames(&asset, SampleMode::Fixed { stride_s: 2 }, &[]).expect("select");
    assert_eq!(fixed.times, vec![0, 2]);
    assert_eq!(fixed.tokens(196).ok(), Some(392));

    let scene = select_frames(
        &asset,
        SampleMode::SceneChange { threshold: 0.5 },
        &[0.1, 0.9, 0.2],
    )
    .expect("select");
    assert_eq!(scene.times, vec![0, 1]);

    let audio = ModalAsset::audio("a.wav", 10_000, 16_000, 1);
    assert!(is_err(select_frames(
        &audio,
        SampleMode::Fixed { stride_s: 1 },
        &[]
    )));
    assert_eq!(video_token_budget(&asset, 196, 3).ok(), Some(588));
    assert!(is_err(video_token_budget(&audio, 196, 3)));
}

#[test]
fn encode_frames_produces_one_prefix_per_frame() {
    let encoder = encoder();
    let (h, w) = (8usize, 8usize);
    let frame0 = vec![0.1f32; h * w];
    let frame1 = vec![0.9f32; h * w];
    let times = [0u32, 1];
    let frames: [&[f32]; 2] = [&frame0, &frame1];

    let out = encode_frames(&encoder, h, w, &frames, &times).expect("frames");
    assert_eq!(out.len(), 2);
    for prefix in &out {
        assert_eq!(prefix.tokens, 4);
        assert_eq!(prefix.d_model, 4);
        assert_eq!(prefix.elements(), 16);
        assert!(!prefix.is_empty());
        assert!(prefix.rows.iter().all(|v| v.is_finite()));
    }
    assert_ne!(out[0].rows, out[1].rows);

    let one: [&[f32]; 1] = [&frame0];
    let tiny = vec![0.0f32; 3];
    let tiny_frames: [&[f32]; 1] = [&tiny];
    let none: [&[f32]; 0] = [];
    assert!(is_err(encode_frames(&encoder, h, w, &none, &[])));
    assert!(is_err(encode_frames(&encoder, h, w, &one, &times)));
    assert!(is_err(encode_frames(&encoder, h, w, &tiny_frames, &[0])));
    assert!(is_err(encode_frames(&encoder, h, w, &frames, &[0])));
}

#[test]
fn video_prefix_budget_matches_encoder_grid() {
    let encoder = encoder();
    let asset = ModalAsset::video("v.mp4", 32, 32, 1_000, 25);
    assert_eq!(encoder.patch_tokens(32, 32).ok(), Some(64));
    assert_eq!(video_token_budget(&asset, 64, 25).ok(), Some(1_600));
}
