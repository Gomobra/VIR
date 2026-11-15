//! Tes nyata F6.0/F6.5 untuk data berpasangan (`sample.rs`: JSONL + lisensi) dan
//! perakitan urutan (`sequence.rs`: level-major, penanda waktu video).
//!
//! Dijalankan job CI `vir-modal`. Semua fungsi murni kecuali `read_jsonl`,
//! yang memakai berkas sementara di `std::env::temp_dir()`.

use vir_common::VirError;
use vir_modal::asset::ModalAsset;
use vir_modal::kind::ModalKind;
use vir_modal::sample::{PairedSample, Segment, SegmentRole, UNKNOWN_LICENSE, read_jsonl};
use vir_modal::sequence::{encode_codes, encode_video};
use vir_modal::standard_layout;
use vir_modal::token::{CodebookKey, ModalToken, Region};

fn is_err<T>(r: Result<T, VirError>) -> bool {
    r.is_err()
}

fn sample() -> PairedSample {
    let mut s = PairedSample::new("s1", "cc-by-4.0", "docs");
    s.push(Segment::text(SegmentRole::User, "what is this?"));
    s.push(Segment::asset(
        SegmentRole::Image,
        ModalAsset::image("img/1.png", 64, 64),
    ));
    s.push(Segment::text(SegmentRole::Assistant, "a cat"));
    s
}

fn time_segment(t: &str) -> Segment {
    let mut seg = Segment::text(SegmentRole::Time, "");
    seg.meta.insert("t".to_string(), t.to_string());
    seg
}

// ---------------------------------------------------------------------------
// Segment / SegmentRole
// ---------------------------------------------------------------------------

#[test]
fn segment_role_modality_and_serde() {
    assert_eq!(SegmentRole::Image.modality(), Some(ModalKind::Image));
    assert_eq!(SegmentRole::Audio.modality(), Some(ModalKind::Audio));
    assert_eq!(SegmentRole::Video.modality(), Some(ModalKind::Video));
    assert_eq!(SegmentRole::User.modality(), None);
    assert_eq!(SegmentRole::Time.modality(), None);
    assert!(SegmentRole::Image.has_asset());
    assert!(!SegmentRole::User.has_asset());
    assert!(SegmentRole::Time.is_time());
    assert!(!SegmentRole::Audio.is_time());
    assert_eq!(SegmentRole::Assistant.as_str(), "assistant");
    assert_eq!(UNKNOWN_LICENSE, "unknown");

    let json = serde_json::to_string(&SegmentRole::Video).expect("json");
    assert_eq!(json, "\"video\"");
    let back: SegmentRole = serde_json::from_str("\"time\"").expect("json");
    assert_eq!(back, SegmentRole::Time);
}

#[test]
fn segment_validate_rules() {
    assert!(Segment::text(SegmentRole::User, "hi").validate().is_ok());
    assert!(Segment::text(SegmentRole::User, "   ").validate().is_err());
    assert!(
        Segment::asset(SegmentRole::Image, ModalAsset::image("a.png", 8, 8))
            .validate()
            .is_ok()
    );
    assert!(Segment::text(SegmentRole::Image, "hi").validate().is_err());
    assert!(
        Segment::asset(SegmentRole::User, ModalAsset::image("a.png", 8, 8))
            .validate()
            .is_err()
    );
    let wrong = Segment::asset(
        SegmentRole::Image,
        ModalAsset::audio("a.wav", 10, 16_000, 1),
    );
    assert!(wrong.validate().is_err());

    let time = time_segment("1.5");
    assert_eq!(time.time_s(), Some(1.5));
    assert!(time.validate().is_ok());
    assert!(Segment::text(SegmentRole::Time, "").validate().is_err());
    assert_eq!(Segment::text(SegmentRole::Time, "").time_s(), None);
    let mut with_asset = time.clone();
    with_asset.asset = Some(ModalAsset::image("a.png", 8, 8));
    assert!(with_asset.validate().is_err());
}

#[test]
fn segment_asset_must_match_metadata() {
    let bad = ModalAsset::image("a.png", 0, 8);
    let seg = Segment::asset(SegmentRole::Image, bad);
    assert!(seg.validate().is_err());
}

// ---------------------------------------------------------------------------
// PairedSample
// ---------------------------------------------------------------------------

#[test]
fn paired_sample_validate_and_text() {
    let s = sample();
    assert!(s.validate(false).is_ok());
    assert_eq!(s.text(), "what is this? a cat");
    assert_eq!(s.count_kind(ModalKind::Image), 1);
    assert_eq!(s.count_kind(ModalKind::Audio), 0);
    assert_eq!(s.assets().count(), 1);

    assert!(
        PairedSample::new("s2", "cc-by-4.0", "d")
            .validate(false)
            .is_err()
    );
    assert!(
        PairedSample::new("", "cc-by-4.0", "d")
            .validate(false)
            .is_err()
    );

    let mut unknown = sample();
    unknown.license = UNKNOWN_LICENSE.to_string();
    assert!(unknown.validate(false).is_err());
    assert!(unknown.validate(true).is_ok());
    assert_eq!(unknown.id, "s1");
}

#[test]
fn jsonl_line_roundtrip() {
    let s = sample();
    let line = s.to_json_line().expect("line");
    assert!(!line.contains('\n'));
    assert_eq!(PairedSample::from_json_line(&line).expect("parse"), s);

    let mut with_meta = PairedSample::new("s3", "mit", "d");
    with_meta.push(time_segment("2.5"));
    let back =
        PairedSample::from_json_line(&with_meta.to_json_line().expect("line")).expect("parse");
    assert_eq!(back.segments[0].time_s(), Some(2.5));
    assert_eq!(back, with_meta);
    assert!(PairedSample::from_json_line("bukan json").is_err());
    assert!(PairedSample::from_json_line("").is_err());
}

#[test]
fn read_jsonl_loads_validates_and_skips_blanks() {
    let dir = std::env::temp_dir().join(format!("vir-modal-jsonl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join("samples.jsonl");

    let mut other = PairedSample::new("s2", "cc0", "docs");
    other.push(Segment::text(SegmentRole::User, "hello"));
    let mut text = String::new();
    for s in [sample(), other] {
        text.push_str(&s.to_json_line().expect("line"));
        text.push('\n');
    }
    text.push('\n');
    std::fs::write(&path, &text).expect("write");

    let path = path.to_str().expect("path").to_string();
    let loaded = read_jsonl(&path, false).expect("read");
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].id, "s1");
    assert_eq!(loaded[1].id, "s2");

    let mut bad = sample();
    bad.license = UNKNOWN_LICENSE.to_string();
    std::fs::write(&path, bad.to_json_line().expect("line")).expect("write");
    assert!(is_err(read_jsonl(&path, false)));
    assert!(read_jsonl(&path, true).expect("read").len() == 1);
    assert!(is_err(read_jsonl("/nonexistent/vir-modal.jsonl", true)));
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// sequence.rs
// ---------------------------------------------------------------------------

#[test]
fn encode_codes_is_level_major_with_markers() {
    let layout = standard_layout(32_000).expect("layout");
    let image = layout.marker(ModalKind::Image).expect("penanda");
    let start0 = layout
        .codebook_span(CodebookKey::new(ModalKind::Image, 0))
        .expect("codebook")
        .start;
    let start1 = layout
        .codebook_span(CodebookKey::new(ModalKind::Image, 1))
        .expect("codebook")
        .start;

    let ids = encode_codes(&layout, ModalKind::Image, &[vec![0, 1], vec![2]]).expect("encode");
    assert_eq!(
        ids,
        vec![image.open, start0, start0 + 1, start1, image.close]
    );
    assert_eq!(
        layout.region(ids[0]),
        Some(Region::Marker(ModalKind::Image))
    );
    assert_eq!(
        layout.region(ids[ids.len() - 1]),
        Some(Region::Marker(ModalKind::Image))
    );
    assert_eq!(
        layout.decode(ids[1]),
        Some(ModalToken::new(ModalKind::Image, 0, 0))
    );
    assert_eq!(
        layout.decode(ids[3]),
        Some(ModalToken::new(ModalKind::Image, 1, 2))
    );

    let audio =
        encode_codes(&layout, ModalKind::Audio, &[vec![5], vec![6], vec![7]]).expect("encode");
    assert_eq!(audio.len(), 5);
    assert!(is_err(encode_codes(&layout, ModalKind::Text, &[vec![0]])));
    assert!(is_err(encode_codes(&layout, ModalKind::Image, &[])));
    assert!(is_err(encode_codes(
        &layout,
        ModalKind::Image,
        &[vec![5_000]]
    )));
    assert!(is_err(encode_codes(
        &layout,
        ModalKind::Image,
        &[vec![0], vec![]]
    )));
}

#[test]
fn encode_video_inserts_time_markers() {
    let layout = standard_layout(32_000).expect("layout");
    let image0 = layout
        .codebook_span(CodebookKey::new(ModalKind::Image, 0))
        .expect("codebook")
        .start;
    let video = layout.marker(ModalKind::Video).expect("penanda");
    let t0 = video.time(0).expect("t0");
    let t1 = video.time(1).expect("t1");
    let t63 = video.time(63).expect("t63");

    let ids = encode_video(&layout, &[vec![0], vec![1]], &[0, 1]).expect("encode");
    assert_eq!(
        ids,
        vec![video.open, t0, image0, t1, image0 + 1, video.close]
    );
    assert_eq!(layout.region(t0), Some(Region::Marker(ModalKind::Video)));
    assert_eq!(
        layout.region(video.close),
        Some(Region::Marker(ModalKind::Video))
    );

    let late = encode_video(&layout, &[vec![0]], &[63]).expect("encode");
    assert_eq!(late, vec![video.open, t63, image0, video.close]);

    assert!(is_err(encode_video(&layout, &[], &[])));
    assert!(is_err(encode_video(&layout, &[vec![0]], &[])));
    assert!(is_err(encode_video(&layout, &[vec![0]], &[64])));
    assert!(is_err(encode_video(&layout, &[vec![4_000]], &[0])));
}
