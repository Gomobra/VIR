//! Tes nyata F6.0 untuk `vir-modal`: `ModalKind`, `Span`, `CodebookKey`,
//! `ModalToken`, dan alokasi blok `VocabLayout` (ADR-009 §5).
//!
//! Dijalankan job CI `vir-modal` (`cargo test --manifest-path
//! crates/vir-modal/Cargo.toml`). Semua fungsi di sini murni (`u32`/`Vec`),
//! jadi diuji langsung tanpa tensor.

use std::str::FromStr;

use vir_common::VirError;
use vir_modal::kind::ModalKind;
use vir_modal::token::{
    CodebookKey, DEFAULT_CODEBOOK, ModalToken, Region, SPAN_ALIGN, Span, TIME_MARKERS, VocabLayout,
    build_layout,
};

/// `true` bila galatnya galat konfigurasi (bukan panik/IO).
fn is_config<T>(r: Result<T, VirError>) -> bool {
    matches!(r, Err(VirError::Config(_)))
}

// ---------------------------------------------------------------------------
// ModalKind
// ---------------------------------------------------------------------------

#[test]
fn kind_all_is_sorted_and_marked() {
    let mut sorted = ModalKind::ALL;
    sorted.sort();
    assert_eq!(sorted, ModalKind::ALL);
    assert_eq!(ModalKind::ALL.len(), 4);
    assert_eq!(
        ModalKind::MARKED,
        [ModalKind::Image, ModalKind::Audio, ModalKind::Video]
    );
}

#[test]
fn kind_str_roundtrip_and_aliases() {
    for kind in ModalKind::ALL {
        assert_eq!(ModalKind::from_str(kind.as_str()).ok(), Some(kind));
        assert_eq!(kind.to_string(), kind.as_str());
    }
    assert_eq!(ModalKind::from_str("IMG").ok(), Some(ModalKind::Image));
    assert_eq!(ModalKind::from_str("aud").ok(), Some(ModalKind::Audio));
    assert_eq!(ModalKind::from_str("speech").ok(), Some(ModalKind::Audio));
    assert_eq!(ModalKind::from_str("vid").ok(), Some(ModalKind::Video));
    assert!(is_config(ModalKind::from_str("hologram")));
}

#[test]
fn kind_predicates_and_markers() {
    assert!(ModalKind::Image.is_generative());
    assert!(ModalKind::Audio.is_generative());
    assert!(!ModalKind::Text.is_generative());
    assert!(!ModalKind::Video.is_generative());

    assert!(ModalKind::Image.is_spatial());
    assert!(ModalKind::Video.is_spatial());
    assert!(!ModalKind::Audio.is_spatial());
    assert!(!ModalKind::Text.is_spatial());

    assert!(ModalKind::Audio.is_temporal());
    assert!(ModalKind::Video.is_temporal());
    assert!(!ModalKind::Image.is_temporal());
    assert!(!ModalKind::Text.is_temporal());

    assert_eq!(ModalKind::Image.markers(), Some(("<img>", "</img>")));
    assert_eq!(ModalKind::Audio.markers(), Some(("<aud>", "</aud>")));
    assert_eq!(ModalKind::Video.markers(), Some(("<vid>", "</vid>")));
    assert_eq!(ModalKind::Text.markers(), None);
    for kind in ModalKind::MARKED {
        let (open, close) = kind.markers().unwrap_or(("", ""));
        assert!(open.starts_with('<') && close.starts_with("</"));
        assert_eq!(open.len() + close.len(), 11, "{kind} penanda");
    }
}

#[test]
fn kind_serde_is_lowercase() {
    let json = serde_json::to_string(&ModalKind::Audio).expect("json");
    assert_eq!(json, "\"audio\"");
    let back: ModalKind = serde_json::from_str("\"video\"").expect("json");
    assert_eq!(back, ModalKind::Video);
}

// ---------------------------------------------------------------------------
// Span / CodebookKey / ModalToken
// ---------------------------------------------------------------------------

#[test]
fn span_arithmetic_and_overlap() {
    let s = Span::new(10, 4);
    assert_eq!((s.start, s.len, s.end()), (10, 4, 14));
    assert!(s.contains(10) && s.contains(13));
    assert!(!s.contains(9) && !s.contains(14));
    assert!(!s.is_empty());
    assert!(Span::new(5, 0).is_empty());
    assert_eq!(s.shift(5), Span::new(15, 4));
    assert!(Span::new(0, 4).overlaps(&Span::new(3, 4)));
    assert!(!Span::new(0, 4).overlaps(&Span::new(4, 4)));
    assert!(Span::new(4, 4).overlaps(&Span::new(0, 4)));
    assert!(!Span::new(0, 0).overlaps(&Span::new(0, 4)));
    assert_eq!(SPAN_ALIGN, 64);
}

#[test]
fn codebook_key_and_token_identity() {
    let key = CodebookKey::new(ModalKind::Audio, 2);
    assert_eq!(key.to_string(), "audio/2");
    assert!(!key.is_text());
    assert!(CodebookKey::new(ModalKind::Text, 0).is_text());
    assert_eq!(key.token(7), ModalToken::new(ModalKind::Audio, 2, 7));
    assert_eq!(key.token(7).key(), key);
    assert!(!key.token(7).is_text());
    assert_eq!(
        ModalToken::text(11),
        ModalToken::new(ModalKind::Text, 0, 11)
    );
    assert!(ModalToken::text(11).is_text());
    assert_eq!(DEFAULT_CODEBOOK, 4_096);
}

// ---------------------------------------------------------------------------
// VocabLayout
// ---------------------------------------------------------------------------

fn sample_layout() -> VocabLayout {
    build_layout(1_000, 2, 3, DEFAULT_CODEBOOK).expect("layout")
}

#[test]
fn layout_new_rejects_zero_text_vocab() {
    assert!(is_config(VocabLayout::new(0)));
    assert!(VocabLayout::new(1).is_ok());
}

#[test]
fn layout_marker_blocks_are_contiguous() {
    let layout = VocabLayout::new(1_000).expect("layout");
    assert_eq!(layout.text_vocab(), 1_000);
    // 2 gambar + 2 audio + (2 + 64) video.
    assert_eq!(layout.marker_len(), 2 + 2 + (2 + TIME_MARKERS));
    assert_eq!(layout.total(), 1_070);
    assert_eq!(layout.codebook_keys().count(), 0);
    assert_eq!(TIME_MARKERS, 64);

    let image = layout.marker(ModalKind::Image).expect("img");
    assert_eq!(
        (image.open, image.close, image.time_base),
        (1_000, 1_001, None)
    );
    assert_eq!(image.span(), Span::new(1_000, 2));
    assert_eq!(image.time(0), None);

    let audio = layout.marker(ModalKind::Audio).expect("aud");
    assert_eq!((audio.open, audio.close), (1_002, 1_003));

    let video = layout.marker(ModalKind::Video).expect("vid");
    assert_eq!(
        (video.open, video.close, video.time_base),
        (1_004, 1_005, Some(1_006))
    );
    assert_eq!(video.span(), Span::new(1_004, 2 + TIME_MARKERS));
    assert_eq!(video.time(0), Some(1_006));
    assert_eq!(video.time(63), Some(1_069));
    assert_eq!(video.time(TIME_MARKERS), None);
}

#[test]
fn layout_regions_classify_every_id() {
    let layout = VocabLayout::new(1_000).expect("layout");
    assert_eq!(layout.region(0), Some(Region::Text));
    assert_eq!(layout.region(999), Some(Region::Text));
    assert_eq!(layout.region(1_000), Some(Region::Marker(ModalKind::Image)));
    assert_eq!(layout.region(1_001), Some(Region::Marker(ModalKind::Image)));
    assert_eq!(layout.region(1_004), Some(Region::Marker(ModalKind::Video)));
    assert_eq!(layout.region(1_069), Some(Region::Marker(ModalKind::Video)));
    assert_eq!(layout.region(1_070), None);
}

#[test]
fn layout_codebooks_tile_without_gaps_or_overlap() {
    let layout = sample_layout();
    assert_eq!(layout.codebook_keys().count(), 5);

    let mut spans: Vec<Span> = Vec::new();
    for kind in ModalKind::MARKED {
        spans.push(layout.marker(kind).expect("penanda").span());
    }
    for key in layout.codebook_keys() {
        spans.push(layout.codebook_span(key).expect("codebook"));
    }
    spans.sort();
    for pair in spans.windows(2) {
        assert!(
            !pair[0].overlaps(pair[1]),
            "blok tumpang tindih: {:?} vs {:?}",
            pair[0],
            pair[1]
        );
    }

    let mut cursor = layout.text_vocab();
    for span in &spans {
        assert_eq!(span.start, cursor, "celah sebelum {span:?}");
        cursor = span.end();
    }
    assert_eq!(cursor, layout.total());
    assert_eq!(layout.total(), 1_000 + 70 + 5 * DEFAULT_CODEBOOK);

    for key in layout.codebook_keys() {
        let span = layout.codebook_span(key).expect("codebook");
        assert_eq!(span.len(), DEFAULT_CODEBOOK, "{key} ukuran");
        assert_eq!(span.len() % SPAN_ALIGN, 0, "{key} sejajar 64");
    }
}

#[test]
fn layout_encode_decode_roundtrip_per_level() {
    let layout = sample_layout();
    for (kind, levels) in [(ModalKind::Image, 2u8), (ModalKind::Audio, 3u8)] {
        for level in 0..levels {
            let key = CodebookKey::new(kind, level);
            let span = layout.codebook_span(key).expect("codebook");
            assert_eq!(layout.encode(key, 0), Some(span.start));
            assert_eq!(
                layout.encode(key, DEFAULT_CODEBOOK - 1),
                Some(span.end() - 1)
            );
            assert_eq!(layout.encode(key, DEFAULT_CODEBOOK), None);
            assert_eq!(
                layout.decode(span.start),
                Some(ModalToken::new(kind, level, 0))
            );
            assert_eq!(
                layout.decode(span.end() - 1),
                Some(ModalToken::new(kind, level, DEFAULT_CODEBOOK - 1))
            );
            assert_eq!(layout.decode(span.start).map(|t| t.key()), Some(key));
        }
    }
    // ID teks bukan kode modalitas.
    assert_eq!(layout.decode(0), None);
    // Video tidak punya codebook (hanya pemahaman).
    assert_eq!(
        layout.encode(CodebookKey::new(ModalKind::Video, 0), 0),
        None
    );
    assert_eq!(
        layout.codebook_span(CodebookKey::new(ModalKind::Video, 0)),
        None
    );
}

#[test]
fn layout_add_codebook_rules() {
    let mut layout = VocabLayout::new(64).expect("layout");
    assert!(is_config(
        layout.add_codebook(CodebookKey::new(ModalKind::Text, 0), 16)
    ));
    assert!(is_config(
        layout.add_codebook(CodebookKey::new(ModalKind::Image, 0), 0)
    ));
    layout
        .add_codebook(CodebookKey::new(ModalKind::Image, 0), 100)
        .expect("tambah");
    let start = layout.text_vocab() + layout.marker_len();
    assert_eq!(
        layout.codebook_span(CodebookKey::new(ModalKind::Image, 0)),
        Some(Span::new(start, 128))
    );
    assert!(is_config(
        layout.add_codebook(CodebookKey::new(ModalKind::Image, 0), 64)
    ));
    assert_eq!(layout.next_free(), start + 128);
}

#[test]
fn layout_build_layout_counts() {
    let layout = build_layout(32_000, 3, 3, DEFAULT_CODEBOOK).expect("layout");
    assert_eq!(layout.codebook_keys().count(), 6);
    assert_eq!(layout.marker_len(), 70);
    assert_eq!(layout.total(), 32_000 + 70 + 6 * DEFAULT_CODEBOOK);
    let none = build_layout(10, 0, 0, 64).expect("layout");
    assert_eq!(none.codebook_keys().count(), 0);
}

#[test]
fn layout_remap_after_text_resize() {
    let layout = sample_layout();
    let old_code = layout
        .encode(CodebookKey::new(ModalKind::Image, 1), 9)
        .expect("id");
    let old_time = layout
        .marker(ModalKind::Video)
        .and_then(|m| m.time(5))
        .expect("t");
    let delta = 200;

    let bigger = layout.remap_after_text_resize(1_200).expect("remap");
    assert_eq!(bigger.text_vocab(), 1_200);
    assert_eq!(bigger.marker_len(), layout.marker_len());
    assert_eq!(bigger.total(), layout.total() + delta);
    assert_eq!(
        bigger.encode(CodebookKey::new(ModalKind::Image, 1), 9),
        Some(old_code + delta)
    );
    assert_eq!(
        bigger.marker(ModalKind::Video).and_then(|m| m.time(5)),
        Some(old_time + delta)
    );
    for key in layout.codebook_keys() {
        let was = layout.codebook_span(key).expect("lama");
        let now = bigger.codebook_span(key).expect("baru");
        assert_eq!(now, was.shift(delta));
    }

    assert!(is_config(layout.remap_after_text_resize(999)));
    assert_eq!(
        layout
            .remap_after_text_resize(1_000)
            .expect("no-op")
            .total(),
        layout.total()
    );
}
