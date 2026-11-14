//! Tes `VocabLayout` (ADR-009 §5): perluasan vocab, codebook per level,
//! dan migrasi setelah resize teks.

use vir_common::VirError;
use vir_modal::{
    build_layout, CodebookKey, ModalKind, Region, Span, VocabLayout, DEFAULT_CODEBOOK, SPAN_ALIGN,
    TIME_MARKERS,
};

fn layout() -> Result<VocabLayout, VirError> {
    build_layout(1_000, 2, 3, DEFAULT_CODEBOOK)
}

#[test]
fn span_dasar() {
    let s = Span::new(10, 4);
    assert_eq!(s.end(), 14);
    assert!(s.contains(10) && s.contains(13));
    assert!(!s.contains(14) && !s.contains(9));
    assert!(s.overlaps(&Span::new(12, 1)));
    assert!(!s.overlaps(&Span::new(14, 1)));
    assert_eq!(s.shift(5), Span::new(15, 4));
    assert!(!s.is_empty());
    assert!(Span::new(0, 0).is_empty());
}

#[test]
fn codebook_tidak_bertumpuk_dengan_penanda() -> Result<(), VirError> {
    let layout = layout()?;
    let after_markers = layout.text_vocab() + layout.marker_len();
    assert_eq!(layout.codebook_keys().count(), 5);
    for key in layout.codebook_keys() {
        let span = layout
            .codebook_span(key)
            .ok_or(VirError::Internal("span hilang".into()))?;
        assert!(span.start >= after_markers, "{key} sebelum penanda");
        assert_eq!(span.len, DEFAULT_CODEBOOK);
        assert_eq!(span.len % SPAN_ALIGN, 0);
    }
    Ok(())
}

#[test]
fn encode_decode_roundtrip() -> Result<(), VirError> {
    let layout = layout()?;
    for key in layout.codebook_keys() {
        for code in [0, 1, 4_095, 4_096] {
            let id = layout.encode(key, code);
            if code < DEFAULT_CODEBOOK {
                let id = id.ok_or(VirError::Internal(format!("{key}:{code}")))?;
                assert_eq!(layout.region(id), Some(Region::Codebook(key)));
                assert_eq!(layout.decode(id), Some(key.token(code)));
            } else {
                assert!(id.is_none(), "kode {code} di luar codebook {key}");
            }
        }
    }
    Ok(())
}

#[test]
fn id_teks_dan_penanda_bukan_kode() -> Result<(), VirError> {
    let layout = layout()?;
    assert_eq!(layout.region(999), Some(Region::Text));
    assert!(layout.decode(999).is_none());
    let m = layout
        .marker(ModalKind::Image)
        .ok_or(VirError::Internal("marker gambar".into()))?;
    assert_eq!(
        layout.region(m.open),
        Some(Region::Marker(ModalKind::Image))
    );
    assert_eq!(
        layout.region(m.close),
        Some(Region::Marker(ModalKind::Image))
    );
    assert!(layout.marker(ModalKind::Text).is_none());
    Ok(())
}

#[test]
fn penanda_waktu_video_di_dalam_span() -> Result<(), VirError> {
    let layout = layout()?;
    let m = layout
        .marker(ModalKind::Video)
        .ok_or(VirError::Internal("marker video".into()))?;
    let base = m.time_base.ok_or(VirError::Internal("time base".into()))?;
    assert_eq!(m.time(0), Some(base));
    let last = m
        .time(TIME_MARKERS - 1)
        .ok_or(VirError::Internal("time akhir".into()))?;
    assert!(m.span().contains(last));
    assert!(m.time(TIME_MARKERS).is_none());
    Ok(())
}

#[test]
fn resize_teks_menjaga_kode_lama() -> Result<(), VirError> {
    let old = layout()?;
    let new = old.remap_after_text_resize(1_088)?;
    let delta = new.text_vocab() - old.text_vocab();
    for key in old.codebook_keys() {
        for code in [0u32, 17, 255] {
            let before = old
                .encode(key, code)
                .ok_or(VirError::Internal("encode lama".into()))?;
            let after = new
                .encode(key, code)
                .ok_or(VirError::Internal("encode baru".into()))?;
            assert_eq!(after, before + delta);
        }
    }
    assert!(old.remap_after_text_resize(999).is_err());
    Ok(())
}

#[test]
fn tolak_codebook_teks_ganda_dan_kosong() -> Result<(), VirError> {
    let mut layout = VocabLayout::new(256)?;
    let key = CodebookKey::new(ModalKind::Image, 0);
    assert!(layout.add_codebook(key, 0).is_err());
    layout.add_codebook(key, 128)?;
    assert!(layout.add_codebook(key, 128).is_err());
    assert!(layout
        .add_codebook(CodebookKey::new(ModalKind::Text, 0), 128)
        .is_err());
    assert!(VocabLayout::new(0).is_err());
    Ok(())
}

proptest::proptest! {
    #[test]
    fn resize_terus_menjaga_offset(
        text in 256u32..4096,
        levels in 1u8..4,
        codebook in 64u32..1024,
        grows in 0u32..512,
    ) {
        if let Ok(base) = build_layout(text, levels, levels, codebook) {
            if let Ok(moved) = base.remap_after_text_resize(text + grows) {
                let delta = moved.text_vocab() - base.text_vocab();
                proptest::prop_assert_eq!(moved.total(), base.total() + delta);
                for key in base.codebook_keys() {
                    let s0 = base.codebook_span(key).unwrap_or(Span::new(0, 0));
                    let s1 = moved.codebook_span(key).unwrap_or(Span::new(0, 0));
                    proptest::prop_assert_eq!(s1.start, s0.start + delta);
                    proptest::prop_assert_eq!(s1.len, s0.len);
                }
            }
        }
    }
}
