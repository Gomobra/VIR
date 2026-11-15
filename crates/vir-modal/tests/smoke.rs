//! Uji asap F6.0–F6.4 di level crate (`lib.rs`): tata letak standar, susunan
//! kode audio tiga level, dan gabungan tokenizer FSQ → decoder + codec SNAC.
//!
//! Dijalankan job CI `vir-modal`.

use vir_modal::token::{DEFAULT_CODEBOOK, SPAN_ALIGN};
use vir_modal::{layout_smoke, modal_smoke, standard_layout};

#[test]
fn standard_layout_matches_prd_vocab_size() {
    let layout = standard_layout(32_000).expect("layout");
    // teks + penanda (70) + codebook gambar/audio (3 + 3 level × 4096).
    assert_eq!(layout.marker_len(), 2 + 2 + (2 + 64));
    assert_eq!(layout.total(), 32_000 + 70 + 6 * DEFAULT_CODEBOOK);
    assert_eq!(layout.total(), 56_646);
    assert_eq!(layout.codebook_keys().count(), 6);
    assert_eq!(DEFAULT_CODEBOOK % SPAN_ALIGN, 0);
}

#[test]
fn layout_smoke_codes_three_audio_levels() {
    let ids = layout_smoke().expect("smoke");
    assert_eq!(ids, 1 + 3 + 2 + 1 + 1);
}

#[test]
fn modal_smoke_roundtrips_image_and_audio() {
    let (decoded, levels) = modal_smoke().expect("smoke");
    // Latent halus 16 posisi × 4 dimensi; codec SNAC 24 kHz punya 3 level.
    assert_eq!(decoded, 16 * 4);
    assert_eq!(levels, 3);
}
