//! Tipe data dan trait multimodal (PRD F6.0, ADR-009).
//!
//! Crate ini adalah **landasan data** untuk Fase 6: ia mendefinisikan
//! representasi modalitas yang dipakai bersama oleh generator, tokenizer, codec,
//! dan loader data. Tidak ada tensor di sini (aturan R1/R7 + Termux): encoder
//! dan decoder nyata mengimplementasikan [`ModalEncoder`] di crate tensor.
//!
//! | Modul | Isi |
//! |---|---|
//! | [`kind`] | [`ModalKind`] — kunci seluruh keputusan representasi |
//! | [`token`] | [`ModalToken`], [`Span`], [`VocabLayout`] — ruang ID + perluasan vocab |
//! | [`sequence`] | perakitan urutan level-major + penanda waktu video |
//! | [`asset`] | [`ModalAsset`] — referensi file + metadata anggaran token |
//! | [`sample`] | [`PairedSample`] — JSONL berpasangan + lisensi |
//! | [`encoder`] | trait [`ModalEncoder`] + [`EncoderSpec`] (encoder beku) |
//! | [`budget`] | rumus anggaran token (196/576, 500, 820, bitrate SNAC) |
//!
//! Keputusan yang sudah dikunci (ADR-009):
//!
//! 1. Dua jalur: **kode diskret berlevel** (generasi gambar/audio) dan **fitur
//!    encoder beku + konektor** (pemahaman). Keduanya berakhir di satu ruang ID.
//! 2. Vocab = blok teks, blok penanda, lalu codebook per `(kind, level)` yang
//!    disejajarkan 64 baris.
//! 3. Video hanya pemahaman: frame → token gambar + penanda waktu.
//! 4. Data berpasangan = JSONL, `license` wajib (`unknown` ditolak bawaan).

pub mod asset;
pub mod budget;
pub mod connector;
pub mod encoder;
pub mod image;
pub mod kind;
pub mod sample;
pub mod sequence;
pub mod token;

pub use asset::ModalAsset;
pub use budget::{
    audio_bitrate, audio_codes, audio_understanding_tokens, image_tokens, prefix_budget,
    snac24k_bitrate, snac24k_codes, video_tokens, DEFAULT_PATCH, SNAC24K_RATES, SNAC_CODEBOOK_BITS,
    SPEECH_POSITIONS_PER_SEC,
};
pub use encoder::{budget_of, spec_budget, EncoderSpec, ModalEncoder};
pub use kind::ModalKind;
pub use sample::{read_jsonl, PairedSample, Segment, SegmentRole, UNKNOWN_LICENSE};
pub use sequence::{encode_codes, encode_video};
pub use token::{
    build_layout, CodebookKey, Markers, ModalToken, Region, Span, VocabLayout, DEFAULT_CODEBOOK,
    SPAN_ALIGN, TIME_MARKERS,
};

use vir_common::VirError;

/// Tata letak standar F6 (ADR-009 §5) dengan angka yang dipakai PRD.
pub fn standard_layout(text_vocab: u32) -> Result<VocabLayout, VirError> {
    build_layout(text_vocab, 3, 3, DEFAULT_CODEBOOK)
}

/// Uji asap landasan: susun satu urutan audio SNAC 3 level dan cek ID-nya.
pub fn layout_smoke() -> Result<usize, VirError> {
    let layout = standard_layout(32_000)?;
    let levels = vec![vec![0, 1, 2], vec![0, 1], vec![0]];
    let ids = encode_codes(&layout, ModalKind::Audio, &levels)?;
    for id in &ids {
        if layout.region(*id).is_none() {
            return Err(VirError::Internal(format!("ID {id} di luar vocab")));
        }
    }
    Ok(ids.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_standar_punya_lima_codebook() -> Result<(), VirError> {
        let layout = standard_layout(32_000)?;
        assert_eq!(layout.codebook_keys().count(), 6);
        assert!(layout.total() > 32_000 + 72);
        Ok(())
    }

    #[test]
    fn asap_layout_audio() -> Result<(), VirError> {
        assert_eq!(layout_smoke()?, 2 + 6);
        Ok(())
    }

    #[test]
    fn anggaran_token_prd() -> Result<(), VirError> {
        assert_eq!(image_tokens(224, 224, DEFAULT_PATCH)?, 196);
        assert_eq!(image_tokens(384, 384, DEFAULT_PATCH)?, 576);
        assert_eq!(snac24k_codes(10_000)?, 820);
        assert!((snac24k_bitrate() - 984.0).abs() < 1e-3);
        Ok(())
    }
}
