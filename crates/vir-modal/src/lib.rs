//! Tipe data dan trait multimodal (PRD F6.0–F6.5, ADR-009).
//!
//! Crate ini adalah **landasan data + algoritma ringan** untuk Fase 6: ia
//! mendefinisikan representasi modalitas yang dipakai bersama oleh generator,
//! tokenizer, codec, dan loader data. Tidak ada tensor dan tidak ada Burn
//! (aturan R1/R7 + Termux): semua numerik memakai `Vec<f32>` row-major, dan
//! jaringan berat (SigLIP, Whisper penuh, vocoder SNAC) tinggal di crate tensor
//! dengan bobot `safetensors`.
//!
//! | Modul | Isi |
//! |---|---|
//! | [`kind`] | [`ModalKind`] — kunci seluruh keputusan representasi |
//! | [`token`] | [`ModalToken`], [`Span`], [`VocabLayout`] — ruang ID + perluasan vocab |
//! | [`sequence`] | perakitan urutan level-major + penanda waktu video |
//! | [`asset`] | [`ModalAsset`] — referensi file + metadata anggaran token |
//! | [`sample`] | [`PairedSample`] — JSONL berpasangan + lisensi |
//! | [`encoder`] | trait [`ModalEncoder`] + [`EncoderSpec`] + [`Prefix`] |
//! | [`vision`] | [`ImageEncoder`] — patchify + konektor (F6.1) |
//! | [`speech`] | [`AudioEncoder`] Whisper-style: log-mel + konektor (F6.2) |
//! | [`mel`] | front-end log-mel + FFT radix-2 (tanpa dependensi) |
//! | [`image`] | [`FsqQuantizer`] + [`ModalTokenizer`] (F6.3) |
//! | [`multiscale`] | [`MultiScaleTokenizer`] — level 0 kasar → halus (F6.3) |
//! | [`decode`] | trait [`ImageDecoder`] + [`LinearImageDecoder`] (F6.3) |
//! | [`audio`] | [`Rvq`] residual (F6.4) |
//! | [`snac`] | [`SnacCodec`] + pemeriksaan bitrate 984 bps (F6.4) |
//! | [`pool`] | operator `Down_k`/`R_k` (average pooling, nearest) |
//! | [`video`] | sampler frame + prefix per frame (F6.5) |
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
pub mod audio;
pub mod budget;
pub mod connector;
pub mod decode;
pub mod encoder;
pub mod image;
pub mod kind;
pub mod mel;
pub mod multiscale;
pub mod pool;
pub mod sample;
pub mod sequence;
pub mod snac;
pub mod speech;
pub mod token;
pub mod video;
pub mod vision;

pub use asset::ModalAsset;
pub use audio::Rvq;
pub use budget::{
    DEFAULT_PATCH, SNAC_CODEBOOK_BITS, SNAC24K_RATES, SPEECH_POSITIONS_PER_SEC, audio_bitrate,
    audio_codes, audio_understanding_tokens, image_tokens, prefix_budget, snac24k_bitrate,
    snac24k_codes, video_tokens,
};
pub use connector::{Connector, Linear, ModalTokenizer};
pub use decode::{ImageDecoder, LinearImageDecoder};
pub use encoder::{EncoderSpec, ModalEncoder, Prefix, budget_of, spec_budget};
pub use image::FsqQuantizer;
pub use kind::ModalKind;
pub use mel::{MelSpec, log_mel};
pub use multiscale::MultiScaleTokenizer;
pub use sample::{PairedSample, Segment, SegmentRole, UNKNOWN_LICENSE, read_jsonl};
pub use sequence::{encode_codes, encode_video};
pub use snac::{SnacCodec, SnacConfig};
pub use speech::AudioEncoder;
pub use token::{
    CodebookKey, DEFAULT_CODEBOOK, Markers, ModalToken, Region, SPAN_ALIGN, Span, TIME_MARKERS,
    VocabLayout, build_layout,
};
pub use video::{FrameSelection, encode_frames, sample_fixed, sample_scene_change, select_frames};
pub use vision::ImageEncoder;

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

/// Uji asap F6.3/F6.4: tokenizer gambar FSQ → decoder, dan codec audio SNAC.
pub fn modal_smoke() -> Result<(usize, usize), VirError> {
    let fsq = FsqQuantizer::identity(vec![8; 4]);
    let lengths = MultiScaleTokenizer::grid_lengths(16, 2)?;
    let tokenizer = MultiScaleTokenizer::new(fsq, lengths)?;
    let latent = vec![0.1f32; 16 * 4];
    let levels = tokenizer.encode(&latent)?;
    let back = tokenizer.decode(&levels)?;
    let codec = SnacCodec::snac24k()?;
    let codes = codec.encode(&vec![0.0f32; codec.cfg.dim * 12])?;
    Ok((back.len(), codes.len()))
}
