//! Tokenizer BPE tingkat byte tanpa HF Tokenizers (PRD F1.2).
//!
//! | modul | isi |
//! |---|---|
//! | [`special`] | token khusus: PAD, BOS, EOS, MASK, FIM, cadangan modalitas |
//! | [`pretok`] | pra-tokenisasi byte aman Jepang/Rusia |
//! | [`bpe`] | vocab dan tabel merge |
//! | [`train`] | pelatih paralel rayon dari iterator teks |
//! | [`encoder`] | trait [`Encoder`], implementasi acuan, registri backend |
//! | [`export`] | ekspor dan pemuat format tiktoken |
//!
//! Jaminan utama: `decode_bytes(encode(text)) == text.as_bytes()` untuk semua
//! teks byte-level, termasuk byte yang bukan UTF-8 (lewat `encode` pada byte mentah).
//! Token khusus tidak pernah dihasilkan `encode`; ia ditambahkan oleh `vir-data`.

mod bpe;
mod encoder;
mod export;
mod pretok;
mod special;
mod train;

pub use bpe::{MergeTable, Pair};
pub use encoder::{BackendKind, BpeEncoder, Encoder, resolve_backend};
pub use export::{export_specials, export_tiktoken, load, load_tiktoken, save};
pub use pretok::{CharClass, DEFAULT_MAX_WORD_BYTES, PreTokenizer, classify, pretokenize};
pub use special::{
    BOS, BYTE_VOCAB, EOS, FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX, MASK, MODAL_RESERVE, PAD,
    SPECIAL_TOKENS, SpecialIds,
};
pub use train::{DEFAULT_MIN_FREQUENCY, DEFAULT_TARGET_VOCAB, TrainConfig, train_bpe};

/// Nama lama `BpeTokenizer`, dipakai `vir-py` dan `vir-cli`.
pub type BpeTokenizer = BpeEncoder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_byte_persis_multibahasa() {
        let encoder = BpeEncoder::default();
        for text in [
            "halo dunia",
            "Привет, мир!",
            "日本語テスト",
            "😀😀",
            "def f(x):\n\treturn x",
        ] {
            let ids = encoder.encode(text);
            assert_eq!(encoder.decode_bytes(&ids), text.as_bytes());
        }
    }
}
