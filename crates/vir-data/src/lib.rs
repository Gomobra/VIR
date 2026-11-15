//! Pipeline data VIR: shard, stream, pack, mix, FIM, sintetis, sumber (PRD F1.3).
//!
//! | modul | isi |
//! |---|---|
//! | [`shard`] | format shard biner (header + blok zstd), penulis dan pembaca |
//! | [`stream`] | urutan deterministik dari seed + state resume |
//! | [`pack`] | blok panjang tetap dengan EOS pemisah dan mask padding |
//! | [`mix`] | pencampuran blok antar bahasa menurut bobot |
//! | [`fim`] | transformasi PSM/SPM dan inversnya |
//! | [`synth`] | korpus sintetis: aritmetika, kode, kalimat templat |
//! | [`sources`] | pemuat `data/sources.json` (sumber dan lisensi nyata) |
//!
//! Alur pakai: `sources::load_repo_sources()` → teks → `vir_text` encode →
//! `pack::pack_documents` → `shard::ShardWriter` → `stream::TokenStream::from_shard`
//! → `mix::mix_blocks`.

mod fim;
mod mix;
mod pack;
mod shard;
mod sources;
mod stream;
mod synth;

pub use fim::{
    FimMarkers, FimMode, FimParts, fim_psm, fim_spm, parse as parse_fim, transform as transform_fim,
};
pub use mix::{MixConfig, SourceMix, mix_blocks, pick_source, proportions, validate_weights};
pub use pack::{
    DEFAULT_BLOCK_SIZE, PackConfig, PackedBlock, pack_blocks, pack_documents, pack_tokens,
};
pub use shard::{
    DEFAULT_BLOCK_TOKENS, DType, HEADER_BYTES, SHARD_MAGIC, SHARD_VERSION, ShardHeader,
    ShardReader, ShardWriter, ZSTD_LEVEL, bytes_to_tokens, read_shard, tokens_to_bytes,
    write_shard,
};
pub use sources::{
    SourceCatalog, SourceEntry, load_repo_sources, load_sources, parse_sources, repo_sources_path,
};
pub use stream::{SplitMix64, StreamConfig, StreamState, TokenStream};
pub use synth::{synth_arithmetic, synth_code, synth_corpus, synth_questions, synth_sentences};

/// Format SFT: instruksi → jawaban; kemas sebagai pasangan `(prefix, target)`.
pub fn sft_pack(instruction: &str, answer: &str) -> (String, String) {
    (instruction.to_string(), answer.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sft_pack_mempertahankan_teks() {
        let (instruction, answer) = sft_pack("tanya", "jawab");
        assert_eq!(instruction, "tanya");
        assert_eq!(answer, "jawab");
    }

    #[test]
    fn jalur_pendek_shard_stream_pack() {
        let texts = synth_arithmetic(16);
        let tokens: Vec<u32> = texts
            .iter()
            .flat_map(|t| t.bytes().map(u32::from))
            .collect();
        let pack = PackConfig {
            block_size: 8,
            ..PackConfig::default()
        };
        let blocks = pack_blocks(&tokens, &pack);
        assert!(!blocks.is_empty());
        assert!(blocks.iter().all(|block| block.len() == 8));
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("vir-data-lib-{pid}.shard"));
        write_shard(&path, &tokens).expect("tulis shard");
        let mut reader = ShardReader::open(&path).expect("buka shard");
        assert_eq!(reader.tokens().expect("token"), tokens);
        let stream = TokenStream::from_shard(
            &path,
            StreamConfig {
                block_size: 8,
                ..StreamConfig::default()
            },
        )
        .expect("stream dari shard");
        assert!(stream.epoch_len() > 0);
        let _ = std::fs::remove_file(&path);
    }
}
