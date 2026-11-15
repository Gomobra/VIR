//! Tes integrasi pipeline data: shard → pack → stream → mix → FIM (PRD F1.3).

use proptest::prelude::*;
use vir_data::{
    FimMarkers, FimMode, FimParts, MixConfig, PackConfig, ShardReader, ShardWriter, SourceMix,
    SplitMix64, StreamConfig, TokenStream, fim_psm, fim_spm, load_repo_sources, mix_blocks,
    pack_blocks, pack_documents, parse_fim, read_shard, synth_arithmetic, synth_code, synth_corpus,
    synth_questions, synth_sentences, transform_fim, validate_weights, write_shard,
};
use vir_text::{BpeEncoder, SpecialIds, TrainConfig};

fn pid_path(name: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    std::env::temp_dir().join(format!("vir-data-it-{name}-{pid}.shard"))
}

fn empty_parts<'a>() -> FimParts<'a> {
    FimParts {
        prefix: &[],
        suffix: &[],
        middle: &[],
    }
}

fn corpus_tokens() -> Vec<u32> {
    let texts = synth_corpus(6);
    let mut encoder = BpeEncoder::new();
    encoder.train(&texts, 300);
    let mut tokens = Vec::new();
    for text in &texts {
        tokens.extend(encoder.encode(text));
    }
    tokens
}

#[test]
fn shard_roundtrip_multi_blok() {
    let path = pid_path("roundtrip");
    let tokens: Vec<u32> = (0..5000).collect();
    let mut writer = ShardWriter::create(&path, 512).expect("buat penulis");
    writer.write_tokens(&tokens[..1000]).expect("tulis 1");
    writer.write_tokens(&tokens[1000..]).expect("tulis 2");
    let header = writer.finish().expect("tutup");
    assert_eq!(header.token_count, 5000);
    assert_eq!(header.block_count, 10);
    let mut reader = ShardReader::open(&path).expect("buka");
    assert_eq!(reader.block_count(), 10);
    assert_eq!(read_shard(&path).expect("baca semua"), tokens);
    reader.verify().expect("verifikasi");
    assert_eq!(reader.block(9).expect("blok terakhir").len(), 392);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn shard_korup_ditolak() {
    let path = pid_path("korup");
    std::fs::write(&path, b"bukan shard sama sekali").expect("tulis");
    assert!(ShardReader::open(&path).is_err());
    assert!(read_shard(&path).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn stream_resume_lintas_epoch() {
    let cfg = StreamConfig {
        seed: 11,
        block_size: 16,
        ..StreamConfig::default()
    };
    let tokens: Vec<u32> = (0..300).collect();
    let mut stream = TokenStream::from_tokens(tokens.clone(), cfg);
    let mut first = Vec::new();
    for _ in 0..25 {
        first.push(stream.next_block().expect("blok"));
    }
    let state = stream.state();
    assert_eq!(state.produced, 25);
    assert_eq!(state.epoch, 1);
    let mut resumed = TokenStream::from_tokens(tokens, cfg);
    assert!(resumed.seek(state));
    let mut second = Vec::new();
    for _ in 0..25 {
        second.push(resumed.next_block().expect("blok"));
    }
    let mut straight = TokenStream::from_tokens((0..300).collect(), cfg);
    let mut all = Vec::new();
    for _ in 0..50 {
        all.push(straight.next_block().expect("blok"));
    }
    assert_eq!(first[..], all[..25]);
    assert_eq!(second[..], all[25..]);
}

#[test]
fn pack_mask_hanya_token_nyata() {
    let cfg = PackConfig {
        block_size: 8,
        ..PackConfig::default()
    };
    let tokens: Vec<u32> = (0..20).collect();
    let blocks = pack_blocks(&tokens, &cfg);
    assert_eq!(blocks.len(), 3);
    let real: Vec<u32> = blocks
        .iter()
        .flat_map(|block| {
            block
                .tokens
                .iter()
                .zip(block.mask.iter())
                .filter(|(_, keep)| **keep)
                .map(|(token, _)| *token)
                .collect::<Vec<u32>>()
        })
        .collect();
    assert_eq!(real, tokens);
    assert_eq!(blocks[2].real_tokens(), 4);
    assert_eq!(blocks[2].pad_tokens(), 4);
}

#[test]
fn mix_menghormati_bobot_dan_sumber_habis() {
    let sources = vec![
        SourceMix::new("id", 0.75, (0..30u32).map(|i| vec![i]).collect()),
        SourceMix::new("ru", 0.25, (0..10u32).map(|i| vec![i + 100]).collect()),
    ];
    assert!(validate_weights(&[0.5, 0.5]).is_ok());
    let cfg = MixConfig {
        seed: 3,
        total_blocks: 40,
    };
    let mixed = mix_blocks(&sources, &cfg).expect("campur");
    assert_eq!(mixed.len(), 40);
    let from_ru = mixed.iter().filter(|b| b[0] >= 100).count();
    assert!(from_ru >= 2, "sumber minoritas terlalu sedikit: {from_ru}");
    assert!(from_ru <= 10, "blok sumber habis dipakai ulang: {from_ru}");
    let unique: std::collections::BTreeSet<Vec<u32>> = mixed.iter().cloned().collect();
    assert_eq!(unique.len(), 40, "blok tidak boleh berulang");
}

#[test]
fn fim_reversibel_semua_mode() {
    let markers = FimMarkers::defaults();
    let ids = SpecialIds::new();
    let prefix = vec![ids.bos, 1, 2];
    let suffix = vec![3, ids.eos];
    let middle = vec![9, 8, 7, ids.mask];
    for mode in [FimMode::Psm, FimMode::Spm] {
        let tokens = transform_fim(mode, &prefix, &suffix, &middle, &markers);
        let parts = parse_fim(&tokens, &markers).unwrap_or_else(empty_parts);
        assert_eq!(parts.prefix, prefix.as_slice(), "mode {mode:?}");
        assert_eq!(parts.suffix, suffix.as_slice(), "mode {mode:?}");
        assert_eq!(parts.middle, middle.as_slice(), "mode {mode:?}");
    }
    let psm = fim_psm(&[1], &[2], &[3]);
    let spm = fim_spm(&[1], &[2], &[3]);
    assert_ne!(psm, spm);
    assert_eq!(psm.len(), 6);
    assert_eq!(parse_fim(&psm, &markers).map(|p| p.middle.len()), Some(1));
}

#[test]
fn synth_deterministik_dan_kaya() {
    assert_eq!(synth_arithmetic(12), synth_arithmetic(12));
    assert_eq!(synth_questions(12), synth_questions(12));
    assert_eq!(synth_code(12), synth_code(12));
    assert_eq!(synth_sentences(12), synth_sentences(12));
    let corpus = synth_corpus(5);
    assert_eq!(corpus.len(), 15);
    let all: String = corpus.concat();
    for word in ["Pada hari", "On day", "В день", "日目に", "def "] {
        assert!(all.contains(word), "sintetis tanpa {word}");
    }
    let mut rng = SplitMix64::new(2);
    assert!(rng.next_f64() < 1.0);
}

#[test]
fn katalog_sumber_repo_muat_nyata() {
    let catalog = load_repo_sources().expect("data/sources.json");
    assert!(catalog.len() >= 4);
    let ids = catalog.ids();
    assert!(ids.contains(&"tinystories"));
    assert!(ids.contains(&"openwebtext"));
    assert!(catalog.weights().iter().all(|w| *w >= 0.0));
}

#[test]
fn pipeline_tokenizer_ke_shard_stream_fim() {
    let tokens = corpus_tokens();
    assert!(tokens.len() > 100);
    let ids = SpecialIds::new();
    let cfg = PackConfig {
        block_size: 32,
        ..PackConfig::default()
    };
    let blocks = pack_documents([tokens.as_slice()], &cfg);
    assert!(blocks.iter().all(|block| block.len() == 32));

    let path = pid_path("pipeline");
    write_shard(&path, &tokens).expect("tulis shard");
    let mut stream = TokenStream::from_shard(
        &path,
        StreamConfig {
            block_size: 32,
            ..StreamConfig::default()
        },
    )
    .expect("stream");
    let mut produced = 0usize;
    while let Some(block) = stream.next_block() {
        produced += block.len();
    }
    let padded = tokens.len().div_ceil(32) * 32;
    assert_eq!(produced, padded);

    let markers = FimMarkers::defaults();
    let third = tokens.len() / 3;
    let (prefix, rest) = tokens.split_at(third);
    let (suffix, middle) = rest.split_at(third);
    let fim = transform_fim(FimMode::Psm, prefix, suffix, middle, &markers);
    assert_eq!(fim.len(), tokens.len() + 3);
    assert_eq!(fim[0], ids.fim_prefix);
    let parts = parse_fim(&fim, &markers).expect("parse fim");
    assert_eq!(parts.prefix, prefix);
    assert_eq!(parts.suffix, suffix);
    assert_eq!(parts.middle, middle);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn tokenizer_belajar_dari_sintetis() {
    let texts = synth_corpus(8);
    let mut encoder = BpeEncoder::new();
    encoder.train(&texts, 320);
    assert!(encoder.table().merge_count() >= 20);
    assert!(encoder.vocab_size() <= 320);
    for text in &texts {
        let encoded = encoder.encode(text);
        assert_eq!(encoder.decode_bytes(&encoded), text.as_bytes());
    }
    let table = BpeEncoder::trained(synth_sentences(4), &TrainConfig::default());
    assert!(table.vocab_size() > 271);
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(32))]

    #[test]
    fn proptest_pack_mask_restoratif(
        tokens in proptest::collection::vec(0u32..1000, 0..64),
        block in 1usize..16
    ) {
        let cfg = PackConfig { block_size: block, ..PackConfig::default() };
        let blocks = pack_blocks(&tokens, &cfg);
        let real: Vec<u32> = blocks
            .iter()
            .flat_map(|b| {
                b.tokens
                    .iter()
                    .zip(b.mask.iter())
                    .filter(|(_, keep)| **keep)
                    .map(|(t, _)| *t)
                    .collect::<Vec<u32>>()
            })
            .collect();
        prop_assert_eq!(real, tokens);
        prop_assert!(blocks.iter().all(|b| b.len() == block));
    }

    #[test]
    fn proptest_fim_reversibel(part in proptest::collection::vec(100u32..200, 0..8)) {
        let markers = FimMarkers::defaults();
        let tokens = transform_fim(FimMode::Psm, &part, &part, &part, &markers);
        let parts = parse_fim(&tokens, &markers);
        prop_assert_eq!(parts.map(|p| p.prefix.len()), Some(part.len()));
        prop_assert_eq!(parts.map(|p| p.middle), Some(part.as_slice()));
    }
}
