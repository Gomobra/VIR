//! Tes nyata F2.3 (pembungkus block-autoregressive) untuk `vir-model`.
//!
//! Dijalankan job CI `vir-model` (`cargo test --manifest-path
//! crates/vir-model/Cargo.toml`). Logika loop murni (closure + `Vec<u32>`),
//! jadi tidak butuh tensor.

use vir_common::VirError;
use vir_model::{BlockArConfig, BlockMeta, block_ar};

/// Blok palsu: kembalikan `n` token yang nilainya = `start..start+n`.
#[derive(Clone)]
struct FakeBlock {
    tokens: Vec<u32>,
}

impl FakeBlock {
    fn new(start: u32, n: u32) -> Self {
        Self {
            tokens: (start..start + n).collect(),
        }
    }
}

#[test]
fn stops_at_max_blocks() {
    let cfg = BlockArConfig {
        max_blocks: 3,
        max_tokens: 100,
        eos: u32::MAX,
    };
    let mut prefix_seen: Vec<usize> = Vec::new();
    let (tokens, cost) = block_ar(
        &cfg,
        |p| {
            prefix_seen.push(p);
            let meta = BlockMeta {
                positions: 4,
                decode_positions: 10,
                next_prefix_len: p + 4,
            };
            Ok((FakeBlock::new(1, 2), meta))
        },
        |b| Ok(b.tokens.clone()),
    )
    .expect("loop harus sukses");

    assert_eq!(cost.blocks, 3, "berhenti di batas blok");
    assert_eq!(tokens.len(), 6);
    assert!(!cost.hit_eos);
    assert_eq!(cost.positions, 12);
    // Prefix membesar tiap blok: 0, 4, 8.
    assert_eq!(prefix_seen, vec![0, 4, 8]);
}

#[test]
fn stops_at_eos_and_excludes_eos() {
    let cfg = BlockArConfig {
        max_blocks: 8,
        max_tokens: 100,
        eos: 3,
    };
    // Blok kedua mengandung EOS (nilai 3) di tengah.
    let blocks = vec![
        FakeBlock::new(1, 2), // [1, 2]
        FakeBlock::new(3, 3), // [3, 4, 5] → EOS di depan
    ];
    let mut i = 0usize;
    let (tokens, cost) = block_ar(
        &cfg,
        |p| {
            let b = blocks.get(i).cloned().unwrap_or(FakeBlock::new(9, 1));
            i += 1;
            let meta = BlockMeta {
                positions: 2,
                decode_positions: 8,
                next_prefix_len: p + 2,
            };
            Ok((b, meta))
        },
        |b| Ok(b.tokens.clone()),
    )
    .expect("loop harus sukses");

    assert!(cost.hit_eos, "harus berhenti karena EOS");
    assert_eq!(tokens, vec![1, 2], "token sebelum EOS saja");
    assert_eq!(cost.blocks, 2);
}

#[test]
fn stops_at_max_tokens() {
    let cfg = BlockArConfig {
        max_blocks: 10,
        max_tokens: 3,
        eos: u32::MAX,
    };
    let (tokens, cost) = block_ar(
        &cfg,
        |_p| {
            let meta = BlockMeta {
                positions: 2,
                decode_positions: 8,
                next_prefix_len: _p + 2,
            };
            Ok((FakeBlock::new(1, 5), meta))
        },
        |b| Ok(b.tokens.clone()),
    )
    .expect("loop harus sukses");

    assert_eq!(tokens.len(), 3, "dipotong di batas token");
    assert!(!cost.hit_eos);
    // Blok pertama sudah mengisi 3 token → loop berhenti.
    assert_eq!(cost.blocks, 1);
}

#[test]
fn prefix_grows_across_blocks() {
    let cfg = BlockArConfig {
        max_blocks: 4,
        max_tokens: 100,
        eos: u32::MAX,
    };
    let mut seen: Vec<usize> = Vec::new();
    let (_, cost) = block_ar(
        &cfg,
        |p| {
            seen.push(p);
            let meta = BlockMeta {
                positions: 1,
                decode_positions: p + 1,
                next_prefix_len: p + 1,
            };
            Ok((FakeBlock::new(1, 1), meta))
        },
        |b| Ok(b.tokens.clone()),
    )
    .expect("loop harus sukses");

    assert_eq!(seen, vec![0, 1, 2, 3], "prefix bertambah 1 tiap blok");
    assert_eq!(cost.blocks, 4);
}

#[test]
fn invalid_config_is_rejected() {
    let bad_blocks = BlockArConfig {
        max_blocks: 0,
        ..BlockArConfig::new()
    };
    let err = block_ar::<(), _, _>(
        &bad_blocks,
        |_| Ok(((), BlockMeta::empty())),
        |_| Ok(vec![]),
    );
    assert!(matches!(err, Err(VirError::Config(_))));

    let bad_tokens = BlockArConfig {
        max_tokens: 0,
        ..BlockArConfig::new()
    };
    let err = block_ar::<(), _, _>(
        &bad_tokens,
        |_| Ok(((), BlockMeta::empty())),
        |_| Ok(vec![]),
    );
    assert!(matches!(err, Err(VirError::Config(_))));
}
