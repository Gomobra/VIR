//! Tes nyata F6.4 untuk RVQ (`audio.rs`) dan codec SNAC (`snac.rs`): laju per
//! level 12/23/47 Hz pada 24 kHz, bitrate 984 bps (PRD F6.4), jumlah kode
//! 82/detik, serta round-trip kuantisasi residual.
//!
//! Dijalankan job CI `vir-modal`. Semua fungsi murni (`Vec<f32>`).

use vir_common::VirError;
use vir_modal::audio::Rvq;
use vir_modal::snac::{SnacCodec, SnacConfig};

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-3
}

fn is_err<T>(r: Result<T, VirError>) -> bool {
    r.is_err()
}

/// RVQ dua level dengan codebook nyata (bukan nol) supaya pencarian kode dan
/// rekonstruksi residual benar-benar teruji.
fn two_level_rvq() -> Rvq {
    let mut rvq = Rvq::new(4, 2, 2);
    rvq.codebooks[0] = vec![
        vec![0.0, 0.0],
        vec![4.0, 0.0],
        vec![0.0, 4.0],
        vec![4.0, 4.0],
    ];
    rvq.codebooks[1] = vec![
        vec![0.0, 0.0],
        vec![2.0, 0.0],
        vec![0.0, 2.0],
        vec![2.0, 2.0],
    ];
    rvq
}

// ---------------------------------------------------------------------------
// Rvq
// ---------------------------------------------------------------------------

#[test]
fn rvq_shapes_and_validation() {
    let rvq = Rvq::new(4, 2, 3);
    assert_eq!(rvq.levels(), 3);
    assert_eq!(rvq.dim(), 2);
    assert_eq!(rvq.codebook_size(), 4);
    assert!(rvq.validate().is_ok());
    assert_eq!(Rvq::new(4, 2, 2).codebooks.len(), 2);
    assert!(Rvq::new(4, 2, 2).codebooks[0].iter().all(|e| e.len() == 2));

    let mut dims = Rvq::new(4, 2, 2);
    dims.dims[1] = 3;
    assert!(is_err(dims.validate()));

    let mut entries = Rvq::new(4, 2, 2);
    entries.codebooks[1][0] = vec![0.0, 0.0, 0.0];
    assert!(is_err(entries.validate()));

    assert!(is_err(Rvq::new(0, 2, 2).validate()));
    assert!(is_err(Rvq::new(4, 0, 2).validate()));
}

#[test]
fn rvq_nearest_entry_search() {
    let rvq = two_level_rvq();
    assert_eq!(rvq.encode_level(0, &[9.0, 9.0], 1, 2).ok(), Some(vec![3]));
    assert_eq!(rvq.encode_level(0, &[0.5, 0.0], 1, 2).ok(), Some(vec![0]));
    assert_eq!(rvq.encode_level(1, &[0.0, 0.0], 1, 2).ok(), Some(vec![0]));
    assert_eq!(
        rvq.decode_level(0, &[3, 2], 2).ok(),
        Some(vec![4.0, 4.0, 0.0, 4.0])
    );
    assert!(is_err(rvq.decode_level(0, &[9], 2)));
    assert!(is_err(rvq.encode_level(3, &[0.0, 0.0], 1, 2)));
    assert!(is_err(rvq.encode_level(0, &[0.0, 0.0, 0.0], 1, 2)));
    assert!(is_err(rvq.encode_level(0, &[0.0, 0.0], 1, 3)));
}

#[test]
fn rvq_residual_roundtrip_is_exact() {
    let rvq = two_level_rvq();
    // [4,4] tepat di codebook level 0; residual nol di level 1.
    let codes = rvq.encode(&[4.0, 4.0], 1, 2).expect("encode");
    assert_eq!(codes, vec![vec![3], vec![0]]);
    let back = rvq.decode(&codes, 1, 2).expect("decode");
    assert_eq!(back, vec![4.0, 4.0]);

    // [5,5]: level 0 memilih [4,4], residual [1,1] → level 1 memilih [2,0].
    let codes = rvq.encode(&[5.0, 5.0], 1, 2).expect("encode");
    assert_eq!(codes, vec![vec![3], vec![1]]);
    let back = rvq.decode(&codes, 1, 2).expect("decode");
    assert_eq!(back, vec![6.0, 4.0]);

    assert!(is_err(rvq.decode(&codes, 2, 2)));
    assert!(is_err(rvq.decode(&[vec![0], vec![0], vec![0]], 1, 2)));
    assert!(is_err(rvq.encode(&[0.0, 0.0, 0.0], 1, 2)));
}

#[test]
fn rvq_bitrate_matches_rates() {
    let rvq = Rvq::new(4_096, 8, 3);
    assert!(approx(
        rvq.bitrate(&[12.0, 23.0, 47.0]).expect("bitrate"),
        984.0
    ));
    assert!(approx(
        rvq.bitrate(&[12.0, 23.0, 47.0]).expect("bitrate"),
        82.0 * 12.0
    ));
    assert!(is_err(rvq.bitrate(&[12.0, 23.0])));
}

// ---------------------------------------------------------------------------
// SnacConfig
// ---------------------------------------------------------------------------

#[test]
fn snac24k_rates_match_prd() {
    let cfg = SnacConfig::snac24k();
    assert_eq!(cfg.sample_rate, 24_000);
    assert_eq!(cfg.strides, vec![4, 2, 1]);
    assert_eq!(cfg.hop, 512);
    assert_eq!((cfg.codebook_bits, cfg.codebook_size), (12, 4_096));
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.rates().expect("rates"), vec![12.0, 23.0, 47.0]);
    assert!(approx(cfg.bitrate_bps().expect("bps"), 984.0));
    assert!(approx(cfg.check_bitrate().expect("check"), 984.0));
    assert_eq!(cfg.total_codes(10_000).ok(), Some(820));
    assert_eq!(cfg.frames(10_000).ok(), Some(vec![117, 234, 468]));
    assert!(is_err(cfg.frames(0)));
}

#[test]
fn snac_other_sample_rates_match_prd() {
    let c32 = SnacConfig::snac32k();
    assert_eq!(c32.strides, vec![8, 4, 2, 1]);
    assert_eq!(c32.hop, 384);
    assert_eq!(c32.rates().expect("rates"), vec![10.0, 21.0, 42.0, 83.0]);
    assert!(approx(c32.check_bitrate().expect("check"), 1_872.0));

    let c44 = SnacConfig::snac44k();
    assert_eq!(c44.sample_rate, 44_100);
    assert_eq!(c44.rates().expect("rates"), vec![14.0, 29.0, 57.0, 115.0]);
    assert!(approx(c44.check_bitrate().expect("check"), 2_580.0));
}

#[test]
fn snac_config_rejects_bad_strides_and_codebooks() {
    let mut cfg = SnacConfig::snac24k();
    cfg.strides = vec![4, 2, 2];
    assert!(is_err(cfg.validate()));
    cfg.strides = vec![4, 2];
    assert!(is_err(cfg.validate()));
    cfg.strides = vec![4, 0, 1];
    assert!(is_err(cfg.validate()));
    cfg.strides = vec![];
    assert!(is_err(cfg.validate()));
    cfg.strides = vec![1];
    assert!(cfg.validate().is_ok());

    let mut cfg = SnacConfig::snac24k();
    cfg.codebook_size = 5;
    assert!(is_err(cfg.validate()));
    cfg.codebook_size = 4_096;
    cfg.hop = 0;
    assert!(is_err(cfg.validate()));
    cfg.hop = 512;
    cfg.dim = 0;
    assert!(is_err(cfg.validate()));
}

#[test]
fn snac_bitrate_must_match_reference_rates() {
    let unknown = SnacConfig {
        sample_rate: 8_000,
        ..SnacConfig::snac24k()
    };
    assert!(unknown.validate().is_ok());
    assert!(is_err(unknown.check_bitrate()));

    let mut drift = SnacConfig::snac24k();
    drift.hop = 500;
    assert_eq!(drift.rates().expect("rates"), vec![12.0, 24.0, 48.0]);
    assert!(is_err(drift.check_bitrate()));
}

// ---------------------------------------------------------------------------
// SnacCodec
// ---------------------------------------------------------------------------

#[test]
fn snac_codec_bitrate_and_codes() {
    let codec = SnacCodec::snac24k().expect("codec");
    assert_eq!(codec.cfg.dim, 8);
    assert!(approx(codec.bitrate_bps().expect("bps"), 984.0));
    assert!(approx(codec.check_bitrate().expect("check"), 984.0));
    assert_eq!(codec.codes(10_000).ok(), Some(820));
    assert_eq!(codec.frames(10_000).ok(), Some(vec![117, 234, 468]));
    assert_eq!(codec.rvq.levels(), 3);
    assert_eq!(codec.rvq.codebook_size(), 4_096);
}

#[test]
fn snac_encode_decode_shapes_and_finite() {
    let codec = SnacCodec::snac24k().expect("codec");
    let latents = vec![0.0f32; codec.cfg.dim * 12];
    let codes = codec.encode(&latents).expect("encode");
    assert_eq!(codes.len(), 3);
    assert_eq!(codes[0].len(), 3);
    assert_eq!(codes[1].len(), 6);
    assert_eq!(codes[2].len(), 12);

    let back = codec.decode(&codes).expect("decode");
    assert_eq!(back.len(), latents.len());
    assert!(back.iter().all(|v| v.is_finite()));

    assert!(is_err(codec.encode(&vec![0.0; 7])));
    assert!(is_err(codec.encode(&[])));
    assert!(is_err(codec.decode(&codes[..2])));
    assert!(is_err(codec.decode(&[vec![0; 3], vec![0; 6], vec![0; 5]])));
    assert!(is_err(codec.decode(&[])));
}

#[test]
fn snac_reconstructs_latent_inside_codebook() {
    let cfg = SnacConfig {
        dim: 2,
        codebook_size: 2,
        codebook_bits: 1,
        ..SnacConfig::snac24k()
    };
    let mut codec = SnacCodec::new(cfg).expect("codec");
    codec.rvq.codebooks = vec![
        vec![vec![1.0, 0.0], vec![-1.0, 0.0]],
        vec![vec![0.0, 0.0], vec![0.5, 0.0]],
        vec![vec![0.0, 0.0], vec![0.25, 0.0]],
    ];
    assert!(codec.rvq.validate().is_ok());

    // [1.5, 0] = level 0 ([1,0]) + level 1 ([0.5,0]); level 2 menutup residual.
    let latents = vec![1.5f32, 0.0, 1.5, 0.0, 1.5, 0.0, 1.5, 0.0];
    let codes = codec.encode(&latents).expect("encode");
    assert_eq!(codes, vec![vec![0], vec![1, 1], vec![0, 0, 0, 0]]);
    let back = codec.decode(&codes).expect("decode");
    assert_eq!(back.len(), latents.len());
    for (got, want) in back.iter().zip(latents.iter()) {
        assert!(approx(*got, *want), "{got} != {want}");
    }
}
