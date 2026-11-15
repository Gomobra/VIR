//! Tes nyata F6.3 untuk kuantisasi gambar: `FsqQuantizer` (kode dalam rentang,
//! `dequantize` = kebalikan `quantize`, indeks kode), tokenizer multiscale
//! (level kasar → halus, rekonstruksi laten), dan decoder piksel linear.
//!
//! Dijalankan job CI `vir-modal`. Semua fungsi murni (`Vec<f32>`).

use vir_common::VirError;
use vir_modal::connector::{Linear, ModalTokenizer};
use vir_modal::decode::{ImageDecoder, LinearImageDecoder};
use vir_modal::image::FsqQuantizer;
use vir_modal::multiscale::MultiScaleTokenizer;

fn is_err<T>(r: Result<T, VirError>) -> bool {
    r.is_err()
}

fn flat_codes(row: u32, rows: usize, n: usize) -> Vec<u32> {
    vec![row; rows * n]
}

// ---------------------------------------------------------------------------
// FsqQuantizer
// ---------------------------------------------------------------------------

#[test]
fn fsq_identity_codebook_size_and_ranges() {
    let fsq = FsqQuantizer::identity(vec![8; 4]);
    assert!(fsq.validate().is_ok());
    assert_eq!(fsq.n, 4);
    assert_eq!(fsq.d, 4);
    assert_eq!(fsq.codebook_size(), 4_096);
    assert_eq!(
        FsqQuantizer::identity(vec![5, 8, 3, 2]).codebook_size(),
        240
    );

    let z = vec![3.0f32, -3.0, 100.0, -100.0];
    let (codes, _) = fsq.quantize(&z).expect("quantize");
    assert_eq!(codes, vec![7, 0, 7, 0]);
    for code in &codes {
        assert!(*code < 8, "kode {code} di luar level");
    }
}

#[test]
fn fsq_zero_maps_to_level_base() {
    let fsq = FsqQuantizer::identity(vec![8; 4]);
    let (codes, q) = fsq.quantize(&[0.0; 4]).expect("quantize");
    assert_eq!(codes, vec![4, 4, 4, 4]);
    assert!(q.iter().all(|v| v.abs() < 1e-6));
    assert_eq!(fsq.dequantize(&codes).expect("dequantize"), vec![0.0; 4]);

    let odd = FsqQuantizer::identity(vec![5; 2]);
    assert_eq!(odd.quantize(&[0.0, 0.0]).expect("quantize").0, vec![2, 2]);
}

#[test]
fn fsq_codes_are_monotone_in_projection() {
    let fsq = FsqQuantizer::identity(vec![8; 4]);
    let z = [-3.0f32, -1.0, 0.0, 1.0];
    let (codes, _) = fsq.quantize(&z).expect("quantize");
    assert_eq!(codes, vec![0, 1, 4, 7]);
    for pair in codes.windows(2) {
        assert!(pair[0] <= pair[1], "kode tidak monoton: {codes:?}");
    }
}

#[test]
fn fsq_dequantize_inverts_quantize() {
    let fsq = FsqQuantizer::identity(vec![8; 4]);
    let z = vec![0.4f32, -1.6, 2.2, -1.2, 3.0, -3.0, 0.0, 1.0];
    let (codes, q) = fsq.quantize(&z).expect("quantize");
    assert_eq!(codes.len(), 8);
    let back = fsq.dequantize(&codes).expect("dequantize");
    assert_eq!(back, q);
    assert!(back.iter().all(|v| (-4.0..=3.0).contains(v)));

    let (r_codes, r_back) = fsq.roundtrip(&z).expect("roundtrip");
    assert_eq!(r_codes, codes);
    assert_eq!(r_back, q);

    // Proyeksi acak: `dequantize(quantize(z))` tetap sama dengan laten kuat.
    let proj = FsqQuantizer {
        levels: vec![8, 8],
        proj: vec![0.5, -0.25, 0.1, 0.3],
        n: 2,
        d: 2,
    };
    let z = vec![0.2f32, 0.4, -0.6, 0.8];
    let (codes, q) = proj.quantize(&z).expect("quantize");
    assert_eq!(codes.len(), 4);
    assert_eq!(proj.dequantize(&codes).expect("dequantize"), q);
    assert!(q.iter().all(|v| v.is_finite()));
}

#[test]
fn fsq_code_index_roundtrip() {
    let fsq = FsqQuantizer::identity(vec![8; 4]);
    assert_eq!(fsq.code_index(&[0; 4]).ok(), Some(0));
    assert_eq!(fsq.code_index(&[7; 4]).ok(), Some(4_095));
    let codes = vec![4u32, 0, 7, 3];
    let index = fsq.code_index(&codes).expect("index");
    assert_eq!(fsq.index_codes(index).ok(), Some(codes));
    for index in [0u32, 1, 255, 2_340, 4_095] {
        let codes = fsq.index_codes(index).expect("codes");
        assert_eq!(fsq.code_index(&codes).ok(), Some(index));
    }
    assert!(is_err(fsq.code_index(&[8, 0, 0, 0])));
    assert!(is_err(fsq.code_index(&[0, 0, 0])));
    assert!(is_err(fsq.index_codes(4_096)));
}

#[test]
fn fsq_validate_and_modal_tokenizer() {
    assert!(is_err(FsqQuantizer::new(vec![], 4).validate()));
    assert!(is_err(FsqQuantizer::new(vec![1; 4], 4).validate()));
    assert!(is_err(FsqQuantizer::new(vec![8; 4], 0).validate()));
    let bad_proj = FsqQuantizer {
        levels: vec![8; 4],
        proj: vec![0.0; 3],
        n: 4,
        d: 4,
    };
    assert!(is_err(bad_proj.validate()));
    assert!(is_err(FsqQuantizer::new(vec![8; 4], 4).quantize(&[0.0; 3])));

    let fsq = FsqQuantizer::identity(vec![8; 4]);
    let z = vec![0.1f32; 8];
    let codes = fsq.tokenize(&z).expect("tokenize");
    assert_eq!(codes.len(), 2);
    let back = fsq.decode(&codes).expect("decode");
    assert_eq!(back.len(), 8);
    assert!(back.iter().all(|v| v.is_finite()));
    assert!(is_err(fsq.decode(&[4_096])));
}

// ---------------------------------------------------------------------------
// MultiScaleTokenizer
// ---------------------------------------------------------------------------

fn tokenizer() -> MultiScaleTokenizer {
    let lengths = MultiScaleTokenizer::grid_lengths(16, 2).expect("lengths");
    MultiScaleTokenizer::new(FsqQuantizer::identity(vec![8; 4]), lengths).expect("tokenizer")
}

#[test]
fn grid_lengths_are_geometric() {
    assert_eq!(
        MultiScaleTokenizer::grid_lengths(16, 2).ok(),
        Some(vec![4, 16])
    );
    assert_eq!(
        MultiScaleTokenizer::grid_lengths(16, 3).ok(),
        Some(vec![1, 4, 16])
    );
    assert_eq!(
        MultiScaleTokenizer::grid_lengths(8, 2).ok(),
        Some(vec![2, 8])
    );
    assert!(is_err(MultiScaleTokenizer::grid_lengths(2, 4)));
    assert!(is_err(MultiScaleTokenizer::grid_lengths(0, 2)));
    assert!(is_err(MultiScaleTokenizer::grid_lengths(16, 0)));
}

#[test]
fn multiscale_new_validates_lengths() {
    let fsq = FsqQuantizer::identity(vec![8; 4]);
    assert!(is_err(MultiScaleTokenizer::new(fsq.clone(), vec![])));
    assert!(is_err(MultiScaleTokenizer::new(fsq.clone(), vec![16, 4])));
    assert!(is_err(MultiScaleTokenizer::new(fsq.clone(), vec![4, 4])));
    assert!(is_err(MultiScaleTokenizer::new(
        FsqQuantizer::new(vec![1; 4], 4),
        vec![4, 16]
    )));
}

#[test]
fn multiscale_down_up_are_inverse_for_exact_factors() {
    let tok = tokenizer();
    let x = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let coarse = tok.down(&x, 2, 1).expect("down");
    assert_eq!(coarse, vec![2.0, 3.0, 4.0, 5.0]);
    let up = tok.up(&coarse, 1, 2).expect("up");
    assert_eq!(up, vec![2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(is_err(tok.down(&x, 2, 4)));
    assert!(is_err(tok.up(&x, 2, 1)));
    assert!(is_err(tok.down(&x, 3, 1)));
}

#[test]
fn multiscale_encode_decode_shapes() {
    let tok = tokenizer();
    assert_eq!(tok.lengths, vec![4, 16]);
    assert_eq!(tok.finest(), 16);

    let z = vec![0.1f32; 16 * 4];
    let levels = tok.encode(&z).expect("encode");
    assert_eq!(levels.len(), 2);
    assert_eq!(levels[0].len(), 4 * 4);
    assert_eq!(levels[1].len(), 16 * 4);
    for level in &levels {
        assert!(level.iter().all(|code| *code < 8));
    }

    let back = tok.decode(&levels).expect("decode");
    assert_eq!(back.len(), z.len());
    assert!(back.iter().all(|v| v.is_finite()));
    assert!(is_err(tok.encode(&vec![0.0; 15])));
    assert!(is_err(tok.decode(&levels[..1])));
    assert!(is_err(tok.decode(&[levels[0].clone(), vec![0; 10]])));
}

#[test]
fn multiscale_reconstructs_representable_latent() {
    let tok = tokenizer();
    let z = vec![3.0f32; 16 * 4];
    let levels = tok.encode(&z).expect("encode");
    assert_eq!(levels[0], flat_codes(7, 4, 4));
    let back = tok.decode(&levels).expect("decode");
    assert_eq!(back.len(), z.len());
    for (got, want) in back.iter().zip(z.iter()) {
        assert!((got - want).abs() < 1e-5, "{got} != {want}");
    }
}

// ---------------------------------------------------------------------------
// LinearImageDecoder
// ---------------------------------------------------------------------------

#[test]
fn decoder_identity_head_roundtrip() {
    let tok = tokenizer();
    let dec = LinearImageDecoder::identity(tok, 2, 1).expect("decoder");
    assert_eq!((dec.patch(), dec.channels()), (2, 1));
    assert_eq!(dec.grid(8, 8).ok(), Some((4, 4)));
    assert!(is_err(dec.grid(8, 7)));
    assert!(is_err(dec.grid(0, 8)));

    let latent = vec![0.0f32; 16 * 4];
    let pixels = dec.decode_latents(&latent, 8, 8).expect("decode");
    assert_eq!(pixels.len(), 8 * 8);
    assert!(pixels.iter().all(|v| (0.0..=1.0).contains(v)));
    assert!(is_err(dec.decode_latents(&latent, 8, 4)));
    assert!(is_err(dec.decode_latents(&vec![0.0; 3], 8, 8)));
}

#[test]
fn decoder_places_patches_in_raster_order() {
    let tok = tokenizer();
    let dec = LinearImageDecoder::identity(tok, 2, 1).expect("decoder");
    // Level 0: 4 baris kasar bernilai 0,0,1,1; level 1: semua kode nol.
    let levels = vec![
        vec![4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5],
        flat_codes(4, 16, 4),
    ];
    let pixels = dec.decode(&levels, 8, 8).expect("decode");
    assert_eq!(pixels.len(), 64);
    for y in 0..8usize {
        for x in 0..8usize {
            let want = if y < 4 { 0.0 } else { 1.0 };
            assert!((pixels[y * 8 + x] - want).abs() < 1e-6, "piksel ({y},{x})");
        }
    }
    assert!(is_err(dec.decode(&levels[..1], 8, 8)));
    assert!(is_err(dec.decode(&levels, 16, 16)));
}

#[test]
fn decoder_head_replacement_checks_dim() {
    let tok = tokenizer();
    let dec = LinearImageDecoder::new(tok, 2, 1).expect("decoder");
    assert!(is_err(LinearImageDecoder::new(tokenizer(), 0, 1)));
    assert!(is_err(LinearImageDecoder::new(tokenizer(), 2, 0)));
    assert!(is_err(LinearImageDecoder::identity(tokenizer(), 3, 1)));
    assert_eq!(dec.head().in_dim, 4);
    assert_eq!(dec.head().out_dim, 4);

    let good = Linear::randomized(5, 4, 4).expect("linear");
    let updated = dec.clone().with_head(good).expect("head");
    assert_eq!(updated.head().w.len(), 16);
    let bad = Linear::randomized(5, 5, 4).expect("linear");
    assert!(is_err(dec.with_head(bad)));
}
