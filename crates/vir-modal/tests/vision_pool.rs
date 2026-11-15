//! Tes nyata F6.1/F6.3 untuk encoder gambar (`vision.rs`), operator antar level
//! (`pool.rs`), dan `Prefix` (`encoder.rs`). Dijalankan job CI `vir-modal`.

use vir_common::VirError;
use vir_modal::asset::ModalAsset;
use vir_modal::connector::{Connector, Linear};
use vir_modal::encoder::{EncoderSpec, ModalEncoder, Prefix};
use vir_modal::kind::ModalKind;
use vir_modal::pool::{add_assign, avg_pool_rows, nearest_up_rows, sub_assign};
use vir_modal::vision::ImageEncoder;

fn is_err<T>(r: Result<T, VirError>) -> bool {
    r.is_err()
}

// ---------------------------------------------------------------------------
// vision.rs
// ---------------------------------------------------------------------------

#[test]
fn vision_patch_tokens_match_prd() {
    let enc = ImageEncoder::new(EncoderSpec::siglip(16), 3, 8).expect("encoder");
    assert_eq!((enc.patch(), enc.channels()), (16, 3));
    assert_eq!(enc.connector().d_in(), 16 * 16 * 3);
    assert_eq!(enc.patch_grid(224, 224).ok(), Some((14, 14)));
    assert_eq!(enc.patch_tokens(224, 224).ok(), Some(196));
    assert_eq!(enc.patch_tokens(384, 384).ok(), Some(576));
    assert_eq!(enc.raw_len_of(224, 224).ok(), Some(150_528));
    assert!(is_err(enc.patch_grid(225, 224)));
    assert!(is_err(enc.patch_grid(0, 224)));
    assert!(is_err(enc.patch_grid(224, 0)));
    assert!(is_err(ImageEncoder::new(EncoderSpec::siglip(16), 0, 8)));
    assert!(is_err(ImageEncoder::new(EncoderSpec::whisper(), 3, 8)));
}

#[test]
fn vision_patchify_order_is_row_major() {
    let enc = ImageEncoder::new(EncoderSpec::siglip(2), 1, 4).expect("encoder");
    let pixels: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let patches = enc.patchify(&pixels, 4, 4).expect("patchify");
    assert_eq!(patches.len(), 4 * 4);
    assert_eq!(patches[0..4].to_vec(), vec![0.0, 1.0, 4.0, 5.0]);
    assert_eq!(patches[4..8].to_vec(), vec![2.0, 3.0, 6.0, 7.0]);
    assert_eq!(patches[8..12].to_vec(), vec![8.0, 9.0, 12.0, 13.0]);
    assert_eq!(patches[12..16].to_vec(), vec![10.0, 11.0, 14.0, 15.0]);
    assert!(is_err(enc.patchify(&pixels, 4, 5)));
    assert!(is_err(enc.patchify(&vec![0.0; 15], 4, 4)));
}

#[test]
fn vision_encode_prefix_and_trait() {
    let enc = ImageEncoder::randomized(13, EncoderSpec::siglip(4), 1, 4).expect("encoder");
    let pixels = vec![0.5f32; 64];
    let prefix = enc.encode_image(&pixels, 8, 8).expect("encode");
    assert_eq!(prefix.tokens, 4);
    assert_eq!(prefix.d_model, 4);
    assert_eq!(prefix.elements(), 16);
    assert!(prefix.rows.iter().all(|v| v.is_finite()));

    let asset = ModalAsset::image("i.png", 8, 8);
    assert!(enc.is_frozen());
    assert_eq!(enc.token_budget(&asset).ok(), Some(4));
    assert_eq!(enc.raw_len(&asset).ok(), Some(64));
    assert_eq!(enc.encode(&asset, &pixels).expect("encode"), prefix);
    assert!(is_err(enc.encode(&asset, &vec![0.0; 10])));
    assert!(is_err(
        enc.encode(&ModalAsset::image("j.png", 8, 9), &pixels)
    ));
    assert!(is_err(
        enc.token_budget(&ModalAsset::video("v.mp4", 8, 8, 1_000, 25))
    ));
    assert!(is_err(
        enc.raw_len(&ModalAsset::audio("a.wav", 10, 16_000, 1))
    ));
}

#[test]
fn vision_connector_replacement_checks_dim() {
    let enc = ImageEncoder::new(EncoderSpec::siglip(4), 1, 4).expect("encoder");
    let good = Connector::randomized(3, 16, 4, 4).expect("connector");
    let updated = enc.clone().with_connector(good).expect("with");
    assert_eq!(updated.connector().d_in(), 16);
    let bad = Connector::randomized(3, 15, 4, 4).expect("connector");
    assert!(is_err(enc.with_connector(bad)));

    // Bobot acak deterministik: seed sama → prefix sama.
    let a = ImageEncoder::randomized(21, EncoderSpec::siglip(4), 1, 4).expect("encoder");
    let b = ImageEncoder::randomized(21, EncoderSpec::siglip(4), 1, 4).expect("encoder");
    let x = vec![0.25f32; 64];
    assert_eq!(
        a.encode_image(&x, 8, 8).expect("a").rows,
        b.encode_image(&x, 8, 8).expect("b").rows
    );
    let head = Linear::randomized(3, 16, 4).expect("linear");
    assert_eq!(head.out_dim, 4);
}

#[test]
fn prefix_rows_are_indexable() {
    let prefix = Prefix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    assert_eq!(prefix.len(), 2);
    assert_eq!(prefix.elements(), 4);
    assert!(!prefix.is_empty());
    assert_eq!(prefix.row(0), Some(&[1.0f32, 2.0][..]));
    assert_eq!(prefix.row(1), Some(&[3.0f32, 4.0][..]));
    assert_eq!(prefix.row(2), None);
    assert!(Prefix::new(vec![], 0, 2).is_empty());
}

// ---------------------------------------------------------------------------
// pool.rs
// ---------------------------------------------------------------------------

#[test]
fn avg_pool_rows_averages_blocks() {
    let x = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    assert_eq!(
        avg_pool_rows(&x, 4, 2, 2).expect("pool"),
        vec![1.0, 2.0, 5.0, 6.0]
    );
    assert_eq!(avg_pool_rows(&x, 4, 4, 2).expect("pool"), x.to_vec());
    assert_eq!(avg_pool_rows(&x, 4, 1, 2).expect("pool"), vec![3.0, 4.0]);
    assert!(is_err(avg_pool_rows(&x, 4, 8, 2)));
    assert!(is_err(avg_pool_rows(&x, 4, 0, 2)));
    assert!(is_err(avg_pool_rows(&x, 4, 2, 0)));
    assert!(is_err(avg_pool_rows(&vec![0.0; 3], 4, 2, 2)));
    assert!(is_err(avg_pool_rows(&[], 0, 1, 2)));
}

#[test]
fn nearest_up_repeats_rows() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(
        nearest_up_rows(&x, 2, 4, 2).expect("up"),
        vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
    );
    assert_eq!(nearest_up_rows(&x, 2, 2, 2).expect("up"), x.to_vec());
    assert!(is_err(nearest_up_rows(&x, 4, 2, 2)));
    assert!(is_err(nearest_up_rows(&x, 2, 4, 3)));
    assert!(is_err(nearest_up_rows(&x, 2, 0, 2)));
}

#[test]
fn add_and_sub_assign_match_lengths() {
    let mut a = vec![1.0f32, 2.0, 3.0];
    add_assign(&mut a, &[0.5, 0.5, 0.5]).expect("add");
    assert_eq!(a, vec![1.5, 2.5, 3.5]);
    sub_assign(&mut a, &[1.5, 2.5, 3.5]).expect("sub");
    assert_eq!(a, vec![0.0; 3]);
    assert!(is_err(add_assign(&mut a, &[1.0])));
    assert!(is_err(sub_assign(&mut a, &[1.0])));
    assert!(is_err(sub_assign(&mut a, &[])));
}

#[test]
fn vision_encoder_is_a_modal_encoder() {
    let enc = ImageEncoder::new(EncoderSpec::siglip(16), 3, 8).expect("encoder");
    assert_eq!(enc.spec().kind, ModalKind::Image);
    assert_eq!(
        enc.name(),
        "siglip/webli-en-patch16 patch=16 stride=1 frozen=true"
    );
    assert!(enc.is_frozen());
    // Trait objek: `ModalEncoder` harus aman dipakai sebagai `&dyn`.
    let dyn_enc: &dyn ModalEncoder<Output = Prefix> = &enc;
    let asset = ModalAsset::image("i.png", 224, 224);
    assert_eq!(dyn_enc.token_budget(&asset).ok(), Some(196));
    assert_eq!(dyn_enc.raw_len(&asset).ok(), Some(150_528));
}
