//! F1.1 — validasi `ModelConfig` (PRD F1.1): tolak jadwal salah dan hidden/head.

use vir_common::config::{LevelScheduleKind, ModelConfig};

fn valid() -> ModelConfig {
    ModelConfig::default()
}

#[test]
fn default_valid() {
    valid().validate().expect("default harus valid");
}

#[test]
fn tolak_num_levels_melebihi_target() {
    let mut c = valid();
    c.target_length = 4;
    c.num_levels = 8;
    assert!(c.validate().is_err(), "K > L harus ditolak");
}

#[test]
fn tolak_hidden_tidak_terbagi_heads() {
    let mut c = valid();
    c.hidden_dim = 100;
    c.num_heads = 12;
    assert!(c.validate().is_err(), "hidden % heads != 0 harus ditolak");
}

#[test]
fn tolak_head_dim_bukan_64_saat_strict() {
    let mut c = valid();
    c.hidden_dim = 768;
    c.num_heads = 12; // head_dim = 64 → ok
    c.validate().expect("64 ok");
    c.num_heads = 16; // head_dim = 48
    c.strict_paper = true;
    assert!(c.validate().is_err(), "strict_paper butuh head_dim 64");
    c.strict_paper = false;
    assert!(c.validate().is_ok(), "non-strict boleh head_dim != 64");
}

#[test]
fn tolak_ema_decay_di_luar_rentang() {
    let mut c = valid();
    c.ema_decay = 1.0;
    assert!(c.validate().is_err(), "ema_decay = 1 harus ditolak");
    c.ema_decay = 0.0;
    assert!(c.validate().is_err(), "ema_decay = 0 harus ditolak");
    c.ema_decay = 0.99;
    assert!(c.validate().is_ok());
}

#[test]
fn tolak_nested_dropout_di_luar_rentang() {
    let mut c = valid();
    c.nested_dropout_prob = 1.5;
    assert!(c.validate().is_err(), "> 1 harus ditolak");
    c.nested_dropout_prob = -0.1;
    assert!(c.validate().is_err(), "< 0 harus ditolak");
}

#[test]
fn levels_turunkan_jadwal() {
    let c = ModelConfig::default();
    let s = c.levels();
    assert_eq!(s.len(), c.num_levels);
    assert_eq!(*s.last().expect("non-kosong"), c.target_length);
    // Default memakai Linear (PRD F1.1 golden).
    assert_eq!(s[1], 13);
    let _ = LevelScheduleKind::Linear;
}
