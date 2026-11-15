//! Tes nyata loss berbobot level (PRD F1.7 langkah 1, rumus R6) dan estimator
//! turunannya. Fungsi bobot murni (`f32`/`usize`), jadi diuji langsung.

use vir_train::{TrainConfig, code_weight, grpo_advantage, level_weight, pass_at_k};

const LEVELS: [usize; 4] = [1, 13, 24, 36];
const SUM: f32 = 74.0;

#[test]
fn level_weights_sum_to_one_for_every_alpha() {
    for alpha in [0.0f32, 0.25, 0.5, 1.0, 1.5, 2.0] {
        let w = level_weight(&LEVELS, alpha);
        let sum: f32 = w.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "alpha {alpha}: jumlah bobot {sum} != 1"
        );
        assert!(w.iter().all(|v| *v > 0.0), "alpha {alpha}: ada bobot <= 0");
    }
}

#[test]
fn level_weights_alpha_zero_are_proportional_to_length() {
    let w = level_weight(&LEVELS, 0.0);
    for (k, &l) in LEVELS.iter().enumerate() {
        assert!(
            (w[k] - l as f32 / SUM).abs() < 1e-6,
            "alpha=0: w[{k}] = {} != {}/{}",
            w[k],
            l,
            SUM
        );
    }
    for k in 1..LEVELS.len() {
        assert!(w[k] > w[k - 1], "alpha=0 harus naik terhadap level");
    }
}

#[test]
fn level_weights_alpha_one_are_uniform() {
    let w = level_weight(&LEVELS, 1.0);
    for v in &w {
        assert!((v - 0.25).abs() < 1e-6, "alpha=1: bobot {v} != 1/K");
    }
}

#[test]
fn level_weights_alpha_above_one_favour_coarse_levels() {
    let w0 = level_weight(&LEVELS, 0.0);
    let w15 = level_weight(&LEVELS, 1.5);
    assert!(
        w15[0] > w0[0],
        "alpha=1,5 harus menaikkan porsi level kasar ({} vs {})",
        w15[0],
        w0[0]
    );
    for k in 1..LEVELS.len() {
        assert!(
            w15[k] < w15[k - 1],
            "alpha=1,5 harus menurun terhadap level"
        );
    }
}

#[test]
fn code_weight_is_per_code_form() {
    let cw = code_weight(&LEVELS, 0.0);
    let sum: f32 = cw.iter().sum();
    assert!(
        (sum - LEVELS.len() as f32 / SUM).abs() < 1e-6,
        "jumlah bobot per kode {sum} != K/sum(levels)"
    );
    for v in &cw {
        assert!((v - 1.0 / SUM).abs() < 1e-6, "alpha=0: bobot per kode {v}");
    }
}

#[test]
fn train_config_weights_match_level_weight() {
    let config = TrainConfig {
        levels: LEVELS.to_vec(),
        alpha: 0.5,
        ..TrainConfig::default()
    };
    let a = config.weights();
    let b = level_weight(&LEVELS, 0.5);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-6, "bobot config {x} != {y}");
    }
}

#[test]
fn pass_at_k_matches_binomial_estimator() {
    assert!((pass_at_k(4, 1, 2) - 0.5).abs() < 1e-9);
    assert!((pass_at_k(4, 4, 1) - 1.0).abs() < 1e-9);
    assert!((pass_at_k(4, 0, 1) - 0.0).abs() < 1e-9);
    assert!((pass_at_k(0, 0, 1) - 0.0).abs() < 1e-9);
    assert!((pass_at_k(3, 1, 3) - 1.0).abs() < 1e-9);
}

#[test]
fn grpo_advantage_is_zero_mean_unit_std() {
    let adv = grpo_advantage(&[1.0, 1.0, 0.0, 0.0], 1e-8);
    assert_eq!(adv.len(), 4);
    let mean = adv.iter().sum::<f32>() / adv.len() as f32;
    assert!(mean.abs() < 1e-5, "mean keuntungan {mean} != 0");
    let var = adv.iter().map(|a| a * a).sum::<f32>() / adv.len() as f32;
    assert!((var - 1.0).abs() < 1e-4, "varians {var} != 1");
    assert!(adv[0] > 0.0 && adv[2] < 0.0, "tanda keuntungan salah");
}
