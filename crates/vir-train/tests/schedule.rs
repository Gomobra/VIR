//! Tes nyata jadwal LR kosinus dan warmup–stabil–decay (PRD F1.7 langkah 3):
//! batas warmup, puncak, peluruhan, dan lantai.

use vir_train::{LrKind, LrSchedule};

const PEAK: f64 = 3e-4;
const FLOOR: f64 = 3e-5;
const WARMUP: u64 = 10;
const TOTAL: u64 = 100;

#[test]
fn cosine_warmup_is_linear_to_peak() {
    let s = LrSchedule::cosine(PEAK, FLOOR, WARMUP, TOTAL);
    assert!((s.lr(0) - PEAK / WARMUP as f64).abs() < 1e-12, "lr(0)");
    for step in 0..WARMUP {
        assert!(s.lr(step) <= PEAK + 1e-12, "lr({step}) melebihi puncak");
    }
    assert!(
        (s.lr(WARMUP - 1) - PEAK).abs() < 1e-12,
        "puncak di akhir warmup"
    );
    assert_eq!(s.kind, LrKind::Cosine);
}

#[test]
fn cosine_decays_to_floor_and_never_exceeds_peak() {
    let s = LrSchedule::cosine(PEAK, FLOOR, WARMUP, TOTAL);
    assert!((s.lr(TOTAL) - FLOOR).abs() < 1e-12, "lantai tidak tercapai");
    assert!(
        s.lr(TOTAL + 1_000) <= FLOOR + 1e-12,
        "di luar rentang harus dijepit"
    );
    for step in WARMUP..TOTAL {
        assert!(
            s.lr(step) >= s.lr(step + 1) - 1e-15,
            "kosinus tidak monoton turun"
        );
    }
    assert!(s.lr(WARMUP) <= PEAK + 1e-12);
}

#[test]
fn wsd_holds_peak_then_decays_linearly_to_floor() {
    let s = LrSchedule::wsd(PEAK, FLOOR, WARMUP, TOTAL, 0.2);
    assert_eq!(s.kind, LrKind::Wsd);
    assert!((s.lr(50) - PEAK).abs() < 1e-12, "fase stabil tidak rata");
    assert!((s.lr(82) - PEAK).abs() < 1e-12, "batas akhir fase stabil");
    assert!(s.lr(90) > s.lr(100), "fase decay tidak turun");
    assert!(
        (s.lr(TOTAL) - FLOOR).abs() < 1e-12,
        "lantai WSD tidak tercapai"
    );
    assert!(s.lr(100) <= PEAK + 1e-12);
}

#[test]
fn wsd_decay_fraction_is_clamped() {
    let s = LrSchedule::wsd(PEAK, FLOOR, 0, TOTAL, 0.0);
    assert!((s.decay_fraction - 0.01).abs() < 1e-12);
    let s = LrSchedule::wsd(PEAK, FLOOR, 0, TOTAL, 5.0);
    assert!((s.decay_fraction - 1.0).abs() < 1e-12);
}

#[test]
fn scaled_schedule_multiplies_lr() {
    let s = LrSchedule::cosine(PEAK, FLOOR, WARMUP, TOTAL);
    for step in [0u64, 5, 50, 100] {
        assert!((s.lr_scaled(step, 0.25) - s.lr(step) * 0.25).abs() < 1e-15);
        assert!((s.lr_f32(step) as f64 - s.lr(step)).abs() < 1e-9);
    }
}

#[test]
fn validate_rejects_invalid_schedules() {
    assert!(
        LrSchedule::cosine(0.0, FLOOR, WARMUP, TOTAL)
            .validate()
            .is_err()
    );
    assert!(
        LrSchedule::cosine(PEAK, -1.0, WARMUP, TOTAL)
            .validate()
            .is_err()
    );
    assert!(
        LrSchedule::cosine(FLOOR, PEAK, WARMUP, TOTAL)
            .validate()
            .is_err()
    );
    assert!(
        LrSchedule::cosine(PEAK, FLOOR, WARMUP, 0)
            .validate()
            .is_err()
    );
    assert!(
        LrSchedule::cosine(PEAK, FLOOR, TOTAL + 1, TOTAL)
            .validate()
            .is_err()
    );
    assert!(
        LrSchedule::cosine(PEAK, FLOOR, WARMUP, TOTAL)
            .validate()
            .is_ok()
    );
}
