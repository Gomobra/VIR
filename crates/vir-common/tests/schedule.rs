//! F1.1 — jadwal ℓ_k: golden value, Σℓ, dan invarian (PRD F1.1).

use vir_common::config::LevelScheduleKind;
use vir_common::level_schedule;

const SEMUA: [LevelScheduleKind; 6] = [
    LevelScheduleKind::Linear,
    LevelScheduleKind::Quadratic,
    LevelScheduleKind::Sqrt,
    LevelScheduleKind::Log,
    LevelScheduleKind::Exp,
    LevelScheduleKind::Cosine,
];

fn total(levels: &[usize]) -> usize {
    levels.iter().sum()
}

#[test]
fn linear_k12_l128_golden() {
    let got = level_schedule(LevelScheduleKind::Linear, 128, 12);
    assert_eq!(got, vec![1, 13, 24, 36, 47, 59, 70, 82, 93, 105, 116, 128]);
    assert_eq!(total(&got), 774);
}

#[test]
fn xsum_k8_l48_golden() {
    let got = level_schedule(LevelScheduleKind::Linear, 48, 8);
    assert_eq!(got, vec![1, 8, 14, 21, 28, 35, 41, 48]);
    assert_eq!(total(&got), 196);
}

#[test]
fn wmt14_k11_l64_golden() {
    let got = level_schedule(LevelScheduleKind::Linear, 64, 11);
    assert_eq!(got, vec![1, 7, 14, 20, 26, 32, 39, 45, 51, 58, 64]);
    assert_eq!(total(&got), 357);
}

#[test]
fn k1_mengembalikan_l() {
    for kind in SEMUA {
        assert_eq!(level_schedule(kind, 128, 1), vec![128], "{kind:?}");
    }
}

#[test]
fn invarian_naik_ketat_berakhir_l() {
    let kasus = [
        (128usize, 12usize),
        (32, 6),
        (8, 3),
        (48, 8),
        (64, 11),
        (1, 1),
        (199, 15),
    ];
    for kind in SEMUA {
        for &(l, k) in &kasus {
            if k > l {
                continue;
            }
            let s = level_schedule(kind, l, k);
            assert_eq!(s.len(), k, "{kind:?} L={l} K={k} panjang");
            assert_eq!(
                *s.last().expect("non-kosong"),
                l,
                "{kind:?} L={l} K={k} akhir"
            );
            assert!(
                s.windows(2).all(|w| w[0] < w[1]),
                "{kind:?} L={l} K={k} harus naik ketat: {s:?}"
            );
        }
    }
}
