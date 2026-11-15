//! F1.1 — proptest jadwal ℓ_k untuk (L,K) acak (PRD F1.1).

use proptest::prelude::*;
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn jadwal_selalu_naik_ketat_berakhir_l(
        l in 1usize..=256,
        k in 1usize..=16usize,
    ) {
        // Hanya uji k <= l (K > L ditolak di validasi Config).
        let Some(k) = (k <= l).then_some(k) else { return Ok(()) };
        for kind in SEMUA {
            let s = level_schedule(kind, l, k);
            prop_assert_eq!(s.len(), k);
            prop_assert_eq!(*s.last().expect("non-kosong"), l);
            prop_assert!(s.windows(2).all(|w| w[0] < w[1]),
                "{kind:?} L={l} K={k} tidak naik ketat: {s:?}");
        }
    }

    #[test]
    fn jumlah_posisi_dalam_kisaran_wajar(
        l in 2usize..=512,
        k in 2usize..=12usize,
    ) {
        let Some(k) = (k <= l).then_some(k) else { return Ok(()) };
        let s = level_schedule(LevelScheduleKind::Linear, l, k);
        let total: usize = s.iter().sum();
        // Σℓ antara L (semua sama) dan K*L (maksimum).
        prop_assert!(total >= l);
        prop_assert!(total <= k * l);
    }
}
