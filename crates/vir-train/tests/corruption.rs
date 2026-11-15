//! Tes nyata korupsi kode masukan per level (PRD F1.7 langkah 2, NCM §3.4):
//! laju per level, jumlah terkorupsi tepat, dan determinisme dengan seed.

use vir_train::{Corruption, RateShape, level_rates};

const LEVELS: [usize; 5] = [8, 16, 32, 64, 128];

#[test]
fn linear_decay_rates_average_to_target() {
    let rates = level_rates(&LEVELS, 0.5, RateShape::LinearDecay);
    assert_eq!(rates.len(), LEVELS.len());
    let mean = rates.iter().sum::<f32>() / rates.len() as f32;
    assert!((mean - 0.5).abs() < 1e-6, "rata-rata laju {mean} != 0,5");
    assert!(
        (rates[0] - 1.0).abs() < 1e-6,
        "laju level kasar {}",
        rates[0]
    );
    assert!(
        rates[4] >= 0.0 && rates[4] <= 1e-6,
        "laju level halus {}",
        rates[4]
    );
    for k in 1..rates.len() {
        assert!(rates[k] <= rates[k - 1], "laju harus turun terhadap level");
    }
}

#[test]
fn flat_and_single_level_shapes() {
    let flat = level_rates(&LEVELS, 0.25, RateShape::Flat);
    assert!(flat.iter().all(|r| (r - 0.25).abs() < 1e-6));
    let single = level_rates(&[8], 0.75, RateShape::LinearDecay);
    assert_eq!(single, vec![0.75]);
    assert!(level_rates(&[], 0.5, RateShape::Flat).is_empty());
}

#[test]
fn next_batch_corrupts_exactly_the_requested_count() {
    let mut c = Corruption::new(3, &LEVELS, 0.5, RateShape::LinearDecay);
    for (k, &len) in LEVELS.iter().enumerate() {
        let batch = c.next_batch(k, 4, len, 16);
        let positions = 4 * len;
        assert_eq!(batch.positions, positions);
        assert_eq!(batch.mask.len(), positions);
        assert_eq!(batch.codes.len(), positions);
        let expected = (c.rates()[k] * positions as f32).round() as usize;
        assert_eq!(batch.corrupted, expected, "level {k}");
        assert_eq!(batch.mask.iter().filter(|m| **m).count(), batch.corrupted);
        assert!((batch.measured_rate() - batch.corrupted as f32 / positions as f32).abs() < 1e-6);
        for (m, code) in batch.mask.iter().zip(batch.codes.iter()) {
            if *m {
                assert!(
                    *code >= 0 && (*code as usize) < 16,
                    "kode {code} di luar codebook"
                );
            } else {
                assert_eq!(*code, 0, "posisi bersih harus berkode 0");
            }
        }
    }
}

#[test]
fn measured_rate_across_levels_matches_target_mean() {
    let mut c = Corruption::new(3, &LEVELS, 0.5, RateShape::LinearDecay);
    for k in 0..LEVELS.len() {
        c.next_batch(k, 4, LEVELS[k], 16);
    }
    let mean = c.mean_measured();
    assert!(
        (mean - 0.5).abs() < 1e-6,
        "laju rata-rata sebenarnya {mean} != 0,5"
    );
}

#[test]
fn same_seed_is_deterministic_and_different_seed_is_not() {
    // Flat 0,5: setiap level punya posisi terkorupsi, jadi seed berbeda
    // pasti menghasilkan batch berbeda (LinearDecay berakhir di laju 0).
    let mut a = Corruption::new(9, &LEVELS, 0.5, RateShape::Flat);
    let mut b = Corruption::new(9, &LEVELS, 0.5, RateShape::Flat);
    let mut other = Corruption::new(10, &LEVELS, 0.5, RateShape::Flat);
    for k in 0..LEVELS.len() {
        let ba = a.next_batch(k, 4, LEVELS[k], 16);
        let bb = b.next_batch(k, 4, LEVELS[k], 16);
        assert_eq!(ba, bb, "seed sama harus menghasilkan batch identik");
        let bo = other.next_batch(k, 4, LEVELS[k], 16);
        assert_ne!(ba, bo, "seed berbeda seharusnya menghasilkan batch berbeda");
    }
    assert_eq!(a.rng().state(), b.rng().state());
}

#[test]
fn zero_rate_never_corrupts() {
    let mut c = Corruption::none(5, &LEVELS);
    for k in 0..LEVELS.len() {
        let batch = c.next_batch(k, 4, LEVELS[k], 16);
        assert_eq!(batch.corrupted, 0);
        assert!(batch.mask.iter().all(|m| !*m));
        assert!(c.rates()[k] == 0.0);
    }
}
