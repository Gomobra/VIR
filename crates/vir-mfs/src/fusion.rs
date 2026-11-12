//! Fusi Reciprocal Rank (RRF) dengan bobot, faktor usia, dan faktor akses.

/// Parameter fusi.
#[derive(Debug, Clone, Copy)]
pub struct FusionConfig {
    /// Konstanta RRF (Cormack dkk., SIGIR 2009); nilai umum 60.
    pub k: f64,
    /// Bobot daftar kata kunci (BM25).
    pub fts_weight: f64,
    /// Bobot daftar vektor.
    pub vector_weight: f64,
    /// Paruh usia dalam hari; skor dibagi dua setiap paruh ini.
    pub half_life_days: f64,
    /// Bobot log akses.
    pub access_weight: f64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            k: 60.0,
            fts_weight: 1.0,
            vector_weight: 1.0,
            half_life_days: 30.0,
            access_weight: 0.1,
        }
    }
}

/// Fusi RRF: `skor(d) = Σ_i w_i / (k + rank_i(d))`, rank 0-based.
///
/// Urutan seri diselesaikan dengan id menaik agar hasil deterministik.
pub fn rrf_fuse(lists: &[(&[i64], f64)], k: f64) -> Vec<(i64, f64)> {
    use std::collections::BTreeMap;
    let mut scores: BTreeMap<i64, f64> = BTreeMap::new();
    for (ids, weight) in lists {
        for (rank, &id) in ids.iter().enumerate() {
            let s = weight / (k + rank as f64 + 1.0);
            *scores.entry(id).or_insert(0.0) += s;
        }
    }
    let mut out: Vec<(i64, f64)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

/// Faktor usia: `0,5^(umur_hari / paruh_hari)`.
pub fn age_factor(age_days: f64, half_life_days: f64) -> f64 {
    if half_life_days <= 0.0 {
        return 1.0;
    }
    let age = if age_days.is_finite() && age_days > 0.0 {
        age_days
    } else {
        0.0
    };
    0.5_f64.powf(age / half_life_days)
}

/// Log akses: `ln(1 + jumlah_akses)`.
pub fn access_factor(count: u32) -> f64 {
    (1.0 + f64::from(count)).ln()
}

/// Pengali akses: `1 + bobot * ln(1 + jumlah_akses)`.
pub fn access_multiplier(count: u32, weight: f64) -> f64 {
    1.0 + weight * access_factor(count)
}

/// Skor akhir: dasar RRF dikali faktor usia dan akses.
pub fn rank_score(base: f64, age_days: f64, accesses: u32, cfg: &FusionConfig) -> f64 {
    base * age_factor(age_days, cfg.half_life_days) * access_multiplier(accesses, cfg.access_weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contoh_prd() {
        // PRD: rank 1 di FTS dan rank 3 di vektor = 1/61 + 1/63 = 0,03227.
        let fts = [10_i64, 20, 30];
        let vec = [20_i64, 40, 10];
        let fused = rrf_fuse(&[(&fts, 1.0), (&vec, 1.0)], 60.0);
        let doc10 = fused.iter().find(|(id, _)| *id == 10).map(|(_, s)| *s);
        let expected = 1.0 / 61.0 + 1.0 / 63.0;
        assert!((doc10.unwrap_or(0.0) - expected).abs() < 1e-12);
        assert_eq!(fused[0].0, 20);
    }

    #[test]
    fn bobot_mengubah_urutan() {
        let fts = [1_i64, 2];
        let vec = [2_i64, 1];
        let light = rrf_fuse(&[(&fts, 1.0), (&vec, 1.0)], 60.0);
        assert_eq!(light[0].0, 1);
        let heavy = rrf_fuse(&[(&fts, 1.0), (&vec, 4.0)], 60.0);
        assert_eq!(heavy[0].0, 2);
    }

    #[test]
    fn usia_menurun_naik_akses() {
        let cfg = FusionConfig::default();
        let fresh = rank_score(1.0, 0.0, 0, &cfg);
        let old = rank_score(1.0, 30.0, 0, &cfg);
        assert!(fresh > old);
        assert!((old - 0.5).abs() < 1e-9);
        assert!(rank_score(1.0, 0.0, 10, &cfg) > fresh);
        assert!((age_factor(0.0, 0.0)) == 1.0);
    }
}
