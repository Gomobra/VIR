//! Metrik utilisasi/perplexity kode (PRD F1.7 task "log metrik JSONL
//! (loss total, loss per level, utilisasi)").
//!
//! Definisi sama dengan `vir-vq` (`QuantizerStats`): utilisasi = entri terpakai
//! / |V|, perplexity = exp(H) dari histogram kode. Di sini dihitung dari
//! histogram host sehingga bisa masuk checkpoint/log tanpa tensor.

/// Histogram kode dari indeks level (nilai di luar `0..codes` diabaikan).
pub fn histogram(indices: &[i32], codes: usize) -> Vec<u64> {
    let mut counts = vec![0u64; codes];
    for &i in indices {
        if i >= 0 && (i as usize) < codes {
            counts[i as usize] += 1;
        }
    }
    counts
}

/// Utilisasi: entri yang terpakai minimal sekali dibagi ukuran codebook.
pub fn utilization(counts: &[u64]) -> f32 {
    if counts.is_empty() {
        return 0.0;
    }
    let used = counts.iter().filter(|c| **c > 0).count();
    used as f32 / counts.len() as f32
}

/// Perplexity `exp(H)` dengan `H` entropi distribusi kode (0 bila kosong).
pub fn perplexity(counts: &[u64]) -> f32 {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let mut h = 0.0f64;
    for &c in counts {
        if c > 0 {
            let p = c as f64 / total as f64;
            h -= p * p.ln();
        }
    }
    h.exp() as f32
}

/// Statistik satu level: dipakai per baris JSONL.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelStat {
    /// Indeks level (0-based).
    pub level: usize,
    /// Ukuran codebook |V_k|.
    pub codes: usize,
    /// Entri terpakai.
    pub used: usize,
    /// `used / codes`.
    pub utilization: f32,
    /// `exp(H)`.
    pub perplexity: f32,
}

impl LevelStat {
    /// Statistik dari indeks satu level.
    pub fn from_indices(level: usize, indices: &[i32], codes: usize) -> Self {
        let counts = histogram(indices, codes);
        Self {
            level,
            codes,
            used: counts.iter().filter(|c| **c > 0).count(),
            utilization: utilization(&counts),
            perplexity: perplexity(&counts),
        }
    }

    /// Utilisasi dianggap sehat bila ≥ 50% (NCM Tabel 1: 61,9–90,1%).
    pub fn is_healthy(&self) -> bool {
        self.utilization >= 0.5
    }

    /// Ringkasan JSON satu baris (untuk log/artefak).
    pub fn to_json(&self) -> String {
        format!(
            "{{\"level\":{},\"codes\":{},\"used\":{},\"utilization\":{:.4},\"perplexity\":{:.2}}}",
            self.level, self.codes, self.used, self.utilization, self.perplexity
        )
    }
}

/// Vektor statistik per level dari histogram per level.
pub fn level_stats(histograms: &[Vec<u64>]) -> Vec<LevelStat> {
    histograms
        .iter()
        .enumerate()
        .map(|(k, counts)| LevelStat {
            level: k,
            codes: counts.len(),
            used: counts.iter().filter(|c| **c > 0).count(),
            utilization: utilization(counts),
            perplexity: perplexity(counts),
        })
        .collect()
}

/// Uji asap metrik: distribusi merata pada 4 kode → utilisasi 1, perplexity 4.
pub fn metrics_smoke() -> Result<(), vir_common::VirError> {
    use vir_common::VirError;
    let counts = vec![10u64; 4];
    if (utilization(&counts) - 1.0).abs() > 1e-6 {
        return Err(VirError::Internal("utilisasi salah".into()));
    }
    if (perplexity(&counts) - 4.0).abs() > 1e-4 {
        return Err(VirError::Internal("perplexity salah".into()));
    }
    let stat = LevelStat::from_indices(0, &[0, 1, 2, 3, 0], 8);
    if (stat.utilization - 0.5).abs() > 1e-6 || stat.perplexity <= 1.9 {
        return Err(VirError::Internal("statistik level salah".into()));
    }
    let dead = vec![0u64, 0, 0, 5];
    if utilization(&dead) != 0.25 {
        return Err(VirError::Internal("utilisasi kode mati salah".into()));
    }
    Ok(())
}