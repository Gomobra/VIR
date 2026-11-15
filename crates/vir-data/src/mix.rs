//! Campuran bahasa berbobot (PRD F1.3 langkah 3).
//!
//! Tiap sumber punya bobot bahasa; pemilihan sumber dilakukan per blok dengan
//! RNG deterministik dari seed. Bila satu sumber habis, bobotnya dibuang dan
//! sisa sumber dinormalkan, sehingga proporsi bahasa tetap mendekati bobot.

use vir_common::VirError;

use crate::stream::SplitMix64;

/// Satu sumber blok dengan bobot bahasa.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceMix {
    pub name: String,
    pub weight: f32,
    pub blocks: Vec<Vec<u32>>,
}

impl SourceMix {
    /// Sumber dengan nama, bobot, dan blok.
    pub fn new(name: impl Into<String>, weight: f32, blocks: Vec<Vec<u32>>) -> Self {
        Self {
            name: name.into(),
            weight,
            blocks,
        }
    }
}

/// Konfigurasi pencampuran.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixConfig {
    pub seed: u64,
    pub total_blocks: usize,
}

impl Default for MixConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            total_blocks: 1024,
        }
    }
}

/// Periksa bobot: tidak negatif, bukan NaN, dan jumlahnya positif.
pub fn validate_weights(weights: &[f32]) -> Result<(), VirError> {
    if weights.is_empty() {
        return Err(VirError::Config("bobot campuran kosong".into()));
    }
    let mut total = 0.0f64;
    for weight in weights {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(VirError::Config(format!(
                "bobot campuran tidak sah: {weight}"
            )));
        }
        total += f64::from(*weight);
    }
    if total <= 0.0 {
        return Err(VirError::Config(
            "jumlah bobot campuran harus positif".into(),
        ));
    }
    Ok(())
}

/// Pilih indeks sumber menurut bobot; `None` bila daftar kosong atau total nol.
pub fn pick_source(weights: &[f32], rng: &mut SplitMix64) -> Option<usize> {
    if weights.is_empty() {
        return None;
    }
    let total: f64 = weights.iter().map(|w| f64::from(*w)).sum();
    if total <= 0.0 {
        return None;
    }
    let target = rng.next_f64() * total;
    let mut acc = 0.0f64;
    for (index, weight) in weights.iter().enumerate() {
        acc += f64::from(*weight);
        if target < acc {
            return Some(index);
        }
    }
    Some(weights.len() - 1)
}

/// Cacah blok per sumber.
pub fn block_counts(sources: &[SourceMix]) -> Vec<usize> {
    sources.iter().map(|s| s.blocks.len()).collect()
}

/// Cacah total blok.
pub fn total_blocks(sources: &[SourceMix]) -> usize {
    sources.iter().map(|s| s.blocks.len()).sum()
}

/// Campur blok dari beberapa sumber menurut bobot.
///
/// `cfg.total_blocks` adalah batas maksimum; bila semua blok habis lebih dulu,
/// hasilnya berhenti di jumlah blok yang tersedia.
pub fn mix_blocks(sources: &[SourceMix], cfg: &MixConfig) -> Result<Vec<Vec<u32>>, VirError> {
    if sources.is_empty() {
        return Err(VirError::Config("tidak ada sumber untuk dicampur".into()));
    }
    let weights: Vec<f32> = sources.iter().map(|s| s.weight).collect();
    validate_weights(&weights)?;
    for source in sources {
        if source.blocks.is_empty() {
            let name = source.name.as_str();
            return Err(VirError::Config(format!(
                "sumber '{name}' tidak punya blok"
            )));
        }
    }
    if cfg.total_blocks == 0 {
        return Err(VirError::Config("total_blocks harus positif".into()));
    }
    let mut rng = SplitMix64::new(cfg.seed);
    let mut cursors = vec![0usize; sources.len()];
    let mut live: Vec<usize> = (0..sources.len()).collect();
    let mut out = Vec::with_capacity(cfg.total_blocks);
    while out.len() < cfg.total_blocks {
        let weights: Vec<f32> = live.iter().map(|i| sources[*i].weight).collect();
        let Some(pick) = pick_source(&weights, &mut rng) else {
            break;
        };
        let index = live[pick];
        out.push(sources[index].blocks[cursors[index]].clone());
        cursors[index] += 1;
        if cursors[index] >= sources[index].blocks.len() {
            live.retain(|i| *i != index);
            if live.is_empty() {
                break;
            }
        }
    }
    Ok(out)
}

/// Proporsi blok per sumber (untuk log dan TUI).
pub fn proportions(sources: &[SourceMix]) -> Vec<f32> {
    let total = total_blocks(sources).max(1) as f32;
    sources
        .iter()
        .map(|s| s.blocks.len() as f32 / total)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sources() -> Vec<SourceMix> {
        vec![
            SourceMix::new("id", 0.7, vec![vec![1], vec![2], vec![3]]),
            SourceMix::new("ru", 0.3, vec![vec![9], vec![8]]),
        ]
    }

    #[test]
    fn bobot_tidak_sah_ditolak() {
        assert!(validate_weights(&[]).is_err());
        assert!(validate_weights(&[0.0, 0.0]).is_err());
        assert!(validate_weights(&[-1.0]).is_err());
        assert!(validate_weights(&[f32::NAN]).is_err());
        assert!(validate_weights(&[0.5, 0.5]).is_ok());
    }

    #[test]
    fn pilih_sumber_mengikuti_bobot() {
        let mut rng = SplitMix64::new(1);
        let mut counts = [0usize; 2];
        for _ in 0..2000 {
            let index = pick_source(&[0.9, 0.1], &mut rng).unwrap_or(0);
            counts[index] += 1;
        }
        assert!(counts[0] > counts[1]);
        assert!(counts[1] > 0);
        assert_eq!(pick_source(&[], &mut rng), None);
        assert_eq!(pick_source(&[0.0, 0.0], &mut rng), None);
    }

    #[test]
    fn campuran_deterministik_dan_terbatas() {
        let cfg = MixConfig {
            seed: 5,
            total_blocks: 4,
        };
        let first = mix_blocks(&sources(), &cfg).expect("campur");
        let second = mix_blocks(&sources(), &cfg).expect("campur");
        assert_eq!(first, second);
        assert_eq!(first.len(), 4);
        let all: Vec<u32> = first.iter().flatten().copied().collect();
        assert_eq!(all.len(), 4);
        // Semua blok habis sebelum batas: hasil berhenti di 5 blok.
        let capped = MixConfig {
            seed: 5,
            total_blocks: 50,
        };
        assert_eq!(mix_blocks(&sources(), &capped).expect("campur").len(), 5);
        let props = proportions(&sources());
        assert!((props[0] - 0.6).abs() < 0.01);
    }

    #[test]
    fn sumber_kosong_ditolak() {
        let cfg = MixConfig::default();
        assert!(mix_blocks(&[], &cfg).is_err());
        let broken = vec![SourceMix::new("id", 1.0, vec![])];
        assert!(mix_blocks(&broken, &cfg).is_err());
        let no_blocks = MixConfig {
            total_blocks: 0,
            ..cfg
        };
        assert!(mix_blocks(&sources(), &no_blocks).is_err());
    }

    #[test]
    fn total_blok_dan_cacah() {
        assert_eq!(block_counts(&sources()), vec![3, 2]);
        assert_eq!(total_blocks(&sources()), 5);
    }
}
