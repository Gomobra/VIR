//! F4.5 — metrik generatif tanpa model eksternal.
//!
//! Gen. PPL dihitung dari distribusi token referensi (bukan model eksternal),
//! dan harus dilaporkan bersama entropi unigram karena PPL rendah bisa berasal
//! dari keragaman yang rendah.

use std::collections::BTreeMap;

/// Distribusi token referensi (unigram), dipakai sebagai "model referensi"
/// untuk Gen. PPL. Dibangun dari hitungan token, bukan dari model eksternal.
#[derive(Clone, Debug)]
pub struct TokenDist {
    counts: BTreeMap<u32, u64>,
    total: u64,
    alpha: f64,
}

impl TokenDist {
    /// Distribusi dari hitungan token `(id, count)`, tanpa smoothing.
    pub fn from_counts(counts: &[(u32, usize)]) -> Self {
        let mut map = BTreeMap::new();
        let mut total = 0u64;
        for &(tok, c) in counts {
            *map.entry(tok).or_insert(0u64) += c as u64;
            total += c as u64;
        }
        Self {
            counts: map,
            total,
            alpha: 0.0,
        }
    }

    /// Distribusi dengan smoothing Laplace berparameter `alpha`.
    pub fn smoothed(counts: &[(u32, usize)], alpha: f64) -> Self {
        let mut d = Self::from_counts(counts);
        d.alpha = alpha.max(0.0);
        d
    }

    /// Probabilitas token. Tanpa smoothing, token tak terlihat bernilai 0.
    pub fn prob(&self, token: u32) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let c = self.counts.get(&token).copied().unwrap_or(0) as f64;
        if self.alpha > 0.0 {
            let vocab = self.counts.len() as f64 + 1.0;
            (c + self.alpha) / (self.total as f64 + self.alpha * vocab)
        } else {
            c / self.total as f64
        }
    }

    /// Entropi (bit) dari distribusi referensi itu sendiri.
    pub fn entropy_bits(&self) -> f64 {
        let mut h = 0.0;
        for &c in self.counts.values() {
            if c == 0 {
                continue;
            }
            let p = c as f64 / self.total as f64;
            h -= p * p.log2();
        }
        h
    }

    /// Jumlah token total referensi.
    pub fn total(&self) -> u64 {
        self.total
    }
}

/// Entropi unigram (bit) dari distribusi empiris sebuah sampel token.
/// Mengukur keragaman; nilai rendah berarti generasi tidak beragam.
pub fn unigram_entropy(tokens: &[u32]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
    for &t in tokens {
        *counts.entry(t).or_insert(0usize) += 1;
    }
    let n = tokens.len() as f64;
    let mut h = 0.0;
    for &c in counts.values() {
        let p = c as f64 / n;
        h -= p * p.log2();
    }
    h
}

/// Gen. PPL: perplexity generatif = `2^H(q, p)`, dengan `q` distribusi empiris
/// sampel dan `p` distribusi referensi. Rendah = dekat referensi, tetapi harus
/// dilaporkan bersama [`unigram_entropy`] karena PPL rendah bisa berasal dari
/// keragaman yang rendah. Token sampel di luar referensi memberi PPL tak hingga
/// (kecuali [`TokenDist`] dibuat dengan smoothing).
pub fn generative_ppl(sample: &[u32], reference: &TokenDist) -> f64 {
    if sample.is_empty() {
        return 1.0;
    }
    let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
    for &t in sample {
        *counts.entry(t).or_insert(0usize) += 1;
    }
    let n = sample.len() as f64;
    let mut cross_bits = 0.0;
    for (&tok, &c) in &counts {
        let p = reference.prob(tok);
        if p <= 0.0 {
            return f64::INFINITY;
        }
        let q = c as f64 / n;
        cross_bits -= q * p.log2();
    }
    2f64.powf(cross_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unigram_entropy_uniform_binary() {
        // [a,a,b,b] -> 1 bit
        assert!((unigram_entropy(&[10, 10, 20, 20]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn unigram_entropy_single_symbol() {
        assert!(unigram_entropy(&[5, 5, 5, 5]).abs() < 1e-12);
        assert!(unigram_entropy(&[]).abs() < 1e-12);
    }

    #[test]
    fn unigram_entropy_four_symbols() {
        // distribusi seragam 4 simbol -> 2 bit
        let toks = [1, 2, 3, 4];
        assert!((unigram_entropy(&toks) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn generative_ppl_uniform_matches_entropy() {
        let reference = TokenDist::from_counts(&[(10, 1), (20, 1)]);
        let sample = [10, 20, 10, 20];
        // q = p = {0.5,0.5} -> cross-entropy 1 bit -> ppl 2
        assert!((generative_ppl(&sample, &reference) - 2.0).abs() < 1e-12);
        assert!((reference.entropy_bits() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn generative_ppl_skewed_reference() {
        let reference = TokenDist::from_counts(&[(10, 3), (20, 1)]);
        let sample = [10, 20];
        // H = -0.5*log2(0.75) - 0.5*log2(0.25) = 1.2075, ppl = 2^1.2075
        let h = -0.5 * (0.75f64).log2() - 0.5 * (0.25f64).log2();
        let expected = 2f64.powf(h);
        let got = generative_ppl(&sample, &reference);
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got} expected {expected}"
        );
    }

    #[test]
    fn generative_ppl_infinite_for_unknown_token() {
        let reference = TokenDist::from_counts(&[(10, 1)]);
        assert!(generative_ppl(&[10, 99], &reference).is_infinite());
    }

    #[test]
    fn generative_ppl_smoothing_finite() {
        let reference = TokenDist::smoothed(&[(10, 3), (20, 1)], 1.0);
        let ppl = generative_ppl(&[10, 99], &reference);
        assert!(ppl.is_finite() && ppl > 0.0);
    }

    #[test]
    fn empty_sample_ppl_is_one() {
        let reference = TokenDist::from_counts(&[(10, 1)]);
        assert!((generative_ppl(&[], &reference) - 1.0).abs() < 1e-12);
    }
}
