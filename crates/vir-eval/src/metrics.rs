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

/// Hitung multiset n-gram dari sekuens token.
fn ngram_counts(tokens: &[u32], n: usize) -> BTreeMap<Vec<u32>, u32> {
    let mut map: BTreeMap<Vec<u32>, u32> = BTreeMap::new();
    if n == 0 || tokens.len() < n {
        return map;
    }
    for w in tokens.windows(n) {
        *map.entry(w.to_vec()).or_insert(0) += 1;
    }
    map
}

/// F1 dari tumpang tindih n-gram (ROUGE-N), token sebagai satuan.
pub fn rouge_n(reference: &[u32], candidate: &[u32], n: usize) -> f64 {
    let r = ngram_counts(reference, n);
    let c = ngram_counts(candidate, n);
    let r_total: u32 = r.values().sum();
    let c_total: u32 = c.values().sum();
    if r_total == 0 || c_total == 0 {
        return 0.0;
    }
    let overlap: u32 = c
        .iter()
        .map(|(gram, &count)| count.min(r.get(gram).copied().unwrap_or(0)))
        .sum();
    let precision = overlap as f64 / c_total as f64;
    let recall = overlap as f64 / r_total as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// Panjang subsequence bersama terpanjang (LCS) untuk ROUGE-L.
fn lcs_len(a: &[u32], b: &[u32]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let mut prev = vec![0usize; b.len() + 1];
    let mut cur = vec![0usize; b.len() + 1];
    for &x in a {
        for (j, &y) in b.iter().enumerate() {
            cur[j + 1] = if x == y {
                prev[j] + 1
            } else {
                prev[j + 1].max(cur[j])
            };
        }
        std::mem::swap(&mut prev, &mut cur);
        for slot in cur.iter_mut() {
            *slot = 0;
        }
    }
    prev[b.len()]
}

/// ROUGE-L: F1 dari panjang LCS terhadap panjang referensi dan kandidat.
pub fn rouge_l(reference: &[u32], candidate: &[u32]) -> f64 {
    let lcs = lcs_len(reference, candidate) as f64;
    if lcs == 0.0 {
        return 0.0;
    }
    let precision = lcs / candidate.len() as f64;
    let recall = lcs / reference.len() as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// BLEU (satu referensi) dengan brevity penalty, sampai 4-gram.
pub fn bleu(reference: &[u32], candidate: &[u32]) -> f64 {
    if candidate.is_empty() || reference.is_empty() {
        return 0.0;
    }
    let max_n = candidate.len().min(4);
    let mut log_sum = 0.0f64;
    for n in 1..=max_n {
        let r = ngram_counts(reference, n);
        let c = ngram_counts(candidate, n);
        let c_total: u32 = c.values().sum();
        if c_total == 0 {
            return 0.0;
        }
        let overlap: u32 = c
            .iter()
            .map(|(gram, &count)| count.min(r.get(gram).copied().unwrap_or(0)))
            .sum();
        let precision = overlap as f64 / c_total as f64;
        if precision <= 0.0 {
            return 0.0;
        }
        log_sum += precision.ln();
    }
    let brevity = if candidate.len() as f64 >= reference.len() as f64 {
        1.0
    } else {
        (1.0 - reference.len() as f64 / candidate.len() as f64).exp()
    };
    brevity * (log_sum / max_n as f64).exp()
}
