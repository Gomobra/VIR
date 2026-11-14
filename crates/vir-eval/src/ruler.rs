//! F5.4 — pembangkit tugas konteks panjang RULER, deterministik dari seed.
//!
//! Kategori: retrieval (needle multi-key), multi-hop (pelacakan variabel),
//! agregasi (tiga kata paling sering). Skema token memakai rentang khusus
//! agar tidak bertabrakan dengan token teks biasa.

/// Token khusus skema RULER (rentang terpisah).
const TOK_EQ: u32 = 9_000_000;
const TOK_ASK: u32 = 9_000_001;
const TOK_VAR_BASE: u32 = 10_000_000;
const TOK_VALUE_BASE: u32 = 20_000_000;
const TOK_WORD_BASE: u32 = 30_000_000;
const TOK_FILLER_BASE: u32 = 40_000_000;
const FILLER_PALETTE: usize = 4;

/// PRNG splitmix64: kecil, tanpa dependensi, deterministik dari seed.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next_u64() % bound as u64) as usize
        }
    }
}

/// Fisher-Yates deterministik.
fn shuffle<T>(rng: &mut SplitMix64, xs: &mut [T]) {
    for i in (1..xs.len()).rev() {
        let j = rng.next_usize(i + 1);
        xs.swap(i, j);
    }
}

/// Pembangkit needle-in-haystack deterministik (sederhana).
pub fn gen_needle(seed: u64, haystack_tokens: usize) -> (Vec<u32>, u32) {
    // jarum = token khusus 42 di posisi acak
    let mut v = vec![1u32; haystack_tokens];
    let pos = (seed as usize * 37) % haystack_tokens.max(1);
    v[pos] = 42;
    (v, 42)
}

/// Satu contoh tugas RULER: haystack token, kueri, dan kunci jawaban.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RulerSample {
    pub haystack: Vec<u32>,
    pub query: Vec<u32>,
    pub answer: Vec<u32>,
}

/// Kategori RULER yang didukung.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RulerCategory {
    /// Retrieval: satu/lebih kunci (needle), nilai, dan kueri.
    Retrieval,
    /// Multi-hop: pelacakan variabel (rantai `X1 = v`, `X2 = X1`, ...).
    MultiHop,
    /// Agregasi: tiga kata paling sering muncul.
    Aggregation,
}

/// Pembangkit needle RULER deterministik (multi-key).
pub fn gen_ruler_needle(seed: u64, haystack_len: usize, num_keys: usize) -> (Vec<u32>, Vec<u32>) {
    let mut hay = vec![1u32; haystack_len];
    if hay.is_empty() {
        return (hay, Vec::new());
    }
    let mut keys = Vec::with_capacity(num_keys);
    for k in 0..num_keys {
        let pos = (seed as usize * (k + 1) * 37) % haystack_len;
        let key = 1000 + k as u32;
        hay[pos] = key;
        keys.push(key);
    }
    (hay, keys)
}

/// Pembangkit tugas RULER sesuai kategori, deterministik dari seed.
/// Panjang haystack minimal `haystack_len` (bisa lebih bila pola perlu muat).
pub fn gen_ruler(seed: u64, haystack_len: usize, category: RulerCategory) -> RulerSample {
    match category {
        RulerCategory::Retrieval => {
            let (haystack, keys) = gen_ruler_needle(seed, haystack_len, 1);
            RulerSample {
                haystack,
                query: vec![TOK_ASK],
                answer: keys,
            }
        }
        RulerCategory::MultiHop => gen_ruler_multi_hop(seed, haystack_len),
        RulerCategory::Aggregation => gen_ruler_aggregation(seed, haystack_len),
    }
}

/// Multi-hop (variable tracing): `V0 = nilai`, `V1 = V0`, ..., `Vk = V(k-1)`;
/// kueri menanyakan nilai variabel terakhir; jawaban = nilai kepala rantai.
fn gen_ruler_multi_hop(seed: u64, haystack_len: usize) -> RulerSample {
    let mut rng = SplitMix64::new(seed);
    let num_vars = 3 + rng.next_usize(4); // 3..=6
    let head_value = TOK_VALUE_BASE + (rng.next_u64() % 1_000) as u32;

    let mut hay: Vec<u32> = Vec::new();
    // V0 = head_value
    hay.extend_from_slice(&[TOK_VAR_BASE, TOK_EQ, head_value]);
    for i in 1..num_vars {
        hay.extend_from_slice(&[
            TOK_VAR_BASE + i as u32,
            TOK_EQ,
            TOK_VAR_BASE + (i - 1) as u32,
        ]);
    }

    // Baris pengganggu: `V<k> = nilai_bebas`, tidak membentuk rantai ke head.
    let mut d = 0u32;
    while hay.len() < haystack_len {
        let val = TOK_VALUE_BASE + 1_000 + (rng.next_u64() % 100_000) as u32;
        let var = TOK_VAR_BASE + 1_000 + d;
        hay.extend_from_slice(&[var, TOK_EQ, val]);
        d += 1;
    }

    shuffle(&mut rng, &mut hay);
    let query = vec![TOK_ASK, TOK_VAR_BASE + (num_vars - 1) as u32];
    RulerSample {
        haystack: hay,
        query,
        answer: vec![head_value],
    }
}

/// Agregasi: tiga kata target muncul paling sering; sisanya kata pengganggu.
fn gen_ruler_aggregation(seed: u64, haystack_len: usize) -> RulerSample {
    let mut rng = SplitMix64::new(seed);
    let n = haystack_len.max(1);
    let hi = (n / 4).max(1);

    // Tiga kata target berbeda.
    let mut words: Vec<u32> = Vec::with_capacity(3);
    while words.len() < 3 {
        let w = TOK_WORD_BASE + (rng.next_u64() % 1_000) as u32;
        if !words.contains(&w) {
            words.push(w);
        }
    }

    let mut hay: Vec<u32> = Vec::with_capacity(n);
    for &w in &words {
        for _ in 0..hi {
            hay.push(w);
        }
    }
    // Sisa token dibagi rata ke palet kata pengganggu (tiap kata < hi).
    let rem = n.saturating_sub(hay.len());
    for i in 0..rem {
        hay.push(TOK_FILLER_BASE + (i % FILLER_PALETTE) as u32);
    }

    shuffle(&mut rng, &mut hay);
    RulerSample {
        haystack: hay,
        query: vec![TOK_ASK],
        answer: words,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_token(hay: &[u32], t: u32) -> usize {
        hay.iter().filter(|&&x| x == t).count()
    }

    #[test]
    fn needle_contains_jarum() {
        let (hay, needle) = gen_needle(7, 100);
        assert!(hay.contains(&needle));
    }

    #[test]
    fn ruler_needle_has_all_keys() {
        let (hay, keys) = gen_ruler_needle(7, 100, 3);
        for k in keys {
            assert!(hay.contains(&k));
        }
    }

    #[test]
    fn ruler_retrieval_answer_present() {
        let s = gen_ruler(7, 100, RulerCategory::Retrieval);
        assert_eq!(s.answer.len(), 1);
        assert!(s.haystack.contains(&s.answer[0]));
    }

    #[test]
    fn ruler_multi_hop_is_deterministic() {
        let a = gen_ruler(42, 200, RulerCategory::MultiHop);
        let b = gen_ruler(42, 200, RulerCategory::MultiHop);
        assert_eq!(a, b);
    }

    #[test]
    fn ruler_multi_hop_answer_present_and_query_targets_last_var() {
        let s = gen_ruler(42, 200, RulerCategory::MultiHop);
        assert_eq!(s.answer.len(), 1);
        assert!(s.haystack.contains(&s.answer[0]));
        assert_eq!(s.query[0], TOK_ASK);
        assert!(s.query[1] >= TOK_VAR_BASE);
        assert!(s.haystack.len() >= 200);
    }

    #[test]
    fn ruler_multi_hop_seed_changes_output() {
        let a = gen_ruler(1, 200, RulerCategory::MultiHop);
        let b = gen_ruler(2, 200, RulerCategory::MultiHop);
        assert_ne!(a, b);
    }

    #[test]
    fn ruler_aggregation_is_deterministic() {
        let a = gen_ruler(9, 400, RulerCategory::Aggregation);
        let b = gen_ruler(9, 400, RulerCategory::Aggregation);
        assert_eq!(a, b);
    }

    #[test]
    fn ruler_aggregation_top_three_are_most_frequent() {
        let s = gen_ruler(9, 400, RulerCategory::Aggregation);
        assert_eq!(s.answer.len(), 3);
        let ans_counts: Vec<usize> = s
            .answer
            .iter()
            .map(|&w| count_token(&s.haystack, w))
            .collect();
        let min_target = ans_counts.iter().copied().min().unwrap_or(0);
        assert!(min_target > 0);
        for &w in &s.answer {
            assert!(count_token(&s.haystack, w) >= min_target);
        }
        // kata pengganggu tidak boleh mengalahkan target
        for i in 0..FILLER_PALETTE {
            let f = TOK_FILLER_BASE + i as u32;
            assert!(count_token(&s.haystack, f) < min_target);
        }
        assert_eq!(s.haystack.len(), 400);
    }

    #[test]
    fn ruler_aggregation_seed_changes_output() {
        let a = gen_ruler(11, 400, RulerCategory::Aggregation);
        let b = gen_ruler(12, 400, RulerCategory::Aggregation);
        assert_ne!(a, b);
    }
}
