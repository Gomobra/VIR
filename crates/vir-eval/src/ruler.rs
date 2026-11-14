//! F5.4 — pembangkit tugas konteks panjang RULER, deterministik dari seed.
//!
//! Kategori mengikuti RULER (arXiv 2404.06654) ditambah satu kategori milik VIR:
//! retrieval (needle multi-key), multi-hop (pelacakan variabel), agregasi (kata
//! paling sering), tanya-jawab dengan pengganggu, dan kode lintas-berkas.
//! Skema token memakai rentang khusus agar tidak bertabrakan dengan token teks
//! biasa. Panjang efektif = panjang terbesar dengan skor rata-rata di atas ambang.

use std::collections::BTreeMap;

/// Token khusus skema RULER (rentang terpisah).
const TOK_EQ: u32 = 9_000_000;
const TOK_ASK: u32 = 9_000_001;
const TOK_FILE: u32 = 9_000_002;
const TOK_VAR_BASE: u32 = 10_000_000;
const TOK_VALUE_BASE: u32 = 20_000_000;
const TOK_WORD_BASE: u32 = 30_000_000;
const TOK_FILLER_BASE: u32 = 40_000_000;
const TOK_KEY_BASE: u32 = 50_000_000;
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
    let mut v = vec![1u32; haystack_tokens];
    if v.is_empty() {
        return (v, 42);
    }
    let pos = (seed as usize * 37) % haystack_tokens;
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

/// Kategori RULER yang didukung (empat kategori makalah + kode lintas-berkas).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RulerCategory {
    /// Retrieval: satu/lebih kunci (needle), nilai, dan kueri.
    Retrieval,
    /// Multi-hop: pelacakan variabel (rantai `X1 = v`, `X2 = X1`, ...).
    MultiHop,
    /// Agregasi: tiga kata paling sering muncul.
    Aggregation,
    /// Tanya-jawab dengan pengganggu: satu fakta relevan di antara banyak.
    QuestionAnswering,
    /// Kode lintas-berkas (khusus VIR): alias simbol melintasi blok berkas.
    CodeCrossFile,
}

impl RulerCategory {
    /// Semua kategori yang didukung, urutan stabil.
    pub const ALL: [RulerCategory; 5] = [
        RulerCategory::Retrieval,
        RulerCategory::MultiHop,
        RulerCategory::Aggregation,
        RulerCategory::QuestionAnswering,
        RulerCategory::CodeCrossFile,
    ];

    /// Nama ringkas kategori (dipakai sebagai label tabel dan konfigurasi).
    pub fn name(self) -> &'static str {
        match self {
            RulerCategory::Retrieval => "retrieval",
            RulerCategory::MultiHop => "multi-hop",
            RulerCategory::Aggregation => "aggregation",
            RulerCategory::QuestionAnswering => "qa",
            RulerCategory::CodeCrossFile => "code-cross-file",
        }
    }
}

/// Panjang efektif: panjang terbesar dengan skor rata-rata di atas ambang.
///
/// Mengembalikan `None` bila tidak ada panjang yang melewati ambang. Panjang
/// boleh tidak terurut; fungsi memilih nilai terbesar di antara yang lolos.
pub fn effective_context_length(
    lengths: &[usize],
    scores: &[f64],
    threshold: f64,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (&len, &score) in lengths.iter().zip(scores.iter()) {
        if score > threshold {
            best = Some(match best {
                Some(current) if current >= len => current,
                _ => len,
            });
        }
    }
    best
}

/// Skor jawaban model terhadap kunci: recall multiset (0..=1).
///
/// Setiap token kunci yang muncul di keluaran (dengan pengulangan) dihitung satu
/// kali. Jawaban kosong hanya cocok bila keluaran juga kosong.
pub fn match_score(expected: &[u32], actual: &[u32]) -> f64 {
    if expected.is_empty() {
        return if actual.is_empty() { 1.0 } else { 0.0 };
    }
    let mut need: BTreeMap<u32, i64> = BTreeMap::new();
    for &tok in expected {
        *need.entry(tok).or_insert(0) += 1;
    }
    let mut hit = 0i64;
    for &tok in actual {
        if let Some(count) = need.get_mut(&tok) {
            if *count > 0 {
                *count -= 1;
                hit += 1;
            }
        }
    }
    hit as f64 / expected.len() as f64
}

/// Pembangkit tugas RULER sesuai kategori, deterministik dari seed.
/// Panjang haystack minimal `haystack_len` (bisa lebih bila pola perlu muat).
pub fn gen_ruler(seed: u64, haystack_len: usize, category: RulerCategory) -> RulerSample {
    match category {
        RulerCategory::Retrieval => {
            let (haystack, query, answer) = build_retrieval(seed, haystack_len, 1);
            RulerSample {
                haystack,
                query,
                answer,
            }
        }
        RulerCategory::MultiHop => gen_ruler_multi_hop(seed, haystack_len),
        RulerCategory::Aggregation => gen_ruler_aggregation(seed, haystack_len),
        RulerCategory::QuestionAnswering => gen_ruler_qa(seed, haystack_len),
        RulerCategory::CodeCrossFile => gen_ruler_code(seed, haystack_len),
    }
}

/// Pembangkit needle RULER deterministik (multi-key).
/// Mengembalikan `(haystack, nilai_jawaban)`; kueri menyusun kunci-kuncinya.
pub fn gen_ruler_needle(seed: u64, haystack_len: usize, num_keys: usize) -> (Vec<u32>, Vec<u32>) {
    let (haystack, _query, answer) = build_retrieval(seed, haystack_len, num_keys);
    (haystack, answer)
}

/// Retrieval multi-key: sebar fakta `key = value` di haystack berisi pengganggu.
fn build_retrieval(
    seed: u64,
    haystack_len: usize,
    num_keys: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut rng = SplitMix64::new(seed);
    let mut hay = vec![TOK_FILLER_BASE; haystack_len];
    // Setiap fakta butuh 3 token; jumlah maksimum yang muat tanpa tumpang tindih.
    let usable = hay.len().saturating_sub(3);
    let max_keys = usable / 3 + 1;
    let num_keys = num_keys.min(max_keys);
    if num_keys == 0 || usable == 0 {
        return (hay, vec![TOK_ASK], Vec::new());
    }
    // Jarak antar fakta >= 3, sehingga blok tidak saling menimpa.
    let stride = if num_keys > 1 {
        usable / (num_keys - 1)
    } else {
        0
    };
    let mut keys = Vec::with_capacity(num_keys);
    let mut values = Vec::with_capacity(num_keys);
    for k in 0..num_keys {
        let key = TOK_KEY_BASE + (rng.next_u64() % 1_000_000) as u32 + k as u32;
        let value = TOK_VALUE_BASE + (rng.next_u64() % 1_000_000) as u32;
        let pos = k * stride;
        hay[pos] = key;
        hay[pos + 1] = TOK_EQ;
        hay[pos + 2] = value;
        keys.push(key);
        values.push(value);
    }
    let mut query = Vec::with_capacity(1 + keys.len());
    query.push(TOK_ASK);
    query.extend_from_slice(&keys);
    (hay, query, values)
}

/// Multi-hop (variable tracing): `V0 = nilai`, `V1 = V0`, ..., `Vk = V(k-1)`;
/// kueri menanyakan nilai variabel terakhir; jawaban = nilai kepala rantai.
/// Baris (bukan token) yang diacak agar setiap assignment tetap utuh.
fn gen_ruler_multi_hop(seed: u64, haystack_len: usize) -> RulerSample {
    let mut rng = SplitMix64::new(seed);
    let num_vars = 3 + rng.next_usize(4); // 3..=6
    let head_value = TOK_VALUE_BASE + (rng.next_u64() % 1_000) as u32;

    let mut lines: Vec<Vec<u32>> = Vec::new();
    lines.push(vec![TOK_VAR_BASE, TOK_EQ, head_value]);
    for i in 1..num_vars {
        lines.push(vec![
            TOK_VAR_BASE + i as u32,
            TOK_EQ,
            TOK_VAR_BASE + (i - 1) as u32,
        ]);
    }
    add_distractor_lines(&mut lines, haystack_len, &mut rng);

    shuffle(&mut rng, &mut lines);
    let haystack = flatten_lines(&lines);
    let query = vec![TOK_ASK, TOK_VAR_BASE + (num_vars - 1) as u32];
    RulerSample {
        haystack,
        query,
        answer: vec![head_value],
    }
}

/// Tambah baris pengganggu `V<k> = nilai_bebas` hingga total token >= target.
fn add_distractor_lines(lines: &mut Vec<Vec<u32>>, target: usize, rng: &mut SplitMix64) {
    let mut total: usize = lines.iter().map(Vec::len).sum();
    let mut d = 0u32;
    while total < target {
        let val = TOK_VALUE_BASE + 1_000 + (rng.next_u64() % 100_000) as u32;
        let var = TOK_VAR_BASE + 1_000 + d;
        lines.push(vec![var, TOK_EQ, val]);
        total += 3;
        d += 1;
    }
}

/// Sambung baris-baris menjadi satu sekuens token.
fn flatten_lines(lines: &[Vec<u32>]) -> Vec<u32> {
    let total: usize = lines.iter().map(Vec::len).sum();
    let mut hay = Vec::with_capacity(total);
    for line in lines {
        hay.extend_from_slice(line);
    }
    hay
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

/// Tanya-jawab dengan pengganggu: banyak fakta `key = value`, satu relevan.
/// Fakta (bukan token) yang diacak agar setiap pasangan kunci-nilai tetap utuh.
fn gen_ruler_qa(seed: u64, haystack_len: usize) -> RulerSample {
    let mut rng = SplitMix64::new(seed);
    let n = haystack_len.max(4);
    let target_key = TOK_KEY_BASE + (rng.next_u64() % 1_000_000) as u32;
    let target_val = TOK_VALUE_BASE + (rng.next_u64() % 1_000_000) as u32;

    let mut facts: Vec<Vec<u32>> = vec![vec![target_key, TOK_EQ, target_val]];
    while (facts.len() + 1) * 3 <= n {
        let key = TOK_KEY_BASE + 1_000_000 + (rng.next_u64() % 1_000_000) as u32;
        let val = TOK_VALUE_BASE + 1_000_000 + (rng.next_u64() % 1_000_000) as u32;
        facts.push(vec![key, TOK_EQ, val]);
    }

    shuffle(&mut rng, &mut facts);
    let mut hay = flatten_lines(&facts);
    while hay.len() < n {
        hay.push(TOK_FILLER_BASE + (hay.len() % FILLER_PALETTE) as u32);
    }

    RulerSample {
        haystack: hay,
        query: vec![TOK_ASK, target_key],
        answer: vec![target_val],
    }
}

/// Kode lintas-berkas: tiap blok berkas mendefinisikan simbol; berkas ke-`i`
/// mengalias ke simbol berkas sebelumnya, sehingga nilai harus dilacak lintas blok.
fn gen_ruler_code(seed: u64, haystack_len: usize) -> RulerSample {
    let mut rng = SplitMix64::new(seed);
    let num_files = 3 + rng.next_usize(3); // 3..=5
    let head = TOK_VALUE_BASE + (rng.next_u64() % 1_000_000) as u32;

    let mut files: Vec<Vec<u32>> = Vec::with_capacity(num_files);
    for f in 0..num_files {
        let mut block = vec![TOK_FILE, f as u32, TOK_VAR_BASE + f as u32, TOK_EQ];
        if f == 0 {
            block.push(head);
        } else {
            block.push(TOK_VAR_BASE + (f - 1) as u32);
        }
        files.push(block);
    }
    pad_files(&mut files, haystack_len, &mut rng);
    shuffle(&mut rng, &mut files);

    let mut haystack = Vec::with_capacity(haystack_len);
    for file in &files {
        haystack.extend_from_slice(file);
    }
    let query = vec![TOK_ASK, TOK_VAR_BASE + (num_files - 1) as u32];
    RulerSample {
        haystack,
        query,
        answer: vec![head],
    }
}

/// Tambahkan token pengganggu ke berkas hingga total token mencapai `target`.
fn pad_files(files: &mut [Vec<u32>], target: usize, rng: &mut SplitMix64) {
    let mut total: usize = files.iter().map(Vec::len).sum();
    let mut idx = 0usize;
    while total < target && !files.is_empty() {
        let filler = TOK_FILLER_BASE + (rng.next_u64() % FILLER_PALETTE as u64) as u32;
        files[idx % files.len()].push(filler);
        total += 1;
        idx += 1;
    }
}
