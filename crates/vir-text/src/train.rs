//! Pelatih BPE paralel (PRD F1.2 langkah 2).
//!
//! Algoritmanya: potong korpus per byte (pra-tokenisasi), hitung frekuensi
//! pasangan secara paralel dengan rayon, lalu gabungkan pasangan paling sering
//! berulang kali. Setiap iterasi menghitung ulang frekuensi dari kata yang
//! tersisa; inkremental (ind per pasangan) salah pada pasangan yang tumpang
//! tindih seperti `aaa` (tiap kemunculan dihitung sekali, tetapi tiap merge
//! mengurangi satu per posisi), sehingga sisa langkah bisa macet atau berbeda
//! menurut urutan. Dengan hitung-ulang, hasilnya fungsi murni dari multiset
//! kata, jadi **tidak bergantung urutan korpus**.
//!
//! Pemilihan memakai orde total: `freq` turun, lalu pasangan `(kiri, kanan)`
//! menurut byte naik. Karena `counts` adalah agregat (bukan posisi) dan
//! pemilihan hanya bergantung pada nilai pasangan, dua urutan korpus memberi
//! tabel merge identik. Biaya `M` merge adalah `O(M · token)`; pada korpus
//! uji (puluhan ribu token) ini beberapa detik di CPU CI.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::bpe::{MergeTable, NIL, Pair};
use crate::pretok::{DEFAULT_MAX_WORD_BYTES, PreTokenizer};

/// Target vocab bawaan (di luar blok dasar).
pub const DEFAULT_TARGET_VOCAB: usize = 512;
/// Frekuensi minimum agar sebuah pasangan digabung.
pub const DEFAULT_MIN_FREQUENCY: u32 = 2;

/// Konfigurasi pelatihan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainConfig {
    /// Ukuran vocab yang dituju, termasuk blok dasar.
    pub target_vocab: usize,
    /// Frekuensi minimum sebuah pasangan.
    pub min_frequency: u32,
    /// Batas panjang potongan pra-tokenisasi (byte).
    pub max_word_bytes: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            target_vocab: DEFAULT_TARGET_VOCAB,
            min_frequency: DEFAULT_MIN_FREQUENCY,
            max_word_bytes: DEFAULT_MAX_WORD_BYTES,
        }
    }
}

/// Satu node daftar terhubung: id token, tetangga kiri/kanan.
#[derive(Debug, Clone, Copy)]
struct Node {
    id: u32,
    prev: u32,
    next: u32,
}

/// Kata sebagai daftar terhubung; merge menimpa node kiri dan menandai node
/// kanan mati (id `NIL`), sehingga lintasan kiri→kanan hanya melewati node hidup.
#[derive(Debug, Clone)]
struct Word {
    nodes: Vec<Node>,
}

impl Word {
    fn from_bytes(bytes: &[u8]) -> Self {
        let len = bytes.len();
        let nodes = bytes
            .iter()
            .enumerate()
            .map(|(i, b)| Node {
                id: *b as u32,
                prev: if i == 0 { NIL } else { i as u32 - 1 },
                next: if i + 1 < len { i as u32 + 1 } else { NIL },
            })
            .collect();
        Self { nodes }
    }

    /// Kumpulkan pasangan hidup `(pair, kiri, kanan)` ke `out` (dikosongkan dulu).
    fn collect_pairs(&self, out: &mut Vec<(Pair, u32, u32)>) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let mut i = 0u32;
        loop {
            let node = self.nodes[i as usize];
            if node.next == NIL {
                break;
            }
            let right = self.nodes[node.next as usize];
            out.push(((node.id, right.id), i, node.next));
            i = node.next;
        }
    }

    /// Gabungkan setiap kemunculan `pair` pada kata ini, kiri→kanan.
    fn apply(&mut self, pair: Pair, new_id: u32, scratch: &mut Vec<(Pair, u32, u32)>) {
        self.collect_pairs(scratch);
        for (candidate, left, _) in scratch.iter().copied() {
            if candidate == pair {
                self.merge_at(left as usize, pair, new_id);
            }
        }
    }

    /// Gabungkan pasangan pada posisi `left` bila masih sah.
    fn merge_at(&mut self, left: usize, pair: Pair, new_id: u32) {
        if left >= self.nodes.len() || self.nodes[left].id != pair.0 {
            return;
        }
        let right = self.nodes[left].next as usize;
        if right >= self.nodes.len() || self.nodes[right].id != pair.1 {
            return;
        }
        let next = self.nodes[right].next;
        self.nodes[left].id = new_id;
        self.nodes[left].next = next;
        self.nodes[right].id = NIL;
        if next != NIL {
            self.nodes[next as usize].prev = left as u32;
        }
    }
}

/// Frekuensi pasangan satu kumpulan kata (paralel, lalu digabung).
fn count_pairs(words: &[Word]) -> HashMap<Pair, u32> {
    let partials: Vec<HashMap<Pair, u32>> = words
        .par_iter()
        .map(|word| {
            let mut counts: HashMap<Pair, u32> = HashMap::new();
            let mut scratch = Vec::new();
            word.collect_pairs(&mut scratch);
            for (pair, _, _) in &scratch {
                *counts.entry(*pair).or_insert(0) += 1;
            }
            counts
        })
        .collect();
    merge_counts(&partials)
}

/// Gabungkan peta frekuensi per potongan menjadi satu agregat.
fn merge_counts(partials: &[HashMap<Pair, u32>]) -> HashMap<Pair, u32> {
    let mut counts: HashMap<Pair, u32> = HashMap::new();
    for partial in partials {
        for (pair, count) in partial {
            *counts.entry(*pair).or_insert(0) += count;
        }
    }
    counts
}

/// Pasangan dengan frekuensi tertinggi; seri diputus orde total byte.
///
/// `max_by` memilih elemen terbesar menurut urutan ini: `count` naik (jadi
/// frekuensi tertinggi menang), lalu `(kiri, kanan)` mengecil (jadi pasangan
/// byte terkecil menang). Ini orde total yang sama dengan tumpukan lama, tetapi
/// dihitung dari agregat sehingga tidak bergantung urutan kemunculan.
fn best_pair(counts: &HashMap<Pair, u32>) -> Option<(Pair, u32)> {
    counts
        .iter()
        .filter(|(_, count)| **count > 0)
        .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
        .map(|(pair, count)| (*pair, *count))
}

/// Terapkan satu merge ke seluruh kata, kiri→kanan.
fn apply_merge(words: &mut [Word], pair: Pair, new_id: u32) {
    let mut scratch = Vec::new();
    for word in words.iter_mut() {
        word.apply(pair, new_id, &mut scratch);
    }
}

/// Potong korpus menjadi kata (paralel).
fn collect_words<S: AsRef<str> + Sync>(texts: &[S], max_word_bytes: usize) -> Vec<Word> {
    let pretok = PreTokenizer::new(max_word_bytes);
    texts
        .par_iter()
        .flat_map_iter(|text| {
            pretok
                .split(text.as_ref())
                .into_iter()
                .map(|chunk| Word::from_bytes(chunk.as_bytes()))
        })
        .collect()
}

/// Latih BPE dari iterator teks; berhenti bila vocab target atau pasangan habis.
pub fn train_bpe<I, S>(texts: I, cfg: &TrainConfig) -> MergeTable
where
    I: IntoIterator<Item = S>,
    S: AsRef<str> + Sync,
{
    let texts: Vec<S> = texts.into_iter().collect();
    let mut words = collect_words(&texts, cfg.max_word_bytes);
    let mut table = MergeTable::new();
    while table.vocab_size() < cfg.target_vocab {
        let Some((pair, count)) = best_pair(&count_pairs(&words)) else {
            break;
        };
        if count < cfg.min_frequency {
            break;
        }
        let Some(new_id) = table.push_merge(pair) else {
            break;
        };
        apply_merge(&mut words, pair, new_id);
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recount(words: &[Word]) -> HashMap<Pair, u32> {
        let mut counts: HashMap<Pair, u32> = HashMap::new();
        let mut scratch = Vec::new();
        for word in words {
            word.collect_pairs(&mut scratch);
            for (pair, _, _) in scratch.iter().copied() {
                *counts.entry(pair).or_insert(0) += 1;
            }
        }
        counts
    }

    #[test]
    fn frekuensi_instan_sepadan_dengan_hitung_ulang() {
        let corpus = ["banana bandana", "abracadabra", "abababab"];
        let mut words = collect_words(&corpus, DEFAULT_MAX_WORD_BYTES);
        let mut table = MergeTable::new();
        while table.merge_count() < 12 {
            let Some((pair, _)) = best_pair(&count_pairs(&words)) else {
                break;
            };
            let Some(new_id) = table.push_merge(pair) else {
                break;
            };
            apply_merge(&mut words, pair, new_id);
            assert_eq!(recount(&words), count_pairs(&words), "frekuensi menyimpang");
        }
        assert!(table.merge_count() > 0);
        assert_eq!(table.vocab_size(), 271 + table.merge_count());
    }

    #[test]
    fn korpus_kecil_menghasilkan_vocab_kecil() {
        let table = train_bpe(["ab ab ab"], &TrainConfig::default());
        assert!(table.vocab_size() > 271);
        assert!(table.vocab_size() < 300);
        assert!(table.merge_count() > 0);
    }

    #[test]
    fn pasangan_tumpang_tindih_tidak_macet() {
        // `aaa` punya pasangan yang tumpang tindih; hitung-ulang harus tetap maju.
        let table = train_bpe(["aaa aaa", "aaaa aaaa"], &TrainConfig::default());
        assert!(table.merge_count() >= 2);
    }

    #[test]
    fn kata_tunggal_tidak_menghasilkan_merge() {
        let table = train_bpe(["a"], &TrainConfig::default());
        assert_eq!(table.merge_count(), 0);
    }
}
