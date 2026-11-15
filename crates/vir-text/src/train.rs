//! Pelatih BPE paralel (PRD F1.2 langkah 2).
//!
//! Algoritmanya: potong korpus per byte (pra-tokenisasi), hitung frekuensi
//! pasangan secara paralel dengan rayon, lalu gabungkan pasangan paling sering
//! berulang kali. Indeks pasangan diperbarui bertahap: hanya kata yang memuat
//! pasangan terpilih yang disentuh, jadi biaya total sebanding dengan jumlah
//! token di korpus. Hasil latih tidak bergantung pada urutan kata (agregat
//! frekuensi + tie-break pasangan terkecil), sehingga resume dan CI deterministik.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
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

/// Kata sebagai daftar terhubung; merge menimpa node kiri sehingga indeks node
/// tetap stabil dan entri indeks pasangan tidak pernah kedaluwarsa.
#[derive(Debug, Clone)]
struct Word {
    nodes: Vec<Node>,
    head: u32,
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
        Self { nodes, head: 0 }
    }

    /// Kumpulkan pasangan hidup `(pair, kiri, kanan)` ke `out` (dikosongkan dulu).
    fn collect_pairs(&self, out: &mut Vec<(Pair, u32, u32)>) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let mut i = self.head;
        while i != NIL {
            let node = self.nodes[i as usize];
            if node.next != NIL {
                let right = self.nodes[node.next as usize];
                out.push(((node.id, right.id), i, node.next));
                i = node.next;
            } else {
                break;
            }
        }
    }
}

/// Indeks pasangan lokal satu kumpulan kata.
#[derive(Debug, Default)]
struct LocalIndex {
    counts: HashMap<Pair, u32>,
    positions: HashMap<Pair, Vec<(u32, u32)>>,
}

impl LocalIndex {
    fn add_word(&mut self, word: &Word, scratch: &mut Vec<(Pair, u32, u32)>) {
        word.collect_pairs(scratch);
        for (pair, left, right) in scratch.iter().copied() {
            *self.counts.entry(pair).or_insert(0) += 1;
            self.positions.entry(pair).or_default().push((left, right));
        }
    }
}

/// Entri heap: jumlah kemunculan + pasangan. Memori lebih besar = lebih dulu.
#[derive(Debug, PartialEq, Eq)]
struct HeapEntry {
    count: u32,
    pair: Pair,
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| other.pair.0.cmp(&self.pair.0))
            .then_with(|| other.pair.1.cmp(&self.pair.1))
    }
}

/// Frekuensi pasangan + posisi kemunculannya, dengan pembaruan bertahap.
#[derive(Debug, Default)]
struct PairIndex {
    counts: HashMap<Pair, u32>,
    positions: HashMap<Pair, Vec<(u32, u32)>>,
    heap: BinaryHeap<HeapEntry>,
}

impl PairIndex {
    /// Hitung frekuensi seluruh kata secara paralel.
    fn build(words: &[Word]) -> Self {
        let locals: Vec<LocalIndex> = words
            .par_iter()
            .map(|word| {
                let mut local = LocalIndex::default();
                let mut scratch = Vec::new();
                local.add_word(word, &mut scratch);
                local
            })
            .collect();
        let mut counts: HashMap<Pair, u32> = HashMap::new();
        let mut positions: HashMap<Pair, Vec<(u32, u32)>> = HashMap::new();
        for local in locals {
            for (pair, count) in local.counts {
                *counts.entry(pair).or_insert(0) += count;
            }
            for (pair, mut pos) in local.positions {
                positions.entry(pair).or_default().append(&mut pos);
            }
        }
        let heap = counts
            .iter()
            .map(|(pair, count)| HeapEntry {
                count: *count,
                pair: *pair,
            })
            .collect();
        Self {
            counts,
            positions,
            heap,
        }
    }

    /// Pasangan dengan frekuensi tertinggi (tie-break: pasangan terkecil).
    fn best(&mut self) -> Option<(Pair, u32)> {
        while let Some(entry) = self.heap.pop() {
            if entry.count > 0 && self.counts.get(&entry.pair) == Some(&entry.count) {
                return Some((entry.pair, entry.count));
            }
        }
        None
    }

    /// Ubah frekuensi satu pasangan; hapus bila habis, catat bila bertambah.
    fn bump(&mut self, pair: Pair, delta: i64) {
        let next = i64::from(self.counts.get(&pair).copied().unwrap_or(0)) + delta;
        if next <= 0 {
            self.counts.remove(&pair);
            return;
        }
        let next = next as u32;
        self.counts.insert(pair, next);
        self.heap.push(HeapEntry { count: next, pair });
    }

    /// Terapkan merge ke seluruh kata yang memuat pasangan tersebut.
    fn apply(&mut self, words: &mut [Word], pair: Pair, new_id: u32) {
        let Some(positions) = self.positions.remove(&pair) else {
            return;
        };
        for (word, left) in positions {
            if self.merge_at(words, word, left, pair, new_id) {
                self.bump(pair, -1);
            }
        }
    }

    /// Gabungkan pasangan pada satu posisi; `true` bila posisi masih sah.
    fn merge_at(&mut self, words: &mut [Word], w: u32, left: u32, pair: Pair, new_id: u32) -> bool {
        let Some(word) = words.get_mut(w as usize) else {
            return false;
        };
        let i = left as usize;
        if i >= word.nodes.len() || word.nodes[i].id != pair.0 {
            return false;
        }
        let j = word.nodes[i].next as usize;
        if j >= word.nodes.len() || word.nodes[j].id != pair.1 {
            return false;
        }
        let p = word.nodes[i].prev;
        let q = word.nodes[j].next;
        if p != NIL {
            let id = word.nodes[p as usize].id;
            self.bump((id, pair.0), -1);
        }
        if q != NIL {
            let id = word.nodes[q as usize].id;
            self.bump((pair.1, id), -1);
        }
        word.nodes[i].id = new_id;
        word.nodes[i].next = q;
        // Node kanan mati: id NIL membuat posisi lama di indeks otomatis kedaluwarsa.
        word.nodes[j].id = NIL;
        if q != NIL {
            word.nodes[q as usize].prev = i as u32;
        }
        if p != NIL {
            let id = word.nodes[p as usize].id;
            self.bump((id, new_id), 1);
            self.positions
                .entry((id, new_id))
                .or_default()
                .push((p, i as u32));
        }
        if q != NIL {
            let id = word.nodes[q as usize].id;
            self.bump((new_id, id), 1);
            self.positions
                .entry((new_id, id))
                .or_default()
                .push((i as u32, q));
        }
        true
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
    let mut index = PairIndex::build(&words);
    let mut table = MergeTable::new();
    while table.vocab_size() < cfg.target_vocab {
        let Some((pair, count)) = index.best() else {
            break;
        };
        if count < cfg.min_frequency {
            break;
        }
        let Some(new_id) = table.push_merge(pair) else {
            break;
        };
        index.apply(&mut words, pair, new_id);
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
    fn indeks_pasangan_sesuai_hitung_ulang() {
        let corpus = ["banana bandana", "abracadabra", "abababab"];
        let mut words = collect_words(&corpus, DEFAULT_MAX_WORD_BYTES);
        let mut index = PairIndex::build(&words);
        let mut table = MergeTable::new();
        let mut merges = 0;
        while merges < 12 {
            let Some((pair, _)) = index.best() else {
                break;
            };
            let Some(new_id) = table.push_merge(pair) else {
                break;
            };
            index.apply(&mut words, pair, new_id);
            assert_eq!(index.counts, recount(&words), "frekuensi menyimpang");
            merges += 1;
        }
        assert!(merges > 0);
        assert_eq!(table.vocab_size(), 271 + merges);
    }

    #[test]
    fn korpus_kecil_menghasilkan_vocab_kecil() {
        let table = train_bpe(["ab ab ab"], &TrainConfig::default());
        assert!(table.vocab_size() > 271);
        assert!(table.vocab_size() < 300);
        assert!(table.merge_count() > 0);
    }

    #[test]
    fn kata_tunggal_tidak_menghasilkan_merge() {
        let table = train_bpe(["a"], &TrainConfig::default());
        assert_eq!(table.merge_count(), 0);
    }
}
