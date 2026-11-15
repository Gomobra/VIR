//! Encoder: trait [`Encoder`], implementasi acuan rank-based, dan registri backend
//! (PRD F1.2 langkah 3 dan 5).
//!
//! Encode memakai daftar terhubung + heap peringkat: setiap langkah menggabungkan
//! pasangan berperingkat terkecil yang masih ada. Jumlah merge ≤ jumlah byte,
//! jadi biayanya linear terhadap panjang masukan dengan faktor log dari heap, dan
//! potongan adversarial (satu kata ratusan ribu byte) tetap aman karena
//! pra-tokenisasi membatasi panjang potongan.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use vir_common::VirError;

use crate::bpe::{MergeTable, NIL, Pair};
use crate::pretok::PreTokenizer;
use crate::train::{TrainConfig, train_bpe};

/// Encoder apa pun: byte-level, lossless, deterministik.
pub trait Encoder: Send + Sync {
    /// Nama backend untuk log, TUI, dan ADR.
    fn name(&self) -> &'static str;
    /// Ukuran vocab termasuk blok dasar.
    fn vocab_size(&self) -> usize;
    /// Teks → id token.
    fn encode(&self, text: &str) -> Vec<u32>;
    /// Id token → byte mentah; `decode_bytes(encode(t)) == t.as_bytes()`.
    fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8>;
}

/// Backend yang dikenal registri.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Implementasi acuan di crate ini (oracle dan cadangan).
    Reference,
    /// Backend cepat snaptokens (PRD F1.2 lapis 2).
    SnapTokens,
    /// Backend cepat Gigatoken (PRD F1.2 lapis 3).
    GigaToken,
}

impl BackendKind {
    /// Uraikan nama backend.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "reference" | "ref" | "bpe" => Some(Self::Reference),
            "snaptokens" => Some(Self::SnapTokens),
            "gigatoken" => Some(Self::GigaToken),
            _ => None,
        }
    }

    /// Nama kanonik backend.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::SnapTokens => "snaptokens",
            Self::GigaToken => "gigatoken",
        }
    }

    /// `true` bila backend dapat dipakai pada build ini.
    ///
    /// Hanya implementasi acuan yang terpasang: backend cepat menunggu crate
    /// `snaptokens`/`gigatoken` ditambahkan beserta ADR pemilihan (PRD F1.2).
    pub const fn is_available(self) -> bool {
        matches!(self, Self::Reference)
    }

    /// Alasan backend cepat belum tersedia pada build ini.
    fn unavailable(self) -> VirError {
        let feature = self.name();
        if cfg!(feature = "snaptokens") || cfg!(feature = "gigatoken") {
            return VirError::Config(format!(
                "backend {feature} diaktifkan tetapi dependensinya belum dipasang"
            ));
        }
        VirError::Config(format!(
            "backend {feature} butuh feature '{feature}'; build ini memakai implementasi acuan"
        ))
    }
}

/// Penyangga kerja encoder; dipakai ulang antar potongan dalam satu panggilan.
#[derive(Debug)]
pub(crate) struct Scratch {
    ids: Vec<u32>,
    prev: Vec<u32>,
    next: Vec<u32>,
    heap: BinaryHeap<Reverse<(u32, u32)>>,
}

impl Scratch {
    pub(crate) fn new() -> Self {
        Self {
            ids: Vec::new(),
            prev: Vec::new(),
            next: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    /// Siapkan `len` node byte segar: `ids[i] = i`, tetangga awal berurutan.
    fn reset(&mut self, len: usize) {
        self.ids.clear();
        self.ids.extend(0..len as u32);
        self.prev.clear();
        self.prev
            .extend((0..len).map(|i| if i == 0 { NIL } else { i as u32 - 1 }));
        self.next.clear();
        self.next
            .extend((0..len).map(|i| if i + 1 < len { i as u32 + 1 } else { NIL }));
        self.heap.clear();
    }
}

/// Encode satu potongan byte ke `out`; catat merge yang dipakai ke `trace`.
pub(crate) fn encode_chunk(
    table: &MergeTable,
    bytes: &[u8],
    scratch: &mut Scratch,
    out: &mut Vec<u32>,
    trace: &mut Vec<Pair>,
) {
    let len = bytes.len();
    if len == 0 {
        return;
    }
    scratch.reset(len);
    for (i, byte) in bytes.iter().enumerate() {
        scratch.ids[i] = u32::from(*byte);
    }
    for (i, pair) in bytes.windows(2).enumerate() {
        let pair = (u32::from(pair[0]), u32::from(pair[1]));
        if let Some(rank) = table.rank(pair) {
            scratch.heap.push(Reverse((rank, i as u32)));
        }
    }
    while let Some(Reverse((rank, left))) = scratch.heap.pop() {
        let i = left as usize;
        if scratch.ids[i] == NIL {
            continue;
        }
        let j = scratch.next[i] as usize;
        if j >= len || scratch.ids[j] == NIL {
            continue;
        }
        let pair = (scratch.ids[i], scratch.ids[j]);
        if table.rank(pair) != Some(rank) {
            continue;
        }
        let Some(new_id) = table.id_for_rank(rank) else {
            continue;
        };
        let q = scratch.next[j];
        scratch.ids[i] = new_id;
        scratch.ids[j] = NIL;
        scratch.next[i] = q;
        if q != NIL {
            scratch.prev[q as usize] = i as u32;
        }
        trace.push(pair);
        push_candidate(table, scratch, i);
        let p = scratch.prev[i];
        if p != NIL {
            push_candidate(table, scratch, p as usize);
        }
    }
    let mut i = 0usize;
    while scratch.ids[i] != NIL {
        out.push(scratch.ids[i]);
        let next = scratch.next[i];
        if next == NIL {
            break;
        }
        i = next as usize;
    }
}

/// Catat pasangan baru di sekitar node `i` bila masih mungkin digabung.
fn push_candidate(table: &MergeTable, scratch: &mut Scratch, i: usize) {
    let j = scratch.next[i] as usize;
    if j >= scratch.ids.len() || scratch.ids[j] == NIL {
        return;
    }
    let pair = (scratch.ids[i], scratch.ids[j]);
    if let Some(rank) = table.rank(pair) {
        scratch.heap.push(Reverse((rank, i as u32)));
    }
}

/// Implementasi acuan: tabel merge + pra-tokenisasi.
#[derive(Debug, Clone)]
pub struct BpeEncoder {
    table: MergeTable,
    pretok: PreTokenizer,
}

impl Default for BpeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl BpeEncoder {
    /// Encoder dengan tabel dasar: `encode` = byte mentah.
    pub fn new() -> Self {
        Self::from_table(MergeTable::new())
    }

    /// Encoder dari tabel yang sudah dilatih atau diimpor.
    pub fn from_table(table: MergeTable) -> Self {
        Self {
            table,
            pretok: PreTokenizer::default(),
        }
    }

    /// Encoder dari tabel dengan pemecah khusus.
    pub fn with_pretok(table: MergeTable, pretok: PreTokenizer) -> Self {
        Self { table, pretok }
    }

    /// Latih encoder dari iterator teks.
    pub fn trained<I, S>(texts: I, cfg: &TrainConfig) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str> + Sync,
    {
        Self::from_table(train_bpe(texts, cfg))
    }

    /// Latih ulang dari kumpulan teks (dipakai `vir-py` dan `vir-cli`).
    pub fn train(&mut self, texts: &[String], target_vocab: usize) {
        let cfg = TrainConfig {
            target_vocab,
            ..TrainConfig::default()
        };
        self.table = train_bpe(texts.iter().map(|t| t.as_str()), &cfg);
    }

    /// Tabel merge encoder ini.
    pub fn table(&self) -> &MergeTable {
        &self.table
    }

    /// Ukuran vocab (mirror trait [`Encoder::vocab_size`]).
    pub fn vocab_size(&self) -> usize {
        self.table.vocab_size()
    }

    /// Teks → id token (mirror trait [`Encoder::encode`]).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        Encoder::encode(self, text)
    }

    /// Token id → teks; id di luar vocab dilewati (mirror `vir-py`).
    pub fn decode(&self, tokens: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(tokens)).into_owned()
    }

    /// Token id → byte mentah.
    pub fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(tokens.len());
        for id in tokens {
            if let Some(bytes) = self.table.token(*id) {
                out.extend_from_slice(bytes);
            }
        }
        out
    }
}

impl Encoder for BpeEncoder {
    fn name(&self) -> &'static str {
        "bpe-reference"
    }

    fn vocab_size(&self) -> usize {
        self.table.vocab_size()
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len());
        let mut scratch = Scratch::new();
        let mut trace: Vec<Pair> = Vec::new();
        for chunk in self.pretok.split(text) {
            trace.clear();
            encode_chunk(
                &self.table,
                chunk.as_bytes(),
                &mut scratch,
                &mut out,
                &mut trace,
            );
        }
        out
    }

    fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        BpeEncoder::decode_bytes(self, tokens)
    }
}

/// Pilih backend berdasarkan nama; backend cepat dilaporkan belum tersedia.
pub fn resolve_backend(name: &str, table: &MergeTable) -> Result<Box<dyn Encoder>, VirError> {
    let Some(kind) = BackendKind::parse(name) else {
        return Err(VirError::Config(format!(
            "backend '{name}' tidak dikenal; pilihan: reference, snaptokens, gigatoken"
        )));
    };
    if !kind.is_available() {
        return Err(kind.unavailable());
    }
    Ok(Box::new(BpeEncoder::from_table(table.clone())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tabel_dasar_menghasilkan_byte_identik() {
        let encoder = BpeEncoder::new();
        let text = "Héllo 世界 😀";
        assert_eq!(
            encoder.encode(text),
            text.bytes().map(u32::from).collect::<Vec<u32>>()
        );
        assert_eq!(encoder.decode_bytes(&encoder.encode(text)), text.as_bytes());
    }

    #[test]
    fn merge_mengurangi_jumlah_token() {
        let encoder = BpeEncoder::trained(["abab abab abab"], &TrainConfig::default());
        let ids = encoder.encode("abab abab abab");
        assert!(ids.len() < "abab abab abab".len());
        assert!(ids.iter().all(|id| (*id as usize) < encoder.vocab_size()));
    }

    #[test]
    fn backend_khusus_reference_saja() {
        let table = MergeTable::new();
        let picked = resolve_backend("reference", &table).map(|b| b.name()).ok();
        assert_eq!(picked, Some("bpe-reference"));
        assert!(resolve_backend("snaptokens", &table).is_err());
        assert!(resolve_backend("gigatoken", &table).is_err());
        assert!(resolve_backend("tidak-ada", &table).is_err());
    }
}
