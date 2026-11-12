//! `vir-mfs`: memori sebagai berkas di ruang nama berpohon.
//!
//! Prinsip (PRD F3.3):
//!
//! - Isi mentah tidak pernah diubah atau dibuang. `write` selalu menambah
//!   versi baru; `forget` hanya memasang penanda hapus lunak yang bisa dibatalkan
//!   `restore`. Tidak ada `UPDATE`/`DELETE` pada kolom `content`.
//! - Membaca tidak mengubah isi; akses masuk ke tabel `access_log` terpisah.
//! - Pencarian menggabungkan kata kunci (FTS5 + BM25) dan vektor (kosinus
//!   brute-force) lewat Reciprocal Rank Fusion dengan bobot, faktor usia, dan
//!   faktor akses.
//! - Duplikat dicegah dengan dedup kosinus yang **menaut** (`duplicate_of`),
//!   bukan menghapus.
//! - Konsolidasi menulis ringkasan sebagai versi baru dengan `source` dan
//!   tautan `derived_from`; isi sumber tetap utuh.
//!
//! Contoh:
//!
//! ```
//! use vir_mfs::MemoryStore;
//!
//! let store = MemoryStore::open_in_memory()?;
//! let v = store.write("/semantik/catatan", "rust memori")?;
//! assert_eq!(store.read(v)?, "rust memori");
//! # Ok::<(), vir_mfs::VirError>(())
//! ```

mod consolidate;
mod db;
mod dedup;
mod diff;
mod fusion;
mod meta;
mod ops;
mod search;
mod store;
mod vec_index;

pub use consolidate::{ConsolidationReport, HeadlineSummarizer, Summarizer};
pub use diff::{Diff, DiffKind, DiffLine};
pub use fusion::{
    FusionConfig, access_factor, access_multiplier, age_factor, rank_score, rrf_fuse,
};
pub use meta::LinkKind;
pub use ops::{
    ReadResult, ReadTarget, TreeNode, diff, find, forget, history, link, links, ls, neighbors,
    read, restore, tag, tree, untag, write,
};
pub use search::{SearchHit, SearchMode, SearchOptions};
pub use store::{
    DEFAULT_DEDUP_THRESHOLD, FileEntry, MAX_CONTENT_BYTES, MemoryStore, VersionInfo, WriteOutcome,
    namespace_of, normalize_path,
};
pub use vec_index::{
    Embedder, HashNgramEmbedder, brute_force, cosine, decode_vector, encode_vector,
};

pub use db::hex_sha256;

pub use vir_common::VirError;

/// Hasil operasi `vir-mfs`.
pub type Result<T> = std::result::Result<T, VirError>;
