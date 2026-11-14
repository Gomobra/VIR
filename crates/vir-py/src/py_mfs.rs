//! `vir.mfs`: memori sebagai berkas (PRD F3.3).

use pyo3::prelude::*;
use vir_mfs::MemoryStore;

use crate::err;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Memory>()?;
    Ok(())
}

/// Buka toko memori dari path (`:memory:` untuk in-memory).
fn open(path: &str) -> PyResult<MemoryStore> {
    let store = if path == ":memory:" {
        MemoryStore::open_in_memory()
    } else {
        MemoryStore::open(path)
    };
    store.map_err(err::runtime)
}

/// Toko memori MFS: SQLite, isi mentah hanya pernah ditambah.
#[pyclass(name = "Memory")]
pub struct Memory {
    store: MemoryStore,
}

#[pymethods]
impl Memory {
    /// Buka basis data berkas, atau `:memory:` untuk toko sementara.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Ok(Self { store: open(path)? })
    }

    /// Toko memori di RAM (untuk tes).
    #[staticmethod]
    fn in_memory() -> PyResult<Self> {
        Ok(Self {
            store: open(":memory:")?,
        })
    }

    /// Tulis versi baru sebuah path; kembalikan id versi.
    fn write(&self, path: &str, content: &str) -> PyResult<i64> {
        self.store.write(path, content).map_err(err::runtime)
    }

    /// Baca satu versi berdasarkan id.
    fn read(&self, version_id: i64) -> PyResult<String> {
        self.store.read(version_id).map_err(err::runtime)
    }

    /// Baca versi terbaru sebuah path.
    fn read_path(&self, path: &str) -> PyResult<String> {
        self.store.read_path(path).map_err(err::runtime)
    }

    /// Cari dengan FTS + vektor (RRF). Kembalikan `(id, isi, skor)`.
    fn find(&self, query: &str, limit: usize) -> PyResult<Vec<(i64, String, f64)>> {
        self.store.find(query, limit).map_err(err::runtime)
    }

    /// Riwayat versi sebuah path (terbaru dulu).
    fn history(&self, path: &str) -> PyResult<Vec<i64>> {
        self.store.history(path).map_err(err::runtime)
    }

    /// Tandai hapus lunak (bisa dibatalkan dengan `restore`).
    fn forget(&self, version_id: i64) -> PyResult<()> {
        self.store.forget(version_id).map_err(err::runtime)
    }

    /// Batalkan hapus lunak.
    fn restore(&self, version_id: i64) -> PyResult<()> {
        self.store.restore(version_id).map_err(err::runtime)
    }

    /// Verifikasi hash isi terhadap katalog.
    fn verify(&self, version_id: i64) -> PyResult<bool> {
        self.store.verify(version_id).map_err(err::runtime)
    }

    /// Hash sha256 isi versi.
    fn hash_of(&self, version_id: i64) -> PyResult<String> {
        self.store.hash_of(version_id).map_err(err::runtime)
    }

    /// Id versi terbaru sebuah path, atau `None`.
    fn current_version(&self, path: &str) -> PyResult<Option<i64>> {
        self.store.current_version(path).map_err(err::runtime)
    }
}
