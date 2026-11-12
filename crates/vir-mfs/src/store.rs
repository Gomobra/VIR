//! Penyimpanan berkas memori: versi tak berubah, tautan, dan hapus lunak.

use rusqlite::{Connection, params};

use crate::db::{
    hex_sha256, i64_vec, init_schema, one_i64, one_string, open_connection, open_memory, sql_err,
    unix_now,
};
use crate::vec_index::{Embedder, HashNgramEmbedder};
use crate::{Result, VirError};

/// Ambang kosinus bawaan untuk mendeteksi duplikat.
pub const DEFAULT_DEDUP_THRESHOLD: f32 = 0.98;
/// Batas isi satu versi (8 MiB).
pub const MAX_CONTENT_BYTES: usize = 8 * 1024 * 1024;
const MAX_PATH_LEN: usize = 1024;

/// Toko memori MFS: SQLite di belakang, isi mentah hanya pernah ditambah.
pub struct MemoryStore {
    pub(crate) conn: Connection,
    pub(crate) embedder: Option<Box<dyn Embedder>>,
    pub(crate) dedup_threshold: f32,
}

/// Hasil penulisan versi baru.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WriteOutcome {
    pub version_id: i64,
    pub supersedes: Option<i64>,
    /// `(versi duplikat, kosinus)` bila melewati ambang dedup.
    pub duplicate_of: Option<(i64, f32)>,
}

impl MemoryStore {
    /// Buka basis data berkas dengan embedder bawaan.
    pub fn open(path: &str) -> Result<Self> {
        Self::from_connection(open_connection(path)?)
    }

    /// Buka basis data in-memory dengan embedder bawaan.
    pub fn open_in_memory() -> Result<Self> {
        Self::from_connection(open_memory()?)
    }

    /// Buka dengan embedder khusus.
    pub fn open_with(path: &str, embedder: Box<dyn Embedder>) -> Result<Self> {
        let mut store = Self::from_connection(open_connection(path)?)?;
        store.embedder = Some(embedder);
        Ok(store)
    }

    /// Buka in-memory dengan embedder khusus.
    pub fn open_in_memory_with(embedder: Box<dyn Embedder>) -> Result<Self> {
        let mut store = Self::from_connection(open_memory()?)?;
        store.embedder = Some(embedder);
        Ok(store)
    }

    /// Buka tanpa indeks vektor (hanya FTS).
    pub fn open_in_memory_without_index() -> Result<Self> {
        let mut store = Self::from_connection(open_memory()?)?;
        store.embedder = None;
        Ok(store)
    }

    fn from_connection(conn: Connection) -> Result<Self> {
        init_schema(&conn)?;
        Ok(Self {
            conn,
            embedder: Some(Box::new(HashNgramEmbedder::default())),
            dedup_threshold: DEFAULT_DEDUP_THRESHOLD,
        })
    }

    /// Atur ambang dedup (mengonsumsi toko).
    pub fn with_dedup_threshold(mut self, threshold: f32) -> Self {
        self.dedup_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn dedup_threshold(&self) -> f32 {
        self.dedup_threshold
    }

    pub fn embedder(&self) -> Option<&dyn Embedder> {
        self.embedder.as_deref()
    }

    /// Versi terbaru suatu berkas, bila ada.
    pub fn current_version(&self, path: &str) -> Result<Option<i64>> {
        let Some(file_id) = self.file_id(path)? else {
            return Ok(None);
        };
        latest_version(&self.conn, file_id)
    }

    pub fn file_id(&self, path: &str) -> Result<Option<i64>> {
        one_i64(
            &self.conn,
            "SELECT id FROM files WHERE path = ?1",
            params![path],
        )
    }

    pub fn path_of(&self, file_id: i64) -> Result<Option<String>> {
        one_string(
            &self.conn,
            "SELECT path FROM files WHERE id = ?1",
            params![file_id],
        )
    }

    pub fn version_exists(&self, version_id: i64) -> Result<bool> {
        Ok(one_i64(
            &self.conn,
            "SELECT id FROM versions WHERE id = ?1",
            params![version_id],
        )?
        .is_some())
    }

    /// Tulis isi sebagai versi baru. Selalu menambah, tidak pernah menimpa.
    pub fn write(&self, path: &str, content: &str) -> Result<i64> {
        Ok(self.write_version(path, content, None)?.version_id)
    }

    /// Tulis versi baru dengan penunjuk asal (untuk konsolidasi).
    pub fn write_version(
        &self,
        path: &str,
        content: &str,
        source: Option<i64>,
    ) -> Result<WriteOutcome> {
        let path = normalize_path(path)?;
        if content.len() > MAX_CONTENT_BYTES {
            return Err(VirError::Config(format!(
                "isi {} bita melebihi batas {MAX_CONTENT_BYTES}",
                content.len()
            )));
        }
        if let Some(src) = source {
            if !self.version_exists(src)? {
                return Err(VirError::Config(format!("versi asal {src} tidak ada")));
            }
        }
        let hash = hex_sha256(content.as_bytes());
        let now = unix_now();
        self.conn
            .execute_batch("BEGIN IMMEDIATE")
            .map_err(sql_err)?;
        match self.write_version_tx(&path, content, &hash, source, now) {
            Ok(outcome) => {
                self.conn.execute_batch("COMMIT").map_err(sql_err)?;
                Ok(outcome)
            }
            Err(e) => {
                let _ = self.conn.execute_batch("ROLLBACK");
                Err(e)
            }
        }
    }

    fn write_version_tx(
        &self,
        path: &str,
        content: &str,
        hash: &str,
        source: Option<i64>,
        now: i64,
    ) -> Result<WriteOutcome> {
        let file_id = self.ensure_file(path, now)?;
        let supersedes = latest_version(&self.conn, file_id)?;
        self.conn
            .execute(
                "INSERT INTO versions(file_id, content, hash, supersedes, source, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![file_id, content, hash, supersedes, source, now],
            )
            .map_err(sql_err)?;
        let version_id = self.conn.last_insert_rowid();
        self.conn
            .execute(
                "INSERT INTO fts(rowid, content) VALUES (?1, ?2)",
                params![version_id, content],
            )
            .map_err(sql_err)?;
        if let Some(prev) = supersedes {
            self.link(version_id, prev, crate::LinkKind::Supersedes)?;
        }
        let duplicate_of = self.index_and_dedup(version_id, content)?;
        Ok(WriteOutcome {
            version_id,
            supersedes,
            duplicate_of,
        })
    }

    fn ensure_file(&self, path: &str, now: i64) -> Result<i64> {
        let namespace = namespace_of(path);
        self.conn
            .execute(
                "INSERT INTO files(path, namespace, created_at) VALUES (?1, ?2, ?3)
                 ON CONFLICT(path) DO NOTHING",
                params![path, namespace, now],
            )
            .map_err(sql_err)?;
        one_i64(
            &self.conn,
            "SELECT id FROM files WHERE path = ?1",
            params![path],
        )?
        .ok_or_else(|| VirError::Internal(format!("berkas {path} gagal dibuat")))
    }

    /// Baca satu versi. Isi tidak pernah diubah; akses dicatat terpisah.
    pub fn read(&self, version_id: i64) -> Result<String> {
        self.read_version(version_id)
    }

    pub fn read_version(&self, version_id: i64) -> Result<String> {
        let row: Option<(String, String, i64)> = self
            .conn
            .query_row(
                "SELECT content, hash, file_id FROM versions WHERE id = ?1",
                params![version_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .ok();
        let (content, hash, file_id) =
            row.ok_or_else(|| VirError::Config(format!("versi {version_id} tidak ada")))?;
        let actual = hex_sha256(content.as_bytes());
        if actual != hash {
            return Err(VirError::Internal(format!(
                "hash versi {version_id} tidak cocok: {actual} != {hash}"
            )));
        }
        self.conn
            .execute(
                "INSERT INTO access_log(file_id, version_id, context) VALUES (?1, ?2, 'read')",
                params![file_id, version_id],
            )
            .map_err(sql_err)?;
        Ok(content)
    }

    /// Baca versi terbaru sebuah path; gagal bila sudah dilupakan.
    pub fn read_path(&self, path: &str) -> Result<String> {
        let path = normalize_path(path)?;
        let vid = self
            .current_version(&path)?
            .ok_or_else(|| VirError::Config(format!("berkas {path} tidak ada")))?;
        if self.is_forgotten(vid)? {
            return Err(VirError::Config(format!("berkas {path} sudah dilupakan")));
        }
        self.read_version(vid)
    }

    /// Semua versi sebuah path, terlama dulu.
    pub fn history(&self, path: &str) -> Result<Vec<i64>> {
        let path = normalize_path(path)?;
        i64_vec(
            &self.conn,
            "SELECT v.id FROM versions v JOIN files f ON f.id = v.file_id
             WHERE f.path = ?1 ORDER BY v.id",
            params![path],
        )
    }

    /// Hapus lunak satu versi.
    pub fn forget(&self, version_id: i64) -> Result<()> {
        self.set_deleted(version_id, true)
    }

    /// Batalkan hapus lunak.
    pub fn restore(&self, version_id: i64) -> Result<()> {
        self.set_deleted(version_id, false)
    }

    fn set_deleted(&self, version_id: i64, deleted: bool) -> Result<()> {
        if !self.version_exists(version_id)? {
            return Err(VirError::Config(format!("versi {version_id} tidak ada")));
        }
        let n = self
            .conn
            .execute(
                "UPDATE versions SET deleted = ?1 WHERE id = ?2",
                params![i64::from(deleted), version_id],
            )
            .map_err(sql_err)?;
        if n == 0 {
            return Err(VirError::Config(format!(
                "versi {version_id} tidak dapat diubah"
            )));
        }
        Ok(())
    }

    pub fn is_forgotten(&self, version_id: i64) -> Result<bool> {
        Ok(one_i64(
            &self.conn,
            "SELECT deleted FROM versions WHERE id = ?1",
            params![version_id],
        )?
        .unwrap_or(0)
            != 0)
    }

    /// Verifikasi hash isi; `false` bila versi tidak ada atau isi diubah.
    pub fn verify(&self, version_id: i64) -> Result<bool> {
        let Some((content, hash)) = self.raw_version(version_id)? else {
            return Ok(false);
        };
        Ok(hex_sha256(content.as_bytes()) == hash)
    }

    pub fn hash_of(&self, version_id: i64) -> Result<String> {
        let info = self.version_info(version_id)?;
        Ok(info.hash)
    }

    fn raw_version(&self, version_id: i64) -> Result<Option<(String, String)>> {
        Ok(self
            .conn
            .query_row(
                "SELECT content, hash FROM versions WHERE id = ?1",
                params![version_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .ok())
    }
}

fn latest_version(conn: &Connection, file_id: i64) -> Result<Option<i64>> {
    one_i64(
        conn,
        "SELECT id FROM versions WHERE file_id = ?1 ORDER BY id DESC LIMIT 1",
        params![file_id],
    )
}

/// Normalkan path: mulai `/`, tanpa segmen kosong atau `.`; `..` memangkas.
pub fn normalize_path(path: &str) -> Result<String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(VirError::Config("path kosong".into()));
    }
    if trimmed.len() > MAX_PATH_LEN {
        return Err(VirError::Config(format!(
            "path lebih dari {MAX_PATH_LEN} karakter"
        )));
    }
    if trimmed.contains('\0') {
        return Err(VirError::Config("path mengandung NUL".into()));
    }
    let mut parts: Vec<&str> = Vec::new();
    for seg in trimmed.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    if parts.is_empty() {
        return Ok("/".into());
    }
    Ok(format!("/{}", parts.join("/")))
}

/// Ruang nama = segmen pertama path.
pub fn namespace_of(path: &str) -> String {
    let path = normalize_path(path).unwrap_or_else(|_| "/".into());
    let trimmed = path.trim_matches('/');
    match trimmed.split_once('/') {
        Some((first, _)) => first.to_string(),
        None if trimmed.is_empty() => "akar".to_string(),
        None => trimmed.to_string(),
    }
}
