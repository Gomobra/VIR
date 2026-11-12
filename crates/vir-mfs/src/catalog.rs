//! Katalog berkas dan metadata versi (lapisan baca).

use rusqlite::params;

use crate::db::sql_err;
use crate::store::{MemoryStore, normalize_path};
use crate::{Result, VirError};

/// Entri berkas dengan versi terbaru.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    pub file_id: i64,
    pub path: String,
    pub namespace: String,
    pub version_id: i64,
    pub hash: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub deleted: bool,
    pub tags: Vec<String>,
}

/// Metadata satu versi (tanpa isi).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionInfo {
    pub version_id: i64,
    pub file_id: i64,
    pub path: String,
    pub hash: String,
    pub content_len: i64,
    pub supersedes: Option<i64>,
    pub source: Option<i64>,
    pub deleted: bool,
    pub created_at: i64,
}

const VERSION_SQL: &str = "SELECT v.id, f.id, f.path, v.hash, length(v.content), v.supersedes,
        v.source, v.deleted, v.created_at
     FROM versions v JOIN files f ON f.id = v.file_id";

impl MemoryStore {
    /// Metadata semua versi sebuah path, terlama dulu.
    pub fn versions(&self, path: &str) -> Result<Vec<VersionInfo>> {
        let path = normalize_path(path)?;
        let mut stmt = self
            .conn
            .prepare(&format!("{VERSION_SQL} WHERE f.path = ?1 ORDER BY v.id"))
            .map_err(sql_err)?;
        let rows = stmt
            .query_map(params![path], map_version)
            .map_err(sql_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(sql_err)?);
        }
        Ok(out)
    }

    /// Metadata satu versi.
    pub fn version_info(&self, version_id: i64) -> Result<VersionInfo> {
        let mut stmt = self
            .conn
            .prepare(&format!("{VERSION_SQL} WHERE v.id = ?1"))
            .map_err(sql_err)?;
        let mut rows = stmt
            .query_map(params![version_id], map_version)
            .map_err(sql_err)?;
        rows.next()
            .transpose()
            .map_err(sql_err)?
            .ok_or_else(|| VirError::Config(format!("versi {version_id} tidak ada")))
    }

    /// Semua berkas dengan versi terbaru (termasuk yang dilupakan).
    pub fn entries(&self) -> Result<Vec<FileEntry>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT f.id, f.path, f.namespace, f.created_at, v.id, v.hash, v.created_at,
                        v.deleted
                 FROM files f
                 JOIN versions v ON v.id =
                     (SELECT id FROM versions WHERE file_id = f.id ORDER BY id DESC LIMIT 1)
                 ORDER BY f.path",
            )
            .map_err(sql_err)?;
        let rows = stmt.query_map([], map_entry).map_err(sql_err)?;
        let mut out = Vec::new();
        for row in rows {
            let mut entry = row.map_err(sql_err)?;
            entry.tags = self.tags(entry.file_id)?;
            out.push(entry);
        }
        Ok(out)
    }
}

fn map_version(r: &rusqlite::Row<'_>) -> rusqlite::Result<VersionInfo> {
    Ok(VersionInfo {
        version_id: r.get(0)?,
        file_id: r.get(1)?,
        path: r.get(2)?,
        hash: r.get(3)?,
        content_len: r.get(4)?,
        supersedes: r.get(5)?,
        source: r.get(6)?,
        deleted: r.get::<_, i64>(7)? != 0,
        created_at: r.get(8)?,
    })
}

fn map_entry(r: &rusqlite::Row<'_>) -> rusqlite::Result<FileEntry> {
    Ok(FileEntry {
        file_id: r.get(0)?,
        path: r.get(1)?,
        namespace: r.get(2)?,
        created_at: r.get(3)?,
        version_id: r.get(4)?,
        hash: r.get(5)?,
        updated_at: r.get(6)?,
        deleted: r.get::<_, i64>(7)? != 0,
        tags: Vec::new(),
    })
}