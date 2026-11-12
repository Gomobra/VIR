//! Tautan, tag, log akses, dan penyimpanan vektor (lapisan turunan).

use std::collections::HashMap;

use rusqlite::params;

use crate::db::{i64_vec, one_i64, sql_err};
use crate::store::MemoryStore;
use crate::vec_index::{decode_vector, encode_vector};
use crate::{Result, VirError};

/// Jenis tautan antar versi.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LinkKind {
    Supersedes,
    DuplicateOf,
    DerivedFrom,
    ConsolidatedFrom,
}

impl LinkKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Supersedes => "supersedes",
            Self::DuplicateOf => "duplicate_of",
            Self::DerivedFrom => "derived_from",
            Self::ConsolidatedFrom => "consolidated_from",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "supersedes" => Some(Self::Supersedes),
            "duplicate_of" => Some(Self::DuplicateOf),
            "derived_from" => Some(Self::DerivedFrom),
            "consolidated_from" => Some(Self::ConsolidatedFrom),
            _ => None,
        }
    }
}

impl std::fmt::Display for LinkKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl MemoryStore {
    /// Tambah tag ke berkas (idempoten).
    pub fn tag(&self, file_id: i64, tag: &str) -> Result<()> {
        let tag = tag.trim();
        if tag.is_empty() {
            return Err(VirError::Config("tag kosong".into()));
        }
        if self.path_of(file_id)?.is_none() {
            return Err(VirError::Config(format!("berkas {file_id} tidak ada")));
        }
        self.conn
            .execute(
                "INSERT OR IGNORE INTO tags(file_id, tag) VALUES (?1, ?2)",
                params![file_id, tag],
            )
            .map_err(sql_err)?;
        Ok(())
    }

    /// Hapus tag dari berkas.
    pub fn untag(&self, file_id: i64, tag: &str) -> Result<()> {
        let n = self
            .conn
            .execute(
                "DELETE FROM tags WHERE file_id = ?1 AND tag = ?2",
                params![file_id, tag.trim()],
            )
            .map_err(sql_err)?;
        if n == 0 {
            return Err(VirError::Config(format!(
                "tag {:?} tidak ada pada berkas {file_id}",
                tag.trim()
            )));
        }
        Ok(())
    }

    pub fn tags(&self, file_id: i64) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT tag FROM tags WHERE file_id = ?1 ORDER BY tag")
            .map_err(sql_err)?;
        let rows = stmt
            .query_map(params![file_id], |r| r.get::<_, String>(0))
            .map_err(sql_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(sql_err)?);
        }
        Ok(out)
    }

    /// Id berkas yang bertag tertentu.
    pub fn files_with_tag(&self, tag: &str) -> Result<Vec<i64>> {
        i64_vec(
            &self.conn,
            "SELECT file_id FROM tags WHERE tag = ?1 ORDER BY file_id",
            params![tag.trim()],
        )
    }

    /// Buat tautan; menolak tautan ke diri sendiri.
    pub fn link(&self, from: i64, to: i64, kind: LinkKind) -> Result<()> {
        if from == to {
            return Err(VirError::Config(
                "tautan tidak boleh ke versi sendiri".into(),
            ));
        }
        if !self.version_exists(from)? || !self.version_exists(to)? {
            return Err(VirError::Config(
                "tautan membutuhkan dua versi yang ada".into(),
            ));
        }
        self.conn
            .execute(
                "INSERT OR IGNORE INTO links(from_version, to_version, kind)
                 VALUES (?1, ?2, ?3)",
                params![from, to, kind.as_str()],
            )
            .map_err(sql_err)?;
        Ok(())
    }

    fn read_links(&self, sql: &str, version_id: i64) -> Result<Vec<(i64, LinkKind)>> {
        let mut stmt = self.conn.prepare(sql).map_err(sql_err)?;
        let rows = stmt
            .query_map(params![version_id], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(sql_err)?;
        let mut out = Vec::new();
        for row in rows {
            let (other, kind) = row.map_err(sql_err)?;
            let kind = LinkKind::parse(&kind)
                .ok_or_else(|| VirError::Internal(format!("jenis tautan tak dikenal: {kind}")))?;
            out.push((other, kind));
        }
        Ok(out)
    }

    /// Tautan keluar dari satu versi.
    pub fn links_from(&self, version_id: i64) -> Result<Vec<(i64, LinkKind)>> {
        self.read_links(
            "SELECT to_version, kind FROM links WHERE from_version = ?1
             ORDER BY to_version, kind",
            version_id,
        )
    }

    /// Tautan masuk ke satu versi.
    pub fn links_to(&self, version_id: i64) -> Result<Vec<(i64, LinkKind)>> {
        self.read_links(
            "SELECT from_version, kind FROM links WHERE to_version = ?1
             ORDER BY from_version, kind",
            version_id,
        )
    }

    /// Tetangga graf tautan (dua arah), unik dan terurut.
    pub fn neighbors(&self, version_id: i64) -> Result<Vec<i64>> {
        i64_vec(
            &self.conn,
            "SELECT to_version FROM links WHERE from_version = ?1
             UNION SELECT from_version FROM links WHERE to_version = ?1
             ORDER BY 1",
            params![version_id],
        )
    }

    /// Jumlah akses tercatat untuk satu versi.
    pub fn access_count(&self, version_id: i64) -> Result<u32> {
        let total: Option<i64> = self
            .conn
            .query_row(
                "SELECT SUM(count) FROM access_log WHERE version_id = ?1",
                params![version_id],
                |r| r.get(0),
            )
            .ok();
        Ok(total.unwrap_or(0).max(0) as u32)
    }

    /// Peta jumlah akses semua versi.
    pub fn access_counts(&self) -> Result<HashMap<i64, u32>> {
        let mut stmt = self
            .conn
            .prepare("SELECT version_id, SUM(count) FROM access_log GROUP BY version_id")
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))
            .map_err(sql_err)?;
        let mut out = HashMap::new();
        for row in rows {
            let (id, count) = row.map_err(sql_err)?;
            out.insert(id, count.max(0) as u32);
        }
        Ok(out)
    }

    /// Simpan vektor turunan sebuah versi.
    pub fn store_vector(&self, version_id: i64, vector: &[f32]) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO vectors(version_id, dim, blob) VALUES (?1, ?2, ?3)",
                params![version_id, vector.len() as i64, encode_vector(vector)],
            )
            .map_err(sql_err)?;
        Ok(())
    }

    pub fn vector_of(&self, version_id: i64) -> Result<Option<Vec<f32>>> {
        let blob: Option<Vec<u8>> = self
            .conn
            .query_row(
                "SELECT blob FROM vectors WHERE version_id = ?1",
                params![version_id],
                |r| r.get(0),
            )
            .ok();
        blob.map(|b| decode_vector(&b)).transpose()
    }

    /// Semua vektor tersimpan sebagai `(version_id, vektor)`.
    pub fn vector_records(&self) -> Result<Vec<(i64, Vec<f32>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT version_id, blob FROM vectors ORDER BY version_id")
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))
            .map_err(sql_err)?;
        let mut out = Vec::new();
        for row in rows {
            let (id, blob) = row.map_err(sql_err)?;
            out.push((id, decode_vector(&blob)?));
        }
        Ok(out)
    }

    /// Indeks satu versi bila belum ada; `true` bila baru diindeks.
    pub fn index_version(&self, version_id: i64) -> Result<bool> {
        let Some(embedder) = self.embedder.as_deref() else {
            return Ok(false);
        };
        if one_i64(
            &self.conn,
            "SELECT version_id FROM vectors WHERE version_id = ?1",
            params![version_id],
        )?
        .is_some()
        {
            return Ok(false);
        }
        let Some(content) = self.content_of(version_id)? else {
            return Err(VirError::Config(format!("versi {version_id} tidak ada")));
        };
        let vector = embedder.embed(&content);
        if vector.is_empty() {
            return Ok(false);
        }
        self.store_vector(version_id, &vector)?;
        Ok(true)
    }

    /// Indeks versi yang belum punya vektor; mengembalikan jumlah baru.
    pub fn ensure_vectors(&self) -> Result<usize> {
        let ids: Vec<i64> = {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT id FROM versions
                     WHERE id NOT IN (SELECT version_id FROM vectors) ORDER BY id",
                )
                .map_err(sql_err)?;
            let rows = stmt
                .query_map([], |r| r.get::<_, i64>(0))
                .map_err(sql_err)?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.map_err(sql_err)?);
            }
            out
        };
        let mut count = 0;
        for id in ids {
            if self.index_version(id)? {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Bangun ulang seluruh indeks vektor.
    pub fn reindex(&self) -> Result<usize> {
        let Some(embedder) = self.embedder.as_deref() else {
            return Ok(0);
        };
        self.conn
            .execute_batch("DELETE FROM vectors")
            .map_err(sql_err)?;
        let mut stmt = self
            .conn
            .prepare("SELECT id, content FROM versions ORDER BY id")
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))
            .map_err(sql_err)?;
        let mut items = Vec::new();
        for row in rows {
            items.push(row.map_err(sql_err)?);
        }
        let mut count = 0;
        for (id, content) in items {
            let vector = embedder.embed(&content);
            if vector.is_empty() {
                continue;
            }
            self.store_vector(id, &vector)?;
            count += 1;
        }
        Ok(count)
    }

    pub(crate) fn content_of(&self, version_id: i64) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT content FROM versions WHERE id = ?1",
                params![version_id],
                |r| r.get(0),
            )
            .ok())
    }
}
