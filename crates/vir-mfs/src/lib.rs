//! Memori file manager (MFS).

use rusqlite::{Connection, params};
use sha2::{Digest, Sha256};
use vir_common::VirError;

fn sql_err(e: rusqlite::Error) -> VirError {
    VirError::Internal(e.to_string())
}

pub struct MemoryStore {
    conn: Connection,
}

impl MemoryStore {
    pub fn open(path: &str) -> Result<Self, VirError> {
        let conn = Connection::open(path).map_err(sql_err)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .map_err(sql_err)?;
        let s = Self { conn };
        s.init_schema()?;
        Ok(s)
    }

    pub fn open_in_memory() -> Result<Self, VirError> {
        let conn = Connection::open_in_memory().map_err(sql_err)?;
        let s = Self { conn };
        s.init_schema()?;
        Ok(s)
    }

    fn init_schema(&self) -> Result<(), VirError> {
        self.conn
            .execute_batch(
                r#"
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                namespace TEXT NOT NULL DEFAULT 'semantik',
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS versions (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id),
                content TEXT NOT NULL,
                hash TEXT NOT NULL,
                supersedes INTEGER,
                deleted INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS tags (
                file_id INTEGER NOT NULL REFERENCES files(id),
                tag TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id),
                context TEXT NOT NULL DEFAULT '',
                count INTEGER NOT NULL DEFAULT 1,
                accessed_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(content, content_rowid='rowid');
            "#,
            )
            .map_err(sql_err)?;
        Ok(())
    }

    pub fn write(&self, path: &str, content: &str) -> Result<i64, VirError> {
        self.conn
            .execute(
                "INSERT INTO files(path) VALUES (?1) ON CONFLICT(path) DO NOTHING",
                params![path],
            )
            .map_err(sql_err)?;
        let file_id: i64 = self
            .conn
            .query_row("SELECT id FROM files WHERE path=?1", params![path], |r| {
                r.get(0)
            })
            .map_err(sql_err)?;
        let prev: Option<i64> = self
            .conn
            .query_row(
                "SELECT id FROM versions WHERE file_id=?1 ORDER BY id DESC LIMIT 1",
                params![file_id],
                |r| r.get(0),
            )
            .ok();
        let hash = hex_sha256(content.as_bytes());
        self.conn
            .execute(
                "INSERT INTO versions(file_id, content, hash, supersedes) VALUES (?1,?2,?3,?4)",
                params![file_id, content, hash, prev],
            )
            .map_err(sql_err)?;
        let vid = self.conn.last_insert_rowid();
        self.conn
            .execute(
                "INSERT INTO fts(rowid, content) VALUES (?1, ?2)",
                params![vid, content],
            )
            .map_err(sql_err)?;
        Ok(vid)
    }

    pub fn read(&self, version_id: i64) -> Result<String, VirError> {
        let (content, hash): (String, String) = self
            .conn
            .query_row(
                "SELECT content, hash FROM versions WHERE id=?1",
                params![version_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(sql_err)?;
        if hex_sha256(content.as_bytes()) != hash {
            return Err(VirError::Internal("hash tidak cocok".into()));
        }
        let file_id: i64 = self
            .conn
            .query_row(
                "SELECT file_id FROM versions WHERE id=?1",
                params![version_id],
                |r| r.get(0),
            )
            .map_err(sql_err)?;
        self.conn
            .execute(
                "INSERT INTO access_log(file_id, context) VALUES (?1, 'read')",
                params![file_id],
            )
            .map_err(sql_err)?;
        Ok(content)
    }

    pub fn find(&self, query: &str, limit: usize) -> Result<Vec<(i64, String, f64)>, VirError> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT v.id, v.content, bm25(fts) AS score FROM fts JOIN versions v ON v.id=fts.rowid WHERE fts MATCH ?1 ORDER BY score LIMIT ?2",
            )
            .map_err(sql_err)?;
        let rows = stmt
            .query_map(params![query, limit as i64], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, f64>(2)?,
                ))
            })
            .map_err(sql_err)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(sql_err)?);
        }
        Ok(out)
    }

    pub fn history(&self, path: &str) -> Result<Vec<i64>, VirError> {
        let file_id: i64 = self
            .conn
            .query_row("SELECT id FROM files WHERE path=?1", params![path], |r| {
                r.get(0)
            })
            .map_err(sql_err)?;
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM versions WHERE file_id=?1 ORDER BY id")
            .map_err(sql_err)?;
        let rows = stmt
            .query_map(params![file_id], |r| r.get::<_, i64>(0))
            .map_err(sql_err)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(sql_err)?);
        }
        Ok(out)
    }

    pub fn forget(&self, version_id: i64) -> Result<(), VirError> {
        self.conn
            .execute(
                "UPDATE versions SET deleted=1 WHERE id=?1",
                params![version_id],
            )
            .map_err(sql_err)?;
        Ok(())
    }

    pub fn restore(&self, version_id: i64) -> Result<(), VirError> {
        self.conn
            .execute(
                "UPDATE versions SET deleted=0 WHERE id=?1",
                params![version_id],
            )
            .map_err(sql_err)?;
        Ok(())
    }
}

fn hex_sha256(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

pub fn rrf_fuse(ranks: &[Vec<usize>], k: f64) -> Vec<(usize, f64)> {
    use std::collections::BTreeMap;
    let mut scores: BTreeMap<usize, f64> = BTreeMap::new();
    for list in ranks {
        for (rank, &doc) in list.iter().enumerate() {
            let s = 1.0 / (k + rank as f64 + 1.0);
            *scores.entry(doc).or_insert(0.0) += s;
        }
    }
    let mut v: Vec<(usize, f64)> = scores.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_read_hash() {
        let s = MemoryStore::open_in_memory().unwrap();
        let vid = s.write("/a.txt", "hello world").unwrap();
        let content = s.read(vid).unwrap();
        assert_eq!(content, "hello world");
    }

    #[test]
    fn versioning_keeps_old() {
        let s = MemoryStore::open_in_memory().unwrap();
        let v1 = s.write("/a.txt", "v1").unwrap();
        let v2 = s.write("/a.txt", "v2").unwrap();
        assert_eq!(s.read(v1).unwrap(), "v1");
        assert_eq!(s.read(v2).unwrap(), "v2");
        let h = s.history("/a.txt").unwrap();
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn read_does_not_modify_content() {
        let s = MemoryStore::open_in_memory().unwrap();
        let vid = s.write("/a.txt", "immutable").unwrap();
        let before = s.read(vid).unwrap();
        let after = s.read(vid).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn find_fts() {
        let s = MemoryStore::open_in_memory().unwrap();
        s.write("/a.txt", "rust programming language").unwrap();
        s.write("/b.txt", "python data science").unwrap();
        let res = s.find("rust", 5).unwrap();
        assert!(!res.is_empty());
        assert!(res[0].1.contains("rust"));
    }

    #[test]
    fn forget_restore() {
        let s = MemoryStore::open_in_memory().unwrap();
        let vid = s.write("/a.txt", "x").unwrap();
        s.forget(vid).unwrap();
        s.restore(vid).unwrap();
        assert_eq!(s.read(vid).unwrap(), "x");
    }

    #[test]
    fn rrf_fuse_basic() {
        let ranks = vec![vec![0usize, 1], vec![1usize, 2]];
        let fused = rrf_fuse(&ranks, 60.0);
        assert_eq!(fused[0].0, 1);
    }
}
