//! Skema SQLite (WAL) dan helper query untuk `vir-mfs`.
//!
//! Aturan isi tak berubah: baris `versions` hanya pernah `INSERT`; `forget`
//! memasang penanda `deleted`, `restore` melepasnya. Tidak ada `UPDATE`/`DELETE`
//! pada kolom `content` di mana pun di crate ini.

use rusqlite::{Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use vir_common::VirError;

use crate::Result;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    namespace TEXT NOT NULL DEFAULT 'semantik',
    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE TABLE IF NOT EXISTS versions (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    hash TEXT NOT NULL,
    supersedes INTEGER REFERENCES versions(id),
    source INTEGER REFERENCES versions(id),
    deleted INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE TABLE IF NOT EXISTS tags (
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (file_id, tag)
);
CREATE TABLE IF NOT EXISTS links (
    from_version INTEGER NOT NULL REFERENCES versions(id) ON DELETE CASCADE,
    to_version INTEGER NOT NULL REFERENCES versions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    PRIMARY KEY (from_version, to_version, kind)
);
CREATE TABLE IF NOT EXISTS access_log (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    version_id INTEGER REFERENCES versions(id) ON DELETE CASCADE,
    context TEXT NOT NULL DEFAULT 'read',
    count INTEGER NOT NULL DEFAULT 1,
    accessed_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE TABLE IF NOT EXISTS vectors (
    version_id INTEGER PRIMARY KEY REFERENCES versions(id) ON DELETE CASCADE,
    dim INTEGER NOT NULL,
    blob BLOB NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(content);
"#;

const INDEXES: &str = r#"
CREATE INDEX IF NOT EXISTS idx_versions_file ON versions(file_id, id);
CREATE INDEX IF NOT EXISTS idx_versions_created ON versions(created_at);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
CREATE INDEX IF NOT EXISTS idx_links_from ON links(from_version);
CREATE INDEX IF NOT EXISTS idx_links_to ON links(to_version);
CREATE INDEX IF NOT EXISTS idx_access_version ON access_log(version_id);
CREATE INDEX IF NOT EXISTS idx_access_file ON access_log(file_id);
"#;

/// Galat SQLite menjadi [`VirError::Internal`].
pub fn sql_err(e: rusqlite::Error) -> VirError {
    VirError::Internal(format!("sqlite: {e}"))
}

/// Buka basis data berkas dengan WAL dan foreign key aktif.
pub fn open_connection(path: &str) -> Result<Connection> {
    let conn = Connection::open(path).map_err(sql_err)?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA synchronous=NORMAL;",
    )
    .map_err(sql_err)?;
    Ok(conn)
}

/// Buka basis data in-memory (untuk tes dan sesi sekali pakai).
pub fn open_memory() -> Result<Connection> {
    let conn = Connection::open_in_memory().map_err(sql_err)?;
    conn.execute_batch("PRAGMA foreign_keys=ON;")
        .map_err(sql_err)?;
    Ok(conn)
}

/// Buat skema bila belum ada, lalu tambahkan kolom yang hilang.
pub fn init_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(SCHEMA).map_err(sql_err)?;
    conn.execute_batch(INDEXES).map_err(sql_err)?;
    add_missing_columns(conn)
}

fn add_missing_columns(conn: &Connection) -> Result<()> {
    let missing: [(&str, &str, &str); 3] = [
        ("versions", "source", "INTEGER"),
        ("versions", "deleted", "INTEGER NOT NULL DEFAULT 0"),
        ("files", "namespace", "TEXT NOT NULL DEFAULT 'semantik'"),
    ];
    for (table, column, decl) in missing {
        if !has_column(conn, table, column)? {
            let sql = format!("ALTER TABLE {table} ADD COLUMN {column} {decl}");
            conn.execute_batch(&sql).map_err(sql_err)?;
        }
    }
    Ok(())
}

fn has_column(conn: &Connection, table: &str, column: &str) -> Result<bool> {
    let sql = format!("PRAGMA table_info({table})");
    let mut stmt = conn.prepare(&sql).map_err(sql_err)?;
    let rows = stmt
        .query_map([], |r| r.get::<_, String>(1))
        .map_err(sql_err)?;
    for row in rows {
        if row.map_err(sql_err)? == column {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn one_i64(conn: &Connection, sql: &str, params: impl rusqlite::Params) -> Result<Option<i64>> {
    conn.query_row(sql, params, |r| r.get::<_, i64>(0))
        .optional()
        .map_err(sql_err)
}

pub fn one_string(
    conn: &Connection,
    sql: &str,
    params: impl rusqlite::Params,
) -> Result<Option<String>> {
    conn.query_row(sql, params, |r| r.get::<_, String>(0))
        .optional()
        .map_err(sql_err)
}

pub fn i64_vec(conn: &Connection, sql: &str, params: impl rusqlite::Params) -> Result<Vec<i64>> {
    let mut stmt = conn.prepare(sql).map_err(sql_err)?;
    let rows = stmt
        .query_map(params, |r| r.get::<_, i64>(0))
        .map_err(sql_err)?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(sql_err)?);
    }
    Ok(out)
}

/// Detik Unix saat ini (UTC).
pub fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// SHA-256 dalam heksadesimal.
pub fn hex_sha256(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrasi_menambah_kolom_yang_hilang() {
        let conn = open_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);
             CREATE TABLE versions (
                 id INTEGER PRIMARY KEY,
                 file_id INTEGER NOT NULL,
                 content TEXT NOT NULL,
                 hash TEXT NOT NULL,
                 supersedes INTEGER,
                 created_at INTEGER NOT NULL DEFAULT 0);",
        )
        .unwrap();
        add_missing_columns(&conn).unwrap();
        assert!(has_column(&conn, "versions", "source").unwrap());
        assert!(has_column(&conn, "versions", "deleted").unwrap());
        assert!(has_column(&conn, "files", "namespace").unwrap());
        assert!(!has_column(&conn, "versions", "tidak_ada").unwrap());
    }

    #[test]
    fn hash_sha256_diketahui() {
        assert_eq!(
            hex_sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
