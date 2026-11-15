//! Jembatan `mem`: operasi MFS nyata untuk RLM (F3.4).
//!
//! Kode Python di REPL memanggil `mem(op, ...)`; host menerjemahkannya menjadi
//! operasi `vir-mfs` pada basis data yang dibuka sekali di awal. Tidak ada
//! jalur yang mengembalikan `null` diam-diam: operasi tak dikenal atau berkas
//! yang tidak ada menjadi galat yang diteruskan ke kode Python.

use std::path::{Path, PathBuf};

use serde_json::{Value as JsonValue, json};
use vir_common::VirError;
use vir_mfs::{MemoryStore, ReadTarget};

/// Operasi memori yang didukung.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemOp {
    /// Daftar berkas di bawah prefiks path.
    Ls { prefix: String },
    /// Baca satu versi (id versi) atau versi terbaru sebuah path.
    Read { target: String },
    /// Tulis versi baru pada path.
    Write { path: String, content: String },
    /// Pencarian hibrida (FTS + vektor).
    Find { query: String, limit: usize },
}

/// Jembatan MFS: memegang toko dan menerjemahkan operasi.
pub struct MemBridge {
    store: MemoryStore,
    origin: PathBuf,
}

impl MemBridge {
    /// Buka toko MFS di `path`.
    pub fn open(path: &Path) -> Result<Self, VirError> {
        let Some(text) = path.to_str() else {
            return Err(VirError::Config(format!(
                "jalur memori tidak valid: {}",
                path.display()
            )));
        };
        Ok(Self {
            store: MemoryStore::open(text)?,
            origin: path.to_path_buf(),
        })
    }

    /// Buka toko MFS di memori (untuk tes).
    pub fn open_in_memory() -> Result<Self, VirError> {
        Ok(Self {
            store: MemoryStore::open_in_memory()?,
            origin: PathBuf::from(":memory:"),
        })
    }

    /// Tulis satu berkas pada path yang diberikan; kembalikan id versi.
    pub fn write_file(&self, path: &str, content: &str) -> Result<i64, VirError> {
        Ok(vir_mfs::write(&self.store, path, content)?.version_id)
    }

    /// Baca satu versi (dipakai tes jembatan).
    pub fn read_version(&self, version_id: i64) -> Result<String, VirError> {
        Ok(vir_mfs::read(&self.store, ReadTarget::Version(version_id))?.content)
    }

    /// Path basis data yang dibuka.
    pub fn origin(&self) -> &Path {
        &self.origin
    }

    /// Terjemahkan satu permintaan RPC `mem` menjadi operasi nyata.
    pub fn call(&self, payload: &JsonValue) -> Result<JsonValue, VirError> {
        let op = self.parse(payload)?;
        self.execute(&op)
    }
}

impl MemBridge {
    fn parse(&self, payload: &JsonValue) -> Result<MemOp, VirError> {
        let name = field(payload, "op", true)?;
        match name.as_str() {
            "ls" => Ok(MemOp::Ls {
                prefix: opt_string(payload, "path")?.unwrap_or_else(|| "/".to_string()),
            }),
            "read" => Ok(MemOp::Read {
                target: opt_string(payload, "path")?.unwrap_or_default(),
            }),
            "write" => Ok(MemOp::Write {
                path: opt_string(payload, "path")?.unwrap_or_else(|| "/rlm".to_string()),
                content: opt_string(payload, "content")?.unwrap_or_default(),
            }),
            "find" => Ok(MemOp::Find {
                query: opt_string(payload, "query")?.unwrap_or_default(),
                limit: payload
                    .get("limit")
                    .and_then(JsonValue::as_u64)
                    .unwrap_or(5) as usize,
            }),
            other => Err(VirError::Config(format!(
                "operasi mem tidak dikenal: {other}"
            ))),
        }
    }

    fn execute(&self, op: &MemOp) -> Result<JsonValue, VirError> {
        match op {
            MemOp::Ls { prefix } => {
                let entries = vir_mfs::ls(&self.store, prefix)?;
                Ok(json!(
                    entries
                        .iter()
                        .map(|e| json!({ "path": e.path, "version_id": e.version_id }))
                        .collect::<Vec<_>>()
                ))
            }
            MemOp::Read { target } => {
                let read = vir_mfs::read(&self.store, self.resolve_target(target)?)?;
                Ok(json!({
                    "version_id": read.version_id,
                    "path": read.path,
                    "hash": read.hash,
                    "content": read.content,
                }))
            }
            MemOp::Write { path, content } => {
                let outcome = vir_mfs::write(&self.store, path, content)?;
                Ok(json!({
                    "version_id": outcome.version_id,
                    "supersedes": outcome.supersedes,
                }))
            }
            MemOp::Find { query, limit } => {
                let hits = vir_mfs::find(&self.store, query, *limit)?;
                Ok(json!(
                    hits.iter()
                        .map(|h| {
                            json!({
                                "version_id": h.version_id,
                                "path": h.path,
                                "score": h.score,
                                "content": h.content,
                            })
                        })
                        .collect::<Vec<_>>()
                ))
            }
        }
    }

    /// Terjemahkan target baca: bilangan bulat → id versi, selain itu path.
    fn resolve_target<'a>(&self, target: &'a str) -> Result<ReadTarget<'a>, VirError> {
        if target.trim().is_empty() {
            return Err(VirError::Config("operasi read butuh path atau id".into()));
        }
        if let Ok(id) = target.trim().parse::<i64>() {
            return Ok(ReadTarget::Version(id));
        }
        Ok(ReadTarget::Path(target))
    }
}

/// Ambil medan string wajib dari payload.
fn field(payload: &JsonValue, key: &str, required: bool) -> Result<String, VirError> {
    match opt_string(payload, key)? {
        Some(value) => Ok(value),
        None if required => Err(VirError::Config(format!(
            "medan '{key}' wajib berupa string"
        ))),
        None => Ok(String::new()),
    }
}

/// Ambil medan string opsional; galat bila ada tetapi bukan string.
fn opt_string(payload: &JsonValue, key: &str) -> Result<Option<String>, VirError> {
    match payload.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(VirError::Config(format!(
            "medan '{key}' harus berupa string"
        ))),
    }
}
