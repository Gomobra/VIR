//! Opsi pencarian, bentuk hasil, dan filter kueri.

use std::collections::HashSet;

use rusqlite::params_from_iter;
use rusqlite::types::Value;

use crate::Result;
use crate::db::sql_err;
use crate::fusion::FusionConfig;
use crate::store::MemoryStore;

/// Mode pencarian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Hybrid,
    Keyword,
    Vector,
}

/// Opsi pencarian.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub query: String,
    pub limit: usize,
    pub mode: SearchMode,
    pub namespace: Option<String>,
    pub path_prefix: Option<String>,
    pub tags: Vec<String>,
    pub since: Option<i64>,
    pub until: Option<i64>,
    pub include_forgotten: bool,
    pub fusion: FusionConfig,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            query: String::new(),
            limit: 10,
            mode: SearchMode::Hybrid,
            namespace: None,
            path_prefix: None,
            tags: Vec::new(),
            since: None,
            until: None,
            include_forgotten: false,
            fusion: FusionConfig::default(),
        }
    }
}

/// Satu hasil pencarian.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub version_id: i64,
    pub file_id: i64,
    pub path: String,
    pub namespace: String,
    pub hash: String,
    pub created_at: i64,
    pub score: f64,
    pub fts_rank: Option<usize>,
    pub vector_rank: Option<usize>,
    pub accesses: u32,
    pub content: String,
}

/// Kandidat hasil dari tabel (metadata untuk peringkat).
#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub version_id: i64,
    pub file_id: i64,
    pub path: String,
    pub namespace: String,
    pub hash: String,
    pub created_at: i64,
}

impl MemoryStore {
    /// Muat kandidat sesuai filter path/tag/waktu.
    pub(crate) fn candidates(&self, opts: &SearchOptions) -> Result<Vec<Candidate>> {
        let mut clauses = Vec::new();
        let mut params: Vec<Value> = Vec::new();
        push_filters(opts, &mut clauses, &mut params);
        if clauses.is_empty() {
            clauses.push("1 = 1".to_string());
        }
        let sql = format!(
            "SELECT v.id, v.file_id, f.path, f.namespace, v.hash, v.created_at
             FROM versions v JOIN files f ON f.id = v.file_id
             WHERE {} ORDER BY v.id",
            clauses.join(" AND ")
        );
        let mut stmt = self.conn.prepare(&sql).map_err(sql_err)?;
        let rows = stmt
            .query_map(params_from_iter(params), |r| {
                Ok(Candidate {
                    version_id: r.get(0)?,
                    file_id: r.get(1)?,
                    path: r.get(2)?,
                    namespace: r.get(3)?,
                    hash: r.get(4)?,
                    created_at: r.get(5)?,
                })
            })
            .map_err(sql_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(sql_err)?);
        }
        Ok(out)
    }
}

fn push_filters(opts: &SearchOptions, clauses: &mut Vec<String>, params: &mut Vec<Value>) {
    if !opts.include_forgotten {
        clauses.push("v.deleted = 0".to_string());
    }
    if let Some(ns) = &opts.namespace {
        clauses.push("f.namespace = ?".to_string());
        params.push(Value::from(ns.clone()));
    }
    if let Some(prefix) = &opts.path_prefix {
        let prefix = crate::path::normalize_path(prefix).unwrap_or_else(|_| "/".to_string());
        if prefix != "/" {
            clauses.push("(f.path = ? OR f.path LIKE ? ESCAPE '\\')".to_string());
            params.push(Value::from(prefix.clone()));
            params.push(Value::from(escape_like(&format!("{prefix}/"))));
        }
    }
    for tag in &opts.tags {
        clauses
            .push("EXISTS (SELECT 1 FROM tags t WHERE t.file_id = f.id AND t.tag = ?)".to_string());
        params.push(Value::from(tag.trim().to_string()));
    }
    if let Some(since) = opts.since {
        clauses.push("v.created_at >= ?".to_string());
        params.push(Value::from(since));
    }
    if let Some(until) = opts.until {
        clauses.push("v.created_at <= ?".to_string());
        params.push(Value::from(until));
    }
}

fn escape_like(prefix: &str) -> String {
    let mut out = String::new();
    for c in prefix.chars() {
        if c == '%' || c == '_' || c == '\\' {
            out.push('\\');
        }
        out.push(c);
    }
    out.push('%');
    out
}

/// Ekspresi FTS5 aman: token alfanumerik dikutip, prefiks, digabung OR.
pub(crate) fn match_expr(query: &str) -> String {
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .take(16)
        .map(|t| format!("\"{}\"*", t.to_lowercase()))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// Himpunan id kandidat.
pub(crate) fn keep_set(candidates: &[Candidate]) -> HashSet<i64> {
    candidates.iter().map(|c| c.version_id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ekspresi_fts_aman() {
        assert_eq!(match_expr("rust-"), "\"rust\"*");
        assert!(match_expr("   ").is_empty());
        assert!(match_expr("\" OR MATCH").contains("or"));
        assert!(match_expr("a b").contains(" OR "));
    }
}
