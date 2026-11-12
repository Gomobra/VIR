//! Pencarian hibrida: BM25 (FTS5) + kosinus brute-force, digabung RRF.

use std::collections::{HashMap, HashSet};

use rusqlite::params_from_iter;
use rusqlite::types::Value;

use crate::Result;
use crate::db::{sql_err, unix_now};
use crate::fusion::{FusionConfig, rank_score, rrf_fuse};
use crate::store::MemoryStore;
use crate::vec_index::{cosine, decode_vector};

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

#[derive(Debug, Clone)]
struct Candidate {
    version_id: i64,
    file_id: i64,
    path: String,
    namespace: String,
    hash: String,
    content: String,
    created_at: i64,
}

impl MemoryStore {
    /// Cari dengan opsi lengkap.
    pub fn search(&self, opts: &SearchOptions) -> Result<Vec<SearchHit>> {
        if opts.limit == 0 || opts.query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let candidates = self.candidates(opts)?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let fts = self.fts_ranked(opts, &candidates)?;
        let vectors = self.vector_ranked(opts, &candidates)?;
        let fts_ids: Vec<i64> = fts.iter().map(|(id, _)| *id).collect();
        let vec_ids: Vec<i64> = vectors.iter().map(|(id, _)| *id).collect();
        let mut lists: Vec<(&[i64], f64)> = Vec::new();
        if matches!(opts.mode, SearchMode::Hybrid | SearchMode::Keyword) && !fts_ids.is_empty() {
            lists.push((fts_ids.as_slice(), opts.fusion.fts_weight));
        }
        if matches!(opts.mode, SearchMode::Hybrid | SearchMode::Vector) && !vec_ids.is_empty() {
            lists.push((vec_ids.as_slice(), opts.fusion.vector_weight));
        }
        let base = rrf_fuse(&lists, opts.fusion.k);
        if base.is_empty() {
            return Ok(Vec::new());
        }
        let accesses = self.access_counts()?;
        let now = unix_now();
        let mut scored: Vec<(i64, f64)> = base
            .into_iter()
            .map(|(id, base)| {
                let cand = candidates.iter().find(|c| c.version_id == id);
                let (created_at, count) = match cand {
                    Some(c) => (c.created_at, accesses.get(&id).copied().unwrap_or(0)),
                    None => (now, 0),
                };
                let age_days = ((now - created_at).max(0) as f64) / 86_400.0;
                (id, rank_score(base, age_days, count, &opts.fusion))
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.truncate(opts.limit);
        let fts_rank: HashMap<i64, usize> = fts
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (*id, i))
            .collect();
        let vec_rank: HashMap<i64, usize> = vectors
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (*id, i))
            .collect();
        let mut hits = Vec::new();
        for (id, score) in scored {
            if let Some(c) = candidates.iter().find(|c| c.version_id == id) {
                hits.push(SearchHit {
                    version_id: c.version_id,
                    file_id: c.file_id,
                    path: c.path.clone(),
                    namespace: c.namespace.clone(),
                    hash: c.hash.clone(),
                    created_at: c.created_at,
                    score,
                    fts_rank: fts_rank.get(&id).copied(),
                    vector_rank: vec_rank.get(&id).copied(),
                    accesses: accesses.get(&id).copied().unwrap_or(0),
                    content: c.content.clone(),
                });
            }
        }
        Ok(hits)
    }

    /// Pencarian kata kunci saja.
    pub fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let opts = SearchOptions {
            query: query.to_string(),
            limit,
            mode: SearchMode::Keyword,
            ..SearchOptions::default()
        };
        self.search(&opts)
    }

    /// Pencarian vektor saja.
    pub fn search_vector(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let opts = SearchOptions {
            query: query.to_string(),
            limit,
            mode: SearchMode::Vector,
            ..SearchOptions::default()
        };
        self.search(&opts)
    }

    /// Pencarian bawaan untuk CLI/binding: hibrida, `(versi, isi, skor)`.
    pub fn find(&self, query: &str, limit: usize) -> Result<Vec<(i64, String, f64)>> {
        let opts = SearchOptions {
            query: query.to_string(),
            limit,
            ..SearchOptions::default()
        };
        Ok(self
            .search(&opts)?
            .into_iter()
            .map(|h| (h.version_id, h.content, h.score))
            .collect())
    }

    fn candidates(&self, opts: &SearchOptions) -> Result<Vec<Candidate>> {
        let mut clauses = Vec::new();
        let mut params: Vec<Value> = Vec::new();
        push_filters(opts, &mut clauses, &mut params);
        if clauses.is_empty() {
            clauses.push("1 = 1".to_string());
        }
        let sql = format!(
            "SELECT v.id, v.file_id, f.path, f.namespace, v.hash, v.content, v.created_at
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
                    content: r.get(5)?,
                    created_at: r.get(6)?,
                })
            })
            .map_err(sql_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(sql_err)?);
        }
        Ok(out)
    }

    fn fts_ranked(
        &self,
        opts: &SearchOptions,
        candidates: &[Candidate],
    ) -> Result<Vec<(i64, f64)>> {
        let expr = match_expr(&opts.query);
        if expr.is_empty() {
            return Ok(Vec::new());
        }
        let keep: HashSet<i64> = candidates.iter().map(|c| c.version_id).collect();
        let mut stmt = self
            .conn
            .prepare(
                "SELECT rowid, bm25(fts) FROM fts WHERE fts MATCH ?1
                 ORDER BY bm25(fts) LIMIT ?2",
            )
            .map_err(sql_err)?;
        let limit = (opts.limit.saturating_mul(20)).max(50) as i64;
        let rows = stmt
            .query_map(
                params_from_iter(vec![Value::from(expr), Value::from(limit)]),
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?)),
            )
            .map_err(sql_err)?;
        let mut out = Vec::new();
        for row in rows {
            let (id, score) = row.map_err(sql_err)?;
            if keep.contains(&id) {
                out.push((id, score));
            }
        }
        Ok(out)
    }

    fn vector_ranked(
        &self,
        opts: &SearchOptions,
        candidates: &[Candidate],
    ) -> Result<Vec<(i64, f64)>> {
        let Some(embedder) = self.embedder.as_deref() else {
            return Ok(Vec::new());
        };
        let query = embedder.embed(&opts.query);
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let keep: HashSet<i64> = candidates.iter().map(|c| c.version_id).collect();
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
            if !keep.contains(&id) {
                continue;
            }
            let vector = decode_vector(&blob)?;
            let score = f64::from(cosine(&query, &vector));
            if score > 0.0 {
                out.push((id, score));
            }
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
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
        let prefix = crate::store::normalize_path(prefix).unwrap_or_else(|_| "/".to_string());
        clauses.push("f.path LIKE ? ESCAPE '\\'".to_string());
        params.push(Value::from(escape_like(&prefix)));
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

fn match_expr(query: &str) -> String {
    let tokens: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .take(16)
        .map(|t| format!("\"{}\"*", t.to_lowercase()))
        .collect();
    tokens.join(" OR ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with_docs() -> crate::MemoryStore {
        let s = crate::MemoryStore::open_in_memory().unwrap();
        s.write("/episodik/e1", "rust memori kepemilikan data")
            .unwrap();
        s.write("/episodik/e2", "python sains data numpy").unwrap();
        s.write("/semantik/s1", "catatan arsitektur rust").unwrap();
        s
    }

    #[test]
    fn kata_kunci_menemukan() {
        let s = store_with_docs();
        let hits = s.search_fts("rust", 5).unwrap();
        assert!(!hits.is_empty());
        assert!(hits.iter().all(|h| h.content.contains("rust")));
    }

    #[test]
    fn vektor_menemukan() {
        let s = store_with_docs();
        let hits = s.search_vector("rust", 5).unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].content.contains("rust"));
    }

    #[test]
    fn filter_tag_dan_path() {
        let s = store_with_docs();
        let e1 = s.file_id("/episodik/e1").unwrap().unwrap();
        s.tag(e1, "penting").unwrap();
        let mut opts = SearchOptions {
            query: "data".into(),
            limit: 10,
            tags: vec!["penting".into()],
            ..SearchOptions::default()
        };
        let hits = s.search(&opts).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "/episodik/e1");
        opts.tags.clear();
        opts.path_prefix = Some("/episodik".into());
        assert!(
            s.search(&opts)
                .unwrap()
                .iter()
                .all(|h| h.path.starts_with("/episodik"))
        );
    }

    #[test]
    fn lupa_menghilangkan_dari_aksi() {
        let s = store_with_docs();
        let vid = s.current_version("/episodik/e1").unwrap().unwrap();
        s.forget(vid).unwrap();
        assert!(
            s.search_fts("rust", 5)
                .unwrap()
                .iter()
                .all(|h| h.version_id != vid)
        );
        let mut opts = SearchOptions {
            query: "rust".into(),
            limit: 5,
            include_forgotten: true,
            ..SearchOptions::default()
        };
        assert!(s.search(&opts).unwrap().iter().any(|h| h.version_id == vid));
        opts.include_forgotten = false;
        assert!(!s.search(&opts).unwrap().iter().any(|h| h.version_id == vid));
    }

    #[test]
    fn batas_waktu() {
        let s = store_with_docs();
        let mut opts = SearchOptions {
            query: "data".into(),
            limit: 5,
            since: Some(crate::db::unix_now() + 3600),
            ..SearchOptions::default()
        };
        assert!(s.search(&opts).unwrap().is_empty());
        opts.since = Some(0);
        opts.until = Some(0);
        assert!(s.search(&opts).unwrap().is_empty());
    }

    #[test]
    fn ekspresi_fts_aman() {
        assert_eq!(match_expr("rust-"), "\"rust\"*");
        assert!(match_expr("   ").is_empty());
        assert!(match_expr("\" OR MATCH").contains("or"));
    }
}
