//! Pencarian hibrida: BM25 (FTS5) + kosinus brute-force, digabung RRF.

use std::collections::{HashMap, HashSet};

use rusqlite::params_from_iter;
use rusqlite::types::Value;

use crate::Result;
use crate::db::{sql_err, unix_now};
use crate::fusion::{rank_score, rrf_fuse};
use crate::query::{Candidate, SearchHit, SearchMode, SearchOptions, keep_set, match_expr};
use crate::store::MemoryStore;
use crate::vec_index::{cosine, decode_vector};

/// Hasil peringkat dua daftar sebelum faktor usia/akses.
#[derive(Debug, Default)]
struct Ranked {
    base: Vec<(i64, f64)>,
    fts: Vec<(i64, f64)>,
    vectors: Vec<(i64, f64)>,
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
        let ranked = self.rank(opts, &candidates)?;
        if ranked.base.is_empty() {
            return Ok(Vec::new());
        }
        let accesses = self.access_counts()?;
        let scored = score_hits(&ranked.base, &candidates, &accesses, opts);
        let mut hits = build_hits(
            &scored,
            &candidates,
            &ranked.fts,
            &ranked.vectors,
            &accesses,
        );
        for hit in &mut hits {
            hit.content = self.content_of(hit.version_id)?.unwrap_or_default();
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

    fn rank(&self, opts: &SearchOptions, candidates: &[Candidate]) -> Result<Ranked> {
        let fts = if matches!(opts.mode, SearchMode::Hybrid | SearchMode::Keyword) {
            self.fts_ranked(opts, candidates)?
        } else {
            Vec::new()
        };
        let vectors = if matches!(opts.mode, SearchMode::Hybrid | SearchMode::Vector) {
            self.vector_ranked(opts, candidates)?
        } else {
            Vec::new()
        };
        let fts_ids: Vec<i64> = fts.iter().map(|(id, _)| *id).collect();
        let vec_ids: Vec<i64> = vectors.iter().map(|(id, _)| *id).collect();
        let mut lists: Vec<(&[i64], f64)> = Vec::new();
        if matches!(opts.mode, SearchMode::Hybrid | SearchMode::Keyword) && !fts_ids.is_empty() {
            lists.push((fts_ids.as_slice(), opts.fusion.fts_weight));
        }
        if matches!(opts.mode, SearchMode::Hybrid | SearchMode::Vector) && !vec_ids.is_empty() {
            lists.push((vec_ids.as_slice(), opts.fusion.vector_weight));
        }
        Ok(Ranked {
            base: rrf_fuse(&lists, opts.fusion.k),
            fts,
            vectors,
        })
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
        let keep: HashSet<i64> = keep_set(candidates);
        let mut stmt = self
            .conn
            .prepare(
                "SELECT rowid, bm25(fts) FROM fts WHERE fts MATCH ?1
                 ORDER BY bm25(fts) LIMIT ?2",
            )
            .map_err(sql_err)?;
        let limit = opts.limit.saturating_mul(20).max(50) as i64;
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
        let keep: HashSet<i64> = keep_set(candidates);
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

fn score_hits(
    base: &[(i64, f64)],
    candidates: &[Candidate],
    accesses: &HashMap<i64, u32>,
    opts: &SearchOptions,
) -> Vec<(i64, f64)> {
    let now = unix_now();
    let mut scored: Vec<(i64, f64)> = base
        .iter()
        .map(|(id, base)| {
            let (created, count) = candidate_meta(candidates, *id, accesses, now);
            let age_days = ((now - created).max(0) as f64) / 86_400.0;
            (*id, rank_score(*base, age_days, count, &opts.fusion))
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.truncate(opts.limit);
    scored
}

fn candidate_meta(
    candidates: &[Candidate],
    id: i64,
    accesses: &HashMap<i64, u32>,
    now: i64,
) -> (i64, u32) {
    match candidates.iter().find(|c| c.version_id == id) {
        Some(c) => (c.created_at, accesses.get(&id).copied().unwrap_or(0)),
        None => (now, 0),
    }
}

fn build_hits(
    scored: &[(i64, f64)],
    candidates: &[Candidate],
    fts: &[(i64, f64)],
    vectors: &[(i64, f64)],
    accesses: &HashMap<i64, u32>,
) -> Vec<SearchHit> {
    let fts_rank: HashMap<i64, usize> = rank_map(fts);
    let vec_rank: HashMap<i64, usize> = rank_map(vectors);
    let mut hits = Vec::new();
    for (id, score) in scored {
        if let Some(c) = candidates.iter().find(|c| c.version_id == *id) {
            hits.push(SearchHit {
                version_id: c.version_id,
                file_id: c.file_id,
                path: c.path.clone(),
                namespace: c.namespace.clone(),
                hash: c.hash.clone(),
                created_at: c.created_at,
                score: *score,
                fts_rank: fts_rank.get(id).copied(),
                vector_rank: vec_rank.get(id).copied(),
                accesses: accesses.get(id).copied().unwrap_or(0),
                content: String::new(),
            });
        }
    }
    hits
}

fn rank_map(list: &[(i64, f64)]) -> HashMap<i64, usize> {
    list.iter()
        .enumerate()
        .map(|(i, (id, _))| (*id, i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with_docs() -> crate::MemoryStore {
        let s = crate::MemoryStore::open_in_memory().unwrap();
        s.write("/episodik/e1", "rust memori kepemilikan data")
            .unwrap();
        s.write("/episodik/e2", "python sains data numpy").unwrap();
        s.write("/episodikx/e3", "data tambahan").unwrap();
        s.write("/semantik/s1", "catatan arsitektur rust").unwrap();
        s.write("/semantik/s2", "catatan episodik tersimpan")
            .unwrap();
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
        let hits = s.search(&opts).unwrap();
        assert!(hits.iter().all(|h| h.path.starts_with("/episodik")));
        assert!(!hits.iter().any(|h| h.path.starts_with("/episodikx")));
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
    fn vektor_menolong_saat_token_tidak_cocok() {
        let s = store_with_docs();
        assert!(s.search_fts("episdot", 5).unwrap().is_empty());
        let hits = s.search_vector("episdot", 5).unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].path == "/semantik/s2");
    }
}
