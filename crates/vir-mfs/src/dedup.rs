//! Dedup kosinus yang menaut, bukan menghapus.

use rusqlite::params;

use crate::Result;
use crate::db::sql_err;
use crate::meta::LinkKind;
use crate::store::MemoryStore;
use crate::vec_index::{cosine, decode_vector};

impl MemoryStore {
    pub(crate) fn index_and_dedup(
        &self,
        version_id: i64,
        content: &str,
    ) -> Result<Option<(i64, f32)>> {
        let Some(embedder) = self.embedder.as_deref() else {
            return Ok(None);
        };
        let vector = embedder.embed(content);
        if vector.is_empty() {
            return Ok(None);
        }
        self.store_vector(version_id, &vector)?;
        let Some((other, score)) =
            self.find_duplicates(version_id, &vector, self.dedup_threshold)?
        else {
            return Ok(None);
        };
        self.link(version_id, other, LinkKind::DuplicateOf)?;
        Ok(Some((other, score)))
    }

    /// Versi terdekat (di luar `version_id`, belum dilupakan) dengan kosinus ≥ ambang.
    pub fn find_duplicates(
        &self,
        version_id: i64,
        vector: &[f32],
        threshold: f32,
    ) -> Result<Option<(i64, f32)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT v.id, vec.blob FROM vectors vec
                 JOIN versions v ON v.id = vec.version_id
                 WHERE v.id <> ?1 AND v.deleted = 0",
            )
            .map_err(sql_err)?;
        let rows = stmt
            .query_map(params![version_id], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
            })
            .map_err(sql_err)?;
        let mut best: Option<(i64, f32)> = None;
        for row in rows {
            let (id, blob) = row.map_err(sql_err)?;
            let other = decode_vector(&blob)?;
            if other.len() != vector.len() {
                continue;
            }
            let score = cosine(vector, &other);
            if score < threshold {
                continue;
            }
            let better = match best {
                Some((_, b)) => score > b,
                None => true,
            };
            if better {
                best = Some((id, score));
            }
        }
        Ok(best)
    }

    /// Cek duplikat untuk isi yang belum ditulis.
    pub fn dedup_check(&self, content: &str) -> Result<Option<(i64, f32)>> {
        let Some(embedder) = self.embedder.as_deref() else {
            return Ok(None);
        };
        let vector = embedder.embed(content);
        if vector.is_empty() {
            return Ok(None);
        }
        self.find_duplicates(-1, &vector, self.dedup_threshold)
    }
}
