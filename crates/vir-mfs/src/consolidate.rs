//! Konsolidasi non-destruktif: episodik → semantik sebagai versi baru.

use rusqlite::params;

use crate::db::sql_err;
use crate::meta::LinkKind;
use crate::store::MemoryStore;
use crate::{Result, VirError};

/// Peringkas isi menjadi teks pendek. Implementasi nyata memakai model.
pub trait Summarizer {
    fn summarize(&self, text: &str) -> String;
}

/// Peringkas berbasis aturan: baris-baris awal sampai batas karakter.
#[derive(Debug, Clone, Copy)]
pub struct HeadlineSummarizer {
    pub max_chars: usize,
}

impl Default for HeadlineSummarizer {
    fn default() -> Self {
        Self { max_chars: 280 }
    }
}

impl Summarizer for HeadlineSummarizer {
    fn summarize(&self, text: &str) -> String {
        let mut out = String::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(line);
            if out.chars().count() >= self.max_chars {
                break;
            }
        }
        truncate_chars(&out, self.max_chars)
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect()
}

/// Hasil satu konsolidasi.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConsolidationReport {
    pub source_version: i64,
    pub target_file_id: i64,
    pub version_id: i64,
    pub duplicate_of: Option<(i64, f32)>,
}

impl MemoryStore {
    /// Ringkas satu versi episodik menjadi versi baru di `target_path`.
    ///
    /// Isi sumber tidak diubah; versi baru menyimpan `source` dan tautan
    /// `derived_from`.
    pub fn consolidate(
        &self,
        source_version: i64,
        target_path: &str,
        summarizer: &dyn Summarizer,
    ) -> Result<ConsolidationReport> {
        if !self.version_exists(source_version)? {
            return Err(VirError::Config(format!(
                "versi sumber {source_version} tidak ada"
            )));
        }
        let content = self.read_version(source_version)?;
        let summary = summarizer.summarize(&content);
        let outcome = self.write_version(target_path, &summary, Some(source_version))?;
        self.link(outcome.version_id, source_version, LinkKind::DerivedFrom)?;
        let info = self.version_info(outcome.version_id)?;
        Ok(ConsolidationReport {
            source_version,
            target_file_id: info.file_id,
            version_id: outcome.version_id,
            duplicate_of: outcome.duplicate_of,
        })
    }

    /// Versi episodik yang belum diturunkan menjadi semantik.
    pub fn pending_episodes(&self, namespace: &str) -> Result<Vec<i64>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT v.id FROM versions v JOIN files f ON f.id = v.file_id
                 WHERE f.namespace = ?1 AND v.deleted = 0
                   AND NOT EXISTS (
                       SELECT 1 FROM links l
                       WHERE l.kind = 'derived_from' AND l.to_version = v.id)
                 ORDER BY v.id",
            )
            .map_err(sql_err)?;
        let rows = stmt
            .query_map(params![namespace], |r| r.get::<_, i64>(0))
            .map_err(sql_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(sql_err)?);
        }
        Ok(out)
    }

    /// Konsolidasikan semua episodik tertunda di satu ruang nama.
    pub fn consolidate_pending(
        &self,
        namespace: &str,
        target_path: &str,
        summarizer: &dyn Summarizer,
    ) -> Result<Vec<ConsolidationReport>> {
        let mut out = Vec::new();
        for version_id in self.pending_episodes(namespace)? {
            out.push(self.consolidate(version_id, target_path, summarizer)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinkKind;

    #[test]
    fn peringkas_memotong() {
        let s = HeadlineSummarizer { max_chars: 10 };
        let out = s.summarize("baris satu\nbaris dua yang panjang");
        assert!(out.chars().count() <= 10);
        assert!(out.starts_with("baris satu"));
    }

    #[test]
    fn konsolidasi_tidak_mengubah_sumber() {
        let store = MemoryStore::open_in_memory().unwrap();
        let src = store
            .write("/episodik/e1", "alpha beta gamma delta")
            .unwrap();
        let before = store.version_info(src).unwrap().hash;
        let report = store
            .consolidate(src, "/semantik/ringkas", &HeadlineSummarizer::default())
            .unwrap();
        assert_eq!(store.read(src).unwrap(), "alpha beta gamma delta");
        assert_eq!(store.version_info(src).unwrap().hash, before);
        let info = store.version_info(report.version_id).unwrap();
        assert_eq!(info.source, Some(src));
        assert!(
            store
                .links_from(report.version_id)
                .unwrap()
                .contains(&(src, LinkKind::DerivedFrom))
        );
        assert_eq!(
            store.read(report.version_id).unwrap(),
            "alpha beta gamma delta"
        );
    }

    #[test]
    fn pending_episodik() {
        let store = MemoryStore::open_in_memory().unwrap();
        let a = store.write("/episodik/a", "satu").unwrap();
        let b = store.write("/episodik/b", "dua").unwrap();
        store.write("/semantik/c", "tiga").unwrap();
        assert_eq!(store.pending_episodes("episodik").unwrap(), vec![a, b]);
        store
            .consolidate(a, "/semantik/a", &HeadlineSummarizer::default())
            .unwrap();
        assert_eq!(store.pending_episodes("episodik").unwrap(), vec![b]);
    }
}
