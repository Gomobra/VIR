//! Pemuat `data/sources.json`: sumber dan lisensi nyata (PRD F1.3 langkah 5).
//!
//! Berkas `data/sources.json` di akar repo berisi daftar sumber korpus beserta
//! lisensi dan URL. Entri `language` dan `weight` bersifat opsional agar berkas
//! yang sudah ada tetap terbaca; bawaannya `und` dan `1.0`. Pemuat memvalidasi
//! id unik, URL, dan bobot tidak negatif.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use vir_common::VirError;

/// Bahasa bawaan bila sumber tidak menyebut bahasa.
pub const DEFAULT_LANGUAGE: &str = "und";
/// Bobot bawaan bila sumber tidak menyebut bobot.
pub const DEFAULT_WEIGHT: f32 = 1.0;

/// Satu sumber korpus.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SourceEntry {
    pub id: String,
    pub license: String,
    pub url: String,
    #[serde(default = "default_language")]
    pub language: String,
    #[serde(default = "default_weight")]
    pub weight: f32,
}

impl SourceEntry {
    /// Bobot sebagai `f64` untuk aritmetika pencampuran.
    pub fn weight_f64(&self) -> f64 {
        f64::from(self.weight)
    }
}

fn default_language() -> String {
    DEFAULT_LANGUAGE.to_string()
}

fn default_weight() -> f32 {
    DEFAULT_WEIGHT
}

/// Katalog sumber: isi `data/sources.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SourceCatalog {
    pub sources: Vec<SourceEntry>,
}

impl SourceCatalog {
    /// Jumlah sumber.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// `true` bila katalog kosong.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Id semua sumber, berurutan.
    pub fn ids(&self) -> Vec<&str> {
        self.sources.iter().map(|s| s.id.as_str()).collect()
    }

    /// Bobot semua sumber, berurutan (untuk `vir-data::mix`).
    pub fn weights(&self) -> Vec<f32> {
        self.sources.iter().map(|s| s.weight).collect()
    }

    /// Cari sumber berdasarkan id.
    pub fn get(&self, id: &str) -> Option<&SourceEntry> {
        self.sources.iter().find(|s| s.id == id)
    }

    /// Sumber dengan bahasa tertentu.
    pub fn by_language(&self, language: &str) -> Vec<&SourceEntry> {
        self.sources
            .iter()
            .filter(|s| s.language == language)
            .collect()
    }

    /// Validasi katalog: id unik dan tidak kosong, URL http(s), bobot tidak negatif.
    pub fn validate(&self) -> Result<(), VirError> {
        if self.sources.is_empty() {
            return Err(VirError::Config("katalog sumber kosong".into()));
        }
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for source in &self.sources {
            if source.id.trim().is_empty() {
                return Err(VirError::Config("id sumber tidak boleh kosong".into()));
            }
            if !seen.insert(source.id.as_str()) {
                let id = source.id.as_str();
                return Err(VirError::Config(format!("id sumber ganda: {id}")));
            }
            if !source.url.starts_with("http://") && !source.url.starts_with("https://") {
                let id = source.id.as_str();
                let url = source.url.as_str();
                return Err(VirError::Config(format!(
                    "url sumber '{id}' harus http(s): {url}"
                )));
            }
            if source.license.trim().is_empty() {
                let id = source.id.as_str();
                return Err(VirError::Config(format!("lisensi sumber '{id}' kosong")));
            }
            if !source.weight.is_finite() || source.weight < 0.0 {
                let id = source.id.as_str();
                let weight = source.weight;
                return Err(VirError::Config(format!(
                    "bobot sumber '{id}' tidak sah: {weight}"
                )));
            }
        }
        Ok(())
    }
}

/// Uraikan katalog dari teks JSON.
pub fn parse_sources(text: &str) -> Result<SourceCatalog, VirError> {
    let catalog: SourceCatalog = serde_json::from_str(text)
        .map_err(|err| VirError::Internal(format!("sources.json tidak valid: {err}")))?;
    catalog.validate()?;
    Ok(catalog)
}

/// Muat katalog dari berkas.
pub fn load_sources(path: impl AsRef<Path>) -> Result<SourceCatalog, VirError> {
    let text = std::fs::read_to_string(path.as_ref())?;
    parse_sources(&text)
}

/// Path `data/sources.json` relatif ke akar repo (dua tingkat di atas crate).
pub fn repo_sources_path() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join("data").join("sources.json"))
        .unwrap_or_else(|| PathBuf::from("data/sources.json"))
}

/// Muat katalog bawaan repo (`data/sources.json`).
pub fn load_repo_sources() -> Result<SourceCatalog, VirError> {
    load_sources(repo_sources_path())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTOH: &str = r#"{
        "sources": [
            {"id": "tinystories", "license": "CC-BY-4.0", "url": "https://example.org/tinystories"},
            {"id": "wmt14", "license": "research-use", "url": "https://example.org/wmt14", "language": "en", "weight": 0.5}
        ]
    }"#;

    #[test]
    fn muat_dan_validasi_katalog_contoh() {
        let catalog = parse_sources(CONTOH).expect("katalog contoh");
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog.ids(), vec!["tinystories", "wmt14"]);
        let default_language = catalog.get("tinystories").map(|s| s.language.clone());
        assert_eq!(default_language, Some(DEFAULT_LANGUAGE.to_string()));
        let default_weight = catalog.get("tinystories").map(|s| s.weight);
        assert_eq!(default_weight, Some(DEFAULT_WEIGHT));
        assert_eq!(catalog.weights(), vec![1.0, 0.5]);
        assert_eq!(catalog.by_language("en").len(), 1);
        assert!(catalog.get("tidak-ada").is_none());
    }

    #[test]
    fn katalog_rusak_ditolak() {
        assert!(parse_sources(r#"{"sources": []}"#).is_err());
        assert!(parse_sources(r#"{"sources": [{"id": "a"}]}"#).is_err());
        assert!(parse_sources("bukan json").is_err());
        let ganda = r#"{"sources": [
            {"id": "a", "license": "MIT", "url": "https://x/y"},
            {"id": "a", "license": "MIT", "url": "https://x/z"}
        ]}"#;
        assert!(parse_sources(ganda).is_err());
        let url_salah = r#"{"sources": [
            {"id": "a", "license": "MIT", "url": "ftp://x/y"}
        ]}"#;
        assert!(parse_sources(url_salah).is_err());
        let bobot = r#"{"sources": [
            {"id": "a", "license": "MIT", "url": "https://x/y", "weight": -1.0}
        ]}"#;
        assert!(parse_sources(bobot).is_err());
    }

    #[test]
    fn katalog_repo_tiap_waktu_terbaca() {
        let catalog = load_repo_sources().expect("data/sources.json harus terbaca");
        assert!(catalog.len() >= 4);
        assert!(catalog.validate().is_ok());
        let tinystories = catalog.get("tinystories");
        assert!(tinystories.is_some());
        assert!(catalog.ids().contains(&"tinystories"));
    }
}
