//! Status backend berat (PRD F3.6).
//!
//! `train` dan `generate` membutuhkan backend Burn (`vir-train`, `vir-model`)
//! yang hidup di workspace terpisah dan tidak ikut ke build inti ini. Alih-alih
//! stub, perintahnya memvalidasi argumen lebih dulu lalu gagal dengan galat yang
//! jelas dan kode keluar khusus.

use std::fmt;

/// Backend latihan.
pub(crate) const TRAIN_BACKEND: &str = "vir-train (Burn)";
/// Backend model.
pub(crate) const MODEL_BACKEND: &str = "vir-model (Burn)";

/// Perintah yang butuh backend Burn tidak tersedia pada build inti.
#[derive(Debug)]
pub(crate) struct BackendUnavailable {
    command: &'static str,
    backend: &'static str,
}

impl BackendUnavailable {
    pub(crate) fn new(command: &'static str, backend: &'static str) -> Self {
        Self { command, backend }
    }

    /// Muatan JSON untuk mode `--json`.
    pub(crate) fn json(&self) -> String {
        let payload = serde_json::json!({
            "command": self.command,
            "status": "unavailable",
            "backend": self.backend,
            "build": vir_common::build_info(),
        });
        payload.to_string()
    }
}

impl fmt::Display for BackendUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "perintah '{}' belum tersedia di build ini: butuh backend {} yang tidak disertakan pada build inti (bangun ulang dengan maturin dari crates/vir-py)",
            self.command, self.backend
        )
    }
}

impl std::error::Error for BackendUnavailable {}
