//! Galat CLI: satu tipe untuk semua kegagalan perintah (PRD F3.6).

use std::fmt;

/// Galat yang dilempar subperintah; `main` mencetaknya lalu keluar ≠ 0.
#[derive(Debug)]
pub(crate) enum CliError {
    /// Pesan bebas (argumen, IO, MFS, model).
    Message(String),
    /// Backend berat tidak tersedia di build ini.
    Backend(crate::backend::BackendUnavailable),
}

impl CliError {
    /// Galat pesan bebas.
    pub(crate) fn other(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => write!(f, "{message}"),
            Self::Backend(backend) => write!(f, "{backend}"),
        }
    }
}

impl std::error::Error for CliError {}

impl From<vir_common::VirError> for CliError {
    fn from(error: vir_common::VirError) -> Self {
        Self::Message(error.to_string())
    }
}
