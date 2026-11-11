use thiserror::Error;

#[derive(Debug, Error)]
pub enum VirError {
    #[error("konfigurasi tidak valid: {0}")]
    Config(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialisasi: {0}")]
    Serde(#[from] serde_yaml::Error),
    #[error("galat internal: {0}")]
    Internal(String),
}
