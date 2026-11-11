//! Fondasi bersama VIR: konfigurasi, galat, RNG, dan jadwal level.

pub mod config;
pub mod error;
pub mod rng;
pub mod schedule;

pub use config::{LevelScheduleKind, ModelConfig};
pub use error::VirError;
pub use rng::DeterministicRng;
pub use schedule::level_schedule;

/// Informasi build yang diekspos ke Python dan CLI.
pub fn build_info() -> String {
    format!(
        "VIR Vbeta (rust, target={}, profile={})",
        std::env::consts::ARCH,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    )
}
