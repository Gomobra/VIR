//! `vir.model`: konfigurasi model (PRD F1.6) dan penjaga backend.

use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use vir_common::{LevelScheduleKind, ModelConfig};

use crate::err;
use crate::py_backend;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModelConfig>()?;
    m.add_function(wrap_pyfunction!(model_available, m)?)?;
    m.add_function(wrap_pyfunction!(model_backend, m)?)?;
    m.add_function(wrap_pyfunction!(model_load, m)?)?;
    Ok(())
}

/// Nama jadwal level → enum konfigurasi.
fn schedule_kind(name: &str) -> PyResult<LevelScheduleKind> {
    match name {
        "linear" => Ok(LevelScheduleKind::Linear),
        "quadratic" => Ok(LevelScheduleKind::Quadratic),
        "sqrt" => Ok(LevelScheduleKind::Sqrt),
        "log" => Ok(LevelScheduleKind::Log),
        "exp" => Ok(LevelScheduleKind::Exp),
        "cosine" => Ok(LevelScheduleKind::Cosine),
        other => Err(PyValueError::new_err(format!(
            "jadwal level tidak dikenal: {other}"
        ))),
    }
}

/// Konfigurasi model VQ-VAE + generator (F1.6).
#[pyclass(name = "ModelConfig")]
#[derive(Clone)]
pub struct PyModelConfig {
    inner: ModelConfig,
}

#[pymethods]
impl PyModelConfig {
    #[new]
    #[pyo3(signature = (
        seed = 42,
        target_length = 128,
        num_levels = 12,
        codebook_size = 4096,
        latent_dim = 384,
        hidden_dim = 768,
        num_heads = 12,
        dropout = 0.1,
        rope_base = 10000.0,
        schedule = "linear",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        seed: u64,
        target_length: usize,
        num_levels: usize,
        codebook_size: usize,
        latent_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        dropout: f32,
        rope_base: f32,
        schedule: &str,
    ) -> PyResult<Self> {
        let inner = ModelConfig {
            seed,
            target_length,
            num_levels,
            codebook_size,
            latent_dim,
            hidden_dim,
            num_heads,
            dropout,
            rope_base,
            schedule: schedule_kind(schedule)?,
            ..ModelConfig::default()
        };
        inner.validate().map_err(err::value)?;
        Ok(Self { inner })
    }

    /// Konfigurasi bawaan PRD (lolos validasi).
    #[staticmethod]
    fn with_defaults() -> Self {
        Self {
            inner: ModelConfig::default(),
        }
    }

    /// Panjang tiap level (naik ketat sampai `target_length`).
    fn levels(&self) -> Vec<usize> {
        self.inner.levels()
    }

    /// Validasi ulang konfigurasi.
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(err::value)
    }

    /// Konfigurasi sebagai JSON (untuk checkpoint/resume).
    fn as_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(err::runtime)
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    #[getter]
    fn target_length(&self) -> usize {
        self.inner.target_length
    }

    #[getter]
    fn num_levels(&self) -> usize {
        self.inner.num_levels
    }

    #[getter]
    fn codebook_size(&self) -> usize {
        self.inner.codebook_size
    }

    #[getter]
    fn latent_dim(&self) -> usize {
        self.inner.latent_dim
    }

    #[getter]
    fn hidden_dim(&self) -> usize {
        self.inner.hidden_dim
    }

    #[getter]
    fn num_heads(&self) -> usize {
        self.inner.num_heads
    }

    #[getter]
    fn schedule(&self) -> String {
        format!("{:?}", self.inner.schedule).to_lowercase()
    }
}

/// `true` bila build ini memuat backend model (Burn).
#[pyfunction]
fn model_available() -> bool {
    false
}

/// Nama backend model pada build ini.
#[pyfunction]
fn model_backend() -> &'static str {
    py_backend::MODEL_BACKEND
}

/// Muat checkpoint. Gagal cepat bila backend model tidak ada di build ini.
#[pyfunction]
fn model_load(path: &str) -> PyResult<()> {
    if !std::path::Path::new(path).exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "checkpoint tidak ditemukan: {path}"
        )));
    }
    py_backend::require("model")
}
