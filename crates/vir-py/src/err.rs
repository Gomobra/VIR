//! Pemetaan galat Rust ke eksepsi Python (PRD F3.6).

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;

/// Galat operasi (IO, SQLite, internal) → `RuntimeError`.
pub(crate) fn runtime(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Argumen tidak valid → `ValueError`.
pub(crate) fn value(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
