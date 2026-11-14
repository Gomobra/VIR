//! `vir.train`: penjaga backend latihan (PRD F1.7).
//!
//! Trainer dua tahap membutuhkan backend Burn (`vir-train`) yang tidak ikut ke
//! build inti. Fungsi di sini memvalidasi argumen lalu gagal cepat dengan pesan
//! yang bisa ditindaklanjuti — bukan stub: tidak ada jalur yang mengaku berhasil.

use std::path::Path;

use pyo3::exceptions::PyFileNotFoundError;
use pyo3::prelude::*;

use crate::py_backend;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_available, m)?)?;
    m.add_function(wrap_pyfunction!(train_backend, m)?)?;
    m.add_function(wrap_pyfunction!(train_run, m)?)?;
    Ok(())
}

/// `true` bila build ini memuat backend latihan (Burn).
#[pyfunction]
fn train_available() -> bool {
    false
}

/// Nama backend latihan pada build ini.
#[pyfunction]
fn train_backend() -> &'static str {
    py_backend::TRAIN_BACKEND
}

/// Jalankan latihan.
///
/// `data` = korpus token (shard `vir-data`), `out` = direktori run (checkpoint +
/// log JSONL), `steps` = jumlah langkah, `resume` = direktori run sebelumnya.
#[pyfunction]
#[pyo3(signature = (data, out, steps = 1000, resume = None))]
fn train_run(data: &str, out: &str, steps: u64, resume: Option<&str>) -> PyResult<()> {
    if !Path::new(data).exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "korpus token tidak ditemukan: {data}"
        )));
    }
    if !Path::new(out).exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "direktori keluaran tidak ditemukan: {out}"
        )));
    }
    if steps == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("steps harus > 0"));
    }
    if let Some(dir) = resume {
        if !Path::new(dir).exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "run resume tidak ditemukan: {dir}"
            )));
        }
    }
    py_backend::require("train")
}
