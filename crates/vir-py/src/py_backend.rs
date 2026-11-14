//! Penjaga backend berat (PRD F3.6).
//!
//! Wheel inti hanya memuat lapisan murni Rust. Backend Burn (`vir-train`,
//! `vir-model`, `vir-vq`) berada di workspace terpisah dan tidak disertakan;
//! fungsi yang memerlukannya gagal cepat dengan pesan yang bisa ditindaklanjuti
//! (bukan stub: tidak ada jalur yang mengaku berhasil).

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Nama backend latihan pada build inti.
pub(crate) const TRAIN_BACKEND: &str = "vir-train (Burn) — tidak disertakan di build inti";
/// Nama backend model pada build inti.
pub(crate) const MODEL_BACKEND: &str = "vir-model (Burn) — tidak disertakan di build inti";
/// Catatan cara mendapatkan backend penuh.
const BURN_NOTE: &str = "bangun ulang dari crates/vir-py dengan maturin dan fitur backend Burn";

/// Pastikan backend `name` ada di build ini; kalau tidak, galat yang jelas.
pub(crate) fn require(name: &str) -> PyResult<()> {
    if matches!(name, "text" | "mfs" | "agent" | "eval") {
        return Ok(());
    }
    Err(PyRuntimeError::new_err(format!(
        "{name} belum tersedia di build ini: backend Burn tidak disertakan; {BURN_NOTE}"
    )))
}

/// Galat `RuntimeError` bila backend `name` tidak ada di build ini.
#[pyfunction]
fn require_backend(name: &str) -> PyResult<()> {
    require(name)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(require_backend, m)?)?;
    m.add_function(wrap_pyfunction!(backend_status, m)?)?;
    Ok(())
}

/// Status tiap namespace: `True`, `False`, atau keterangan.
#[pyfunction]
fn backend_status(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("text", true)?;
    dict.set_item("mfs", true)?;
    dict.set_item("agent", true)?;
    dict.set_item("eval", true)?;
    dict.set_item("vq", "partial: hanya rumus FSQ")?;
    dict.set_item("model", false)?;
    dict.set_item("train", false)?;
    dict.set_item("burn", false)?;
    Ok(dict.into())
}
