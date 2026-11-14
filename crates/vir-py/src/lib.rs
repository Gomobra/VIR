#![allow(clippy::useless_conversion)]
//! Binding Python untuk inti VIR (PRD F3.6).
//!
//! Satu modul per lapisan (`py_text`, `py_vq`, `py_model`, `py_train`, `py_mfs`,
//! `py_agent`, `py_eval`) plus `py_backend` yang menjaga lapisan berat. Operasi
//! berat (latih tokenizer, loop RLM) melepas GIL lewat [`Python::allow_threads`];
//! galat Rust dipetakan ke eksepsi Python di [`err`].

mod err;
mod py_agent;
mod py_backend;
mod py_eval;
mod py_mfs;
mod py_model;
mod py_text;
mod py_train;
mod py_vq;
use pyo3::prelude::*;

/// Info build (dipakai `vir.info()`).
#[pyfunction]
fn build_info() -> String {
    vir_common::build_info()
}

/// Versi Vbeta.
#[pyfunction]
fn version() -> &'static str {
    "Vbeta"
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    py_text::register(m)?;
    py_vq::register(m)?;
    py_model::register(m)?;
    py_train::register(m)?;
    py_mfs::register(m)?;
    py_agent::register(m)?;
    py_eval::register(m)?;
    py_backend::register(m)?;
    Ok(())
}
