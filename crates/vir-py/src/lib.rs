use pyo3::prelude::*;
use vir_text::ByteTokenizer;

#[pyfunction]
fn build_info() -> String {
    vir_common::build_info()
}

#[pyfunction]
fn version() -> &'static str {
    "Vbeta"
}

#[pyfunction]
fn text_encode(text: &str) -> Vec<u32> {
    ByteTokenizer::new().encode(text)
}

#[pyfunction]
fn text_decode(tokens: Vec<u32>) -> String {
    ByteTokenizer::new().decode(&tokens)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(text_encode, m)?)?;
    m.add_function(wrap_pyfunction!(text_decode, m)?)?;
    Ok(())
}
