#![allow(clippy::useless_conversion)]
use pyo3::prelude::*;
use vir_agent::{run_rlm, RlmConfig, ScriptedLm};
use vir_mfs::MemoryStore;
use vir_text::BpeTokenizer;

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
    BpeTokenizer::new().encode(text)
}

#[pyfunction]
fn text_decode(tokens: Vec<u32>) -> String {
    BpeTokenizer::new().decode(&tokens)
}

// ---- MemoryStore ----
#[pyclass]
struct PyMemory {
    store: MemoryStore,
}

#[pymethods]
impl PyMemory {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let store = if path == ":memory:" {
            MemoryStore::open_in_memory()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        } else {
            MemoryStore::open(path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        };
        Ok(Self { store })
    }

    fn write(&self, path: &str, content: &str) -> PyResult<i64> {
        self.store
            .write(path, content)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn read(&self, version_id: i64) -> PyResult<String> {
        self.store
            .read(version_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn find(&self, query: &str, limit: usize) -> PyResult<Vec<(i64, String, f64)>> {
        self.store
            .find(query, limit)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

// ---- RLM ----
#[pyfunction]
fn rlm_run(
    responses: Vec<String>,
    context: &str,
    max_depth: usize,
    max_calls: usize,
) -> PyResult<Option<String>> {
    let lm = ScriptedLm::new(responses);
    let cfg = RlmConfig {
        max_depth,
        max_calls,
    };
    let r = run_rlm(&lm, context, &cfg)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(r.final_answer)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(text_encode, m)?)?;
    m.add_function(wrap_pyfunction!(text_decode, m)?)?;
    m.add_function(wrap_pyfunction!(rlm_run, m)?)?;
    m.add_class::<PyMemory>()?;
    Ok(())
}
