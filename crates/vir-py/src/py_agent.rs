//! `vir.agent`: RLM rekursif di atas LM berskrip (PRD F3.4).
//!
//! Loop RLM berjalan di Rust dan menjalankan kode di REPL CPython terpisah;
//! karena itu GIL dilepas selama loop (`Python::allow_threads`). Bila `mem`
//! diberikan, callable Python dipanggil lewat GIL untuk setiap operasi memori.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use vir_agent::{run_rlm, RlmConfig, RlmRun, ScriptedLm};
use vir_common::VirError;

use crate::err;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rlm_run, m)?)?;
    m.add_function(wrap_pyfunction!(rlm_run_info, m)?)?;
    m.add_class::<Rlm>()?;
    Ok(())
}

/// Anggaran RLM dari argumen Python.
fn config(max_depth: usize, max_calls: usize) -> RlmConfig {
    RlmConfig {
        max_depth,
        max_calls,
        max_steps: max_calls.max(1),
    }
}

/// Jembatan memori tanpa host: semua operasi mengembalikan null.
fn null_mem(_op: &str, _payload: &Value) -> Result<Value, VirError> {
    Ok(Value::Null)
}

/// Jalankan loop RLM dengan LM berskrip; GIL dilepas selama loop.
fn run_scripted(
    py: Python<'_>,
    responses: Vec<String>,
    context: String,
    cfg: RlmConfig,
) -> PyResult<RlmRun> {
    py.allow_threads(move || {
        let lm = ScriptedLm::new(responses);
        run_rlm(&lm, &context, &cfg, null_mem).map_err(err::runtime)
    })
}

/// Hasil RLM sebagai dict Python.
fn outcome_dict<'py>(py: Python<'py>, out: &RlmRun) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("final_answer", out.final_answer.as_deref())?;
    dict.set_item("steps", out.steps)?;
    dict.set_item("calls_used", out.calls_used)?;
    dict.set_item("recovered", out.recovered)?;
    Ok(dict)
}

/// Jalankan RLM dengan LM berskrip; kembalikan jawaban akhir saja.
#[pyfunction]
#[pyo3(signature = (responses, context, max_depth = 3, max_calls = 8))]
fn rlm_run(
    py: Python<'_>,
    responses: Vec<String>,
    context: String,
    max_depth: usize,
    max_calls: usize,
) -> PyResult<Option<String>> {
    let out = run_scripted(py, responses, context, config(max_depth, max_calls))?;
    Ok(out.final_answer)
}

/// Sama seperti [`rlm_run`], tetapi mengembalikan dict statistik.
#[pyfunction]
#[pyo3(signature = (responses, context, max_depth = 3, max_calls = 8))]
fn rlm_run_info<'py>(
    py: Python<'py>,
    responses: Vec<String>,
    context: String,
    max_depth: usize,
    max_calls: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let out = run_scripted(py, responses, context, config(max_depth, max_calls))?;
    outcome_dict(py, &out)
}

/// RLM rekursif dengan LM berskrip dan jembatan memori opsional.
#[pyclass(name = "RLM")]
pub struct Rlm {
    responses: Vec<String>,
    mem: Option<Py<PyAny>>,
    max_depth: usize,
    max_calls: usize,
}

#[pymethods]
impl Rlm {
    /// `responses` = urutan jawaban LM; `mem` = callable `(op, payload_json)`.
    #[new]
    #[pyo3(signature = (responses, mem = None, max_depth = 3, max_calls = 8))]
    fn new(
        responses: Vec<String>,
        mem: Option<Py<PyAny>>,
        max_depth: usize,
        max_calls: usize,
    ) -> Self {
        Self {
            responses,
            mem,
            max_depth,
            max_calls,
        }
    }

    /// Jalankan RLM pada tugas; kembalikan dict hasil.
    fn run<'py>(&self, py: Python<'py>, context: &str) -> PyResult<Bound<'py, PyDict>> {
        let out = self.run_inner(py, context)?;
        outcome_dict(py, &out)
    }

    /// Kedalaman panggilan maksimum.
    #[getter]
    fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Jumlah panggilan model maksimum.
    #[getter]
    fn max_calls(&self) -> usize {
        self.max_calls
    }
}

impl Rlm {
    fn run_inner(&self, py: Python<'_>, context: &str) -> PyResult<RlmRun> {
        let responses = self.responses.clone();
        let mem = self.mem.as_ref().map(|m| m.clone_ref(py));
        let (max_depth, max_calls) = (self.max_depth, self.max_calls);
        let task = context.to_string();
        py.allow_threads(move || {
            let lm = ScriptedLm::new(responses);
            let cfg = config(max_depth, max_calls);
            let mut bridge = |op: &str, payload: &Value| call_mem(&mem, op, payload);
            run_rlm(&lm, &task, &cfg, &mut bridge).map_err(err::runtime)
        })
    }
}

/// Panggil callable `mem` milik host lewat GIL; null bila tidak ada.
fn call_mem(mem: &Option<Py<PyAny>>, op: &str, payload: &Value) -> Result<Value, VirError> {
    let Some(callable) = mem else {
        return Ok(Value::Null);
    };
    Python::with_gil(|py| {
        let out = callable
            .call1(py, (op, payload.to_string()))
            .map_err(|e| VirError::Internal(e.to_string()))?;
        let json: String = out
            .extract(py)
            .map_err(|e| VirError::Internal(e.to_string()))?;
        serde_json::from_str(&json).map_err(|e| VirError::Internal(e.to_string()))
    })
}
