//! `vir.agent`: RLM rekursif dengan LM nyata dan MFS nyata (PRD F3.4, F3.5).
//!
//! Loop RLM berjalan di Rust dan menjalankan kode di REPL CPython terpisah;
//! karena itu GIL dilepas selama loop (`Python::allow_threads`).
//!
//! Root LM adalah callable Python `(prompt) -> str` (ADR-005: default Vbeta
//! adalah callable, bukan model VIR). `mem` adalah callable
//! `(op, payload_json) -> json_str`; tanpa callable, operasi memori gagal
//! dengan galat yang jelas, bukan `null` diam-diam.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use vir_agent::{run_rlm, ResLm, RlmConfig, RlmRun};
use vir_common::VirError;

use crate::err;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Rlm>()?;
    Ok(())
}

/// Anggaran RLM dari argumen Python.
fn config(max_depth: usize, max_calls: usize, max_steps: usize) -> RlmConfig {
    RlmConfig {
        max_depth,
        max_calls,
        max_steps: max_steps.max(1),
    }
}

/// RLM dengan root callable Python dan jembatan memori opsional.
#[pyclass(name = "RLM")]
pub struct Rlm {
    generate: Py<PyAny>,
    mem: Option<Py<PyAny>>,
    max_depth: usize,
    max_calls: usize,
    max_steps: usize,
}

impl Rlm {
    fn run_inner(&self, py: Python<'_>, context: &str) -> PyResult<RlmRun> {
        let generate = self.generate.clone_ref(py);
        let mem = self.mem.as_ref().map(|m| m.clone_ref(py));
        let (max_depth, max_calls, max_steps) = (self.max_depth, self.max_calls, self.max_steps);
        let task = context.to_string();
        py.allow_threads(move || {
            let lm = ResLm::new(move |prompt: &str| call_generate(&generate, prompt));
            let cfg = config(max_depth, max_calls, max_steps);
            run_rlm(&lm, &task, &cfg, &mut |payload: &Value| {
                call_mem(&mem, payload)
            })
            .map_err(err::runtime)
        })
    }
}

#[pymethods]
impl Rlm {
    /// `generate` = callable `(prompt: str) -> str`; `mem` = callable
    /// `(op: str, payload_json: str) -> json_str`.
    #[new]
    #[pyo3(signature = (generate, mem = None, max_depth = 3, max_calls = 8, max_steps = 8))]
    fn new(
        generate: Py<PyAny>,
        mem: Option<Py<PyAny>>,
        max_depth: usize,
        max_calls: usize,
        max_steps: usize,
    ) -> Self {
        Self {
            generate,
            mem,
            max_depth,
            max_calls,
            max_steps,
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

    /// Jumlah langkah maksimum.
    #[getter]
    fn max_steps(&self) -> usize {
        self.max_steps
    }
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

/// Panggil callable root LM lewat GIL.
fn call_generate(callable: &Py<PyAny>, prompt: &str) -> Result<String, VirError> {
    Python::with_gil(|py| {
        let out = callable
            .call1(py, (prompt,))
            .map_err(|e| VirError::Internal(e.to_string()))?;
        out.extract::<String>(py)
            .map_err(|e| VirError::Internal(e.to_string()))
    })
}

/// Panggil callable `mem` milik host lewat GIL.
fn call_mem(mem: &Option<Py<PyAny>>, payload: &Value) -> Result<Value, VirError> {
    let Some(callable) = mem else {
        return Err(VirError::Config(
            "operasi memori butuh callable `mem` (op, payload_json) -> json".into(),
        ));
    };
    let op = payload
        .get("op")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Python::with_gil(|py| {
        let out = callable
            .call1(py, (&op, payload.to_string()))
            .map_err(|e| VirError::Internal(e.to_string()))?;
        let json: String = out
            .extract(py)
            .map_err(|e| VirError::Internal(e.to_string()))?;
        serde_json::from_str(&json).map_err(|e| VirError::Internal(e.to_string()))
    })
}
