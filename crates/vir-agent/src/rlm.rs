//! RLM: orkestrator rekursif dengan anggaran dan penghentian (F3.4).
//!
//! Loop root menerima ringkasan lingkungan, menulis kode Python, menjalankannya
//! di REPL stateful, lalu menambahkan keluaran (terpotong) ke transkrip sampai
//! kode mencetak `FINAL(...)`. `llm_query` memanggil sub-model dengan anggaran
//! kedalaman + jumlah panggilan; `mem` adalah jembatan ke MFS host.

use serde_json::{Value as JsonValue, json};
use vir_common::VirError;

use crate::lm::LanguageModel;
use crate::repl::{ExecOutcome, Repl, RpcRequest};

/// Anggaran dan batas langkah RLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RlmConfig {
    pub max_depth: usize,
    pub max_calls: usize,
    pub max_steps: usize,
}

impl Default for RlmConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_calls: 16,
            max_steps: 8,
        }
    }
}

/// Hasil satu sesi RLM.
#[derive(Debug, Clone)]
pub struct RlmRun {
    /// Jawaban `FINAL(...)` yang benar-benar dihasilkan model; `None` bila
    /// anggaran habis atau model tidak pernah mencetaknya.
    pub final_answer: Option<String>,
    pub steps: usize,
    pub calls_used: usize,
    /// Pernah terjadi pemulihan setelah crash REPL.
    pub recovered: bool,
}

/// Panggil sub-model dengan anggaran kedalaman dan jumlah panggilan.
pub fn llm_query(
    root: &dyn LanguageModel,
    prompt: &str,
    depth: usize,
    cfg: &RlmConfig,
    calls: &mut usize,
) -> Result<String, VirError> {
    if depth > cfg.max_depth {
        return Err(VirError::Internal("anggaran kedalaman habis".into()));
    }
    if *calls >= cfg.max_calls {
        return Err(VirError::Internal("anggaran panggilan habis".into()));
    }
    *calls += 1;
    root.generate(prompt)
}

/// Jalankan loop RLM sampai `FINAL(...)` atau anggaran habis.
///
/// `mem` menerima payload operasi MFS (`op`, `path`, ...) dan mengembalikan
/// hasil JSON yang nyata — jembatan tidak boleh mengarang nilai.
pub fn run_rlm(
    root: &dyn LanguageModel,
    context: &str,
    cfg: &RlmConfig,
    mut mem: impl FnMut(&JsonValue) -> Result<JsonValue, VirError>,
) -> Result<RlmRun, VirError> {
    let mut state = State::new(context);
    let mut repl = Repl::start()?;
    while !state.finished(cfg) {
        state.step(root, cfg, &mut repl, &mut mem)?;
    }
    Ok(state.into_run())
}

/// Status loop yang dapat dilanjutkan antar langkah.
struct State {
    calls: usize,
    steps: usize,
    recovered: bool,
    transcript: String,
    final_answer: Option<String>,
}

impl State {
    fn new(context: &str) -> Self {
        Self {
            calls: 0,
            steps: 0,
            recovered: false,
            transcript: format!("TUGAS:\n{context}\n\nRIWAYAT:\n"),
            final_answer: None,
        }
    }

    fn finished(&self, cfg: &RlmConfig) -> bool {
        self.final_answer.is_some() || self.calls >= cfg.max_calls || self.steps >= cfg.max_steps
    }

    fn step(
        &mut self,
        root: &dyn LanguageModel,
        cfg: &RlmConfig,
        repl: &mut Repl,
        mem: &mut impl FnMut(&JsonValue) -> Result<JsonValue, VirError>,
    ) -> Result<(), VirError> {
        self.calls += 1;
        let code = root.generate(&self.prompt())?;
        self.steps += 1;
        let mut handler = |req: &RpcRequest| self.handle(req, root, cfg, mem);
        let outcome = repl.exec(&code, &mut handler)?;
        self.absorb(&outcome);
        Ok(())
    }

    fn prompt(&self) -> String {
        format!(
            "{}\nTulis satu blok kode Python. Cetak FINAL(...) bila selesai.",
            self.transcript
        )
    }

    fn absorb(&mut self, outcome: &ExecOutcome) {
        if outcome.recovered {
            self.recovered = true;
        }
        match &outcome.error {
            Some(err) => self.transcript.push_str(&format!("GALAT: {err}\n")),
            None => {
                self.transcript.push_str(&outcome.output);
                self.transcript.push('\n');
            }
        }
        if let Some(pos) = outcome.output.find("FINAL(") {
            self.final_answer = Some(outcome.output[pos..].trim().to_string());
        }
    }

    fn handle(
        &mut self,
        req: &RpcRequest,
        root: &dyn LanguageModel,
        cfg: &RlmConfig,
        mem: &mut impl FnMut(&JsonValue) -> Result<JsonValue, VirError>,
    ) -> Result<JsonValue, VirError> {
        match req.method.as_str() {
            "llm_query" => {
                let prompt = req
                    .payload
                    .get("prompt")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("");
                let text = llm_query(root, prompt, 1, cfg, &mut self.calls)?;
                Ok(JsonValue::String(text))
            }
            "mem" => mem(&req.payload),
            other => Err(VirError::Internal(format!(
                "metode rpc tidak dikenal: {other}"
            ))),
        }
    }

    fn into_run(self) -> RlmRun {
        RlmRun {
            final_answer: self.final_answer,
            steps: self.steps,
            calls_used: self.calls,
            recovered: self.recovered,
        }
    }
}

/// Bantuan: bungkus nilai menjadi objek `{"result": ...}`.
pub fn result(value: JsonValue) -> JsonValue {
    json!({ "result": value })
}
