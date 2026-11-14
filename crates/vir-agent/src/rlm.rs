//! RLM: orkestrator rekursif dengan anggaran dan penghentian (F3.4).
//!
//! Loop root menerima ringkasan lingkungan, menulis kode Python, menjalankannya
//! di REPL stateful, lalu menambahkan keluaran (terpotong) ke transkrip sampai
//! kode mencetak `FINAL(...)`. `llm_query` memanggil sub-model dengan anggaran
//! kedalaman + jumlah panggilan; `mem` adalah jembatan ke host.

use serde_json::{Value as JsonValue, json};
use vir_common::VirError;

use crate::lm::LanguageModel;
use crate::repl::{Repl, RpcRequest};

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
/// `mem` adalah jembatan ke memori host: menerima operasi (mis. `read`,
/// `write`) dan payload RPC, mengembalikan hasil JSON.
pub fn run_rlm(
    root: &dyn LanguageModel,
    context: &str,
    cfg: &RlmConfig,
    mut mem: impl FnMut(&str, &JsonValue) -> Result<JsonValue, VirError>,
) -> Result<RlmRun, VirError> {
    let mut calls = 0usize;
    let mut steps = 0usize;
    let mut recovered = false;
    let mut repl = Repl::start()?;
    let mut transcript = format!("TUGAS:\n{context}\n\nRIWAYAT:\n");
    let mut final_answer = None;

    while final_answer.is_none() && calls < cfg.max_calls && steps < cfg.max_steps {
        calls += 1;
        let prompt =
            format!("{transcript}\nTulis satu blok kode Python. Cetak FINAL(...) bila selesai.");
        let code = root.generate(&prompt)?;
        steps += 1;

        let mut handler = |req: &RpcRequest| -> Result<JsonValue, VirError> {
            match req.method.as_str() {
                "llm_query" => {
                    let sub_prompt = req
                        .payload
                        .get("prompt")
                        .and_then(JsonValue::as_str)
                        .unwrap_or("");
                    let text = llm_query(root, sub_prompt, 1, cfg, &mut calls)?;
                    Ok(JsonValue::String(text))
                }
                "mem" => {
                    let op = req
                        .payload
                        .get("op")
                        .and_then(JsonValue::as_str)
                        .unwrap_or("");
                    mem(op, &req.payload)
                }
                other => Err(VirError::Internal(format!(
                    "metode rpc tidak dikenal: {other}"
                ))),
            }
        };

        let outcome = repl.exec(&code, &mut handler)?;
        if outcome.recovered {
            recovered = true;
        }
        if let Some(err) = &outcome.error {
            transcript.push_str(&format!("GALAT: {err}\n"));
        } else {
            transcript.push_str(&outcome.output);
            transcript.push('\n');
        }
        if let Some(pos) = outcome.output.find("FINAL(") {
            final_answer = Some(outcome.output[pos..].trim().to_string());
            break;
        }
    }

    Ok(RlmRun {
        final_answer,
        steps,
        calls_used: calls,
        recovered,
    })
}

/// Bantuan: bungkus nilai menjadi objek `{"result": ...}`.
pub fn result(value: JsonValue) -> JsonValue {
    json!({ "result": value })
}
