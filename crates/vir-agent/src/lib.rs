//! Agen: trait LanguageModel, REPL, RLM.

use vir_common::VirError;
use vir_sandbox::{Policy, RunResult, run};

pub trait LanguageModel {
    fn generate(&self, prompt: &str) -> Result<String, VirError>;
}

#[derive(Debug, Clone)]
pub struct ScriptedLm {
    pub responses: Vec<String>,
    pub idx: std::cell::Cell<usize>,
}

impl ScriptedLm {
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            idx: std::cell::Cell::new(0),
        }
    }
}

impl LanguageModel for ScriptedLm {
    fn generate(&self, _prompt: &str) -> Result<String, VirError> {
        let i = self.idx.get();
        if i < self.responses.len() {
            self.idx.set(i + 1);
            Ok(self.responses[i].clone())
        } else {
            Ok(String::new())
        }
    }
}

/// RLM: loop root menulis kode, dijalankan di REPL (Python), sampai FINAL(...) atau anggaran habis.
pub struct RlmConfig {
    pub max_depth: usize,
    pub max_calls: usize,
}

pub struct RlmRun {
    pub final_answer: Option<String>,
    pub steps: usize,
    pub calls_used: usize,
}

pub fn run_rlm(
    root: &dyn LanguageModel,
    _context: &str,
    cfg: &RlmConfig,
) -> Result<RlmRun, VirError> {
    let mut calls = 0usize;
    let mut steps = 0usize;
    let mut final_answer = None;

    // root menghasilkan satu kode; coba jalankan di python REPL sederhana
    while calls < cfg.max_calls && final_answer.is_none() {
        calls += 1;
        let prompt = "tulis kode python yang mencetak FINAL(answer)";
        let code = root.generate(prompt)?;
        steps += 1;

        // jalankan kode di sandbox python
        let res: RunResult = run("python3", &["-c", &code], &Policy::default())?;
        // deteksi FINAL(...)
        if let Some(pos) = res.stdout.find("FINAL(") {
            let rest = &res.stdout[pos..];
            final_answer = Some(rest.to_string());
            break;
        }
        if steps >= cfg.max_depth {
            break;
        }
    }

    Ok(RlmRun {
        final_answer,
        steps,
        calls_used: calls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripted_lm_cycles() {
        let lm = ScriptedLm::new(vec!["a".into(), "b".into()]);
        assert_eq!(lm.generate("x").unwrap(), "a");
        assert_eq!(lm.generate("x").unwrap(), "b");
        assert_eq!(lm.generate("x").unwrap(), "");
    }

    #[test]
    fn rlm_runs_with_scripted() {
        let lm = ScriptedLm::new(vec!["print('FINAL(42)')".into()]);
        let cfg = RlmConfig {
            max_depth: 2,
            max_calls: 2,
        };
        let r = run_rlm(&lm, "ctx", &cfg).unwrap();
        assert!(r.final_answer.is_some());
        assert!(r.final_answer.unwrap().contains("FINAL("));
    }
}
