//! Agen: trait LanguageModel, REPL stateful, RLM rekursif.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
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

/// REPL stateful: proses CPython berumur panjang, variabel bertahan antar eksekusi.
pub struct Repl {
    child: Child,
    stdin: std::process::ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
}

impl Repl {
    pub fn start() -> Result<Self, VirError> {
        let mut child = Command::new("python3")
            .arg("-u")
            .arg("-c")
            .arg("import sys\nwhile True:\n    line=sys.stdin.readline()\n    if not line: break\n    exec(line)\n    print('__DONE__', flush=True)")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| VirError::Internal(e.to_string()))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| VirError::Internal("stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| VirError::Internal("stdout".into()))?;
        Ok(Self {
            child,
            stdin,
            reader: BufReader::new(stdout),
        })
    }

    /// Eksekusi kode, kembalikan output (sampai marker __DONE__).
    pub fn exec(&mut self, code: &str) -> Result<String, VirError> {
        writeln!(self.stdin, "{code}").map_err(|e| VirError::Internal(e.to_string()))?;
        self.stdin
            .flush()
            .map_err(|e| VirError::Internal(e.to_string()))?;
        let mut out = String::new();
        let mut line = String::new();
        loop {
            line.clear();
            let n = self
                .reader
                .read_line(&mut line)
                .map_err(|e| VirError::Internal(e.to_string()))?;
            if n == 0 {
                break;
            }
            if line.contains("__DONE__") {
                break;
            }
            out.push_str(&line);
        }
        Ok(out)
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
    }
}

impl Drop for Repl {
    fn drop(&mut self) {
        self.kill();
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

/// Panggil sub-model rekursif dengan anggaran kedalaman + jumlah panggilan.
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

pub fn run_rlm(
    root: &dyn LanguageModel,
    context: &str,
    cfg: &RlmConfig,
) -> Result<RlmRun, VirError> {
    let mut calls = 0usize;
    let mut steps = 0usize;
    let mut final_answer = None;

    while calls < cfg.max_calls && final_answer.is_none() {
        calls += 1;
        let prompt = format!("tugas: {context}\ntulis kode python yang mencetak FINAL(answer)");
        let code = root.generate(&prompt)?;
        steps += 1;

        let res: RunResult = run("python3", &["-c", &code], &Policy::default())?;
        if let Some(pos) = res.stdout.find("FINAL(") {
            final_answer = Some(res.stdout[pos..].to_string());
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

    #[test]
    fn repl_persists_state() {
        let mut repl = Repl::start().unwrap();
        let out1 = repl.exec("x = 41").unwrap();
        assert_eq!(out1.trim(), "");
        let out2 = repl.exec("print(x + 1)").unwrap();
        assert_eq!(out2.trim(), "42");
    }
}
