//! Verifier kode: jalankan kandidat dan unit test di sandbox (F4.2, F4.4).
//!
//! Kandidat ditulis sebagai modul `candidate.py` di direktori kerja sementara
//! dan — bila verifier diberi sumber tes — sumber tes sebagai `test_candidate.py`
//! di direktori yang sama, lalu dijalankan dengan `python3 -m unittest`.
//! Eksekusi memakai [`vir_sandbox::run`] dengan [`Policy`]: rlimit, timeout
//! dinding, dan (bila fitur `sandbox` aktif) Landlock + seccomp. `Policy::workdir`
//! selalu disetel ke direktori kandidat, sehingga konfinemen kernel tidak pernah
//! hilang diam-diam karena workdir lupa diisi.
//!
//! Skor bersifat biner: `1.0` hanya bila proses keluar 0 **dan** semua tes lulus.
//! Keluaran tes yang tidak dapat dibaca (`Ran N tests`) atau `N < min_tests`
//! dianggap gagal; tanpa pagar ini kumpulan tes kosong akan memberi skor `1.0`
//! (reward hacking, PRD F4.3 butir 5). `task` dipakai sebagai label laporan,
//! bukan sebagai sumber tes: sumber tes adalah kebijakan verifier, bukan masukan
//! yang bisa ditulis model.

use std::path::Path;
use std::time::Duration;

use vir_common::VirError;
use vir_sandbox::{Policy, RunResult, run};

use crate::verifier::Verifier;

const CANDIDATE_FILE: &str = "candidate.py";
const TEST_FILE: &str = "test_candidate.py";
const TEST_MODULE: &str = "test_candidate";
const DEFAULT_TIMEOUT_SECS: u64 = 10;
/// Minimal jumlah tes yang harus terbaca dari keluaran agar kandidat lulus.
const DEFAULT_MIN_TESTS: usize = 1;

/// Hasil eksekusi satu kandidat.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    /// Label tugas yang dipakai saat pemanggilan.
    pub task: String,
    pub passed: bool,
    pub exit_code: Option<i32>,
    /// Proses melewati batas waktu dan dibunuh.
    pub timed_out: bool,
    /// Jumlah tes yang dilaporkan `unittest`; `None` bila tak terbaca.
    pub tests_run: Option<usize>,
    pub stdout: String,
    pub stderr: String,
}

/// Verifier kode yang menjalankan kandidat di sandbox.
#[derive(Debug, Clone)]
pub struct CodeVerifier {
    interpreter: String,
    policy: Policy,
    /// Sumber modul tes (`unittest`); `None` berarti jalankan skrip kandidat saja.
    tests: Option<String>,
    min_tests: usize,
}

impl Default for CodeVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Kebijakan bawaan: timeout dinding 10 s dan konfinemen kernel mengikuti
/// `vir-sandbox` (aktif bila fitur `sandbox` dikompilasi).
fn default_policy() -> Policy {
    Policy {
        timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        ..Policy::default()
    }
}

impl CodeVerifier {
    /// Verifier dengan `python3`, timeout 10 s, dan kebijakan bawaan sandbox.
    pub fn new() -> Self {
        Self {
            interpreter: "python3".to_owned(),
            policy: default_policy(),
            tests: None,
            min_tests: DEFAULT_MIN_TESTS,
        }
    }

    /// Ganti program interpreter (mis. `python3.11` di host yang minim).
    pub fn with_interpreter(mut self, interpreter: impl Into<String>) -> Self {
        self.interpreter = interpreter.into();
        self
    }

    /// Pasang sumber modul tes berbasis `unittest` yang mengimpor `candidate`.
    pub fn with_tests(mut self, source: impl Into<String>) -> Self {
        self.tests = Some(source.into());
        self
    }

    /// Minimal jumlah tes yang harus terbaca; bawaan 1.
    pub fn with_min_tests(mut self, min_tests: usize) -> Self {
        self.min_tests = min_tests;
        self
    }

    /// Ganti kebijakan eksekusi (timeout, batas sumber daya, konfinemen).
    pub fn with_policy(mut self, policy: Policy) -> Self {
        self.policy = policy;
        self
    }

    /// Kebijakan eksekusi; `workdir` disetel per kandidat, bukan di sini.
    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    /// Jalankan kandidat dan kembalikan laporan mentah (skor: [`Verifier::score`]).
    pub fn run(&self, task: &str, candidate: &str) -> Result<VerifyReport, VirError> {
        let dir = tempfile::TempDir::new().map_err(VirError::Io)?;
        let root = dir.path();
        write_file(root, CANDIDATE_FILE, candidate)?;
        let args = self.args(root)?;
        let policy = self.policy_for(root);
        let out = run(&self.interpreter, &args, &policy)?;
        Ok(self.report(task, &out))
    }

    /// Argumen interpreter: skrip kandidat, atau modul tes bila ada.
    fn args(&self, root: &Path) -> Result<Vec<&str>, VirError> {
        match self.tests.as_deref() {
            Some(source) => {
                write_file(root, TEST_FILE, source)?;
                Ok(vec!["-m", "unittest", "-v", TEST_MODULE])
            }
            None => Ok(vec![CANDIDATE_FILE]),
        }
    }

    /// Kebijakan dengan `workdir` diarahkan ke direktori kandidat.
    fn policy_for(&self, workdir: &Path) -> Policy {
        let mut policy = self.policy.clone();
        policy.workdir = Some(workdir.to_path_buf());
        policy
    }

    fn report(&self, task: &str, out: &RunResult) -> VerifyReport {
        let tests_run = count_tests(out);
        VerifyReport {
            task: task.to_owned(),
            passed: self.passed(out, tests_run),
            exit_code: out.exit_code,
            timed_out: out.timed_out,
            tests_run,
            stdout: out.stdout.clone(),
            stderr: out.stderr.clone(),
        }
    }

    /// Lulus = keluar 0, tidak timeout, dan (bila ada tes) jumlah tes cukup.
    fn passed(&self, out: &RunResult, tests_run: Option<usize>) -> bool {
        if out.timed_out || out.exit_code != Some(0) {
            return false;
        }
        match (self.tests.is_some(), tests_run) {
            (false, _) => true,
            (true, Some(ran)) => ran >= self.min_tests,
            // Keluaran tes tak terbaca (mis. terpotong): jangan percaya.
            (true, None) => false,
        }
    }
}

impl Verifier for CodeVerifier {
    fn score(&self, task: &str, candidate: &str) -> Result<f32, VirError> {
        let report = self.run(task, candidate)?;
        Ok(if report.passed { 1.0 } else { 0.0 })
    }
}

/// Tulis satu berkas ke direktori kerja kandidat.
fn write_file(dir: &Path, name: &str, source: &str) -> Result<(), VirError> {
    let path = dir.join(name);
    std::fs::write(path, source).map_err(VirError::Io)
}

/// Baca `Ran N tests` dari keluaran `unittest` (stderr, cadangan stdout).
fn count_tests(out: &RunResult) -> Option<usize> {
    out.stderr
        .lines()
        .chain(out.stdout.lines())
        .find_map(|line| {
            line.strip_prefix("Ran ")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}
