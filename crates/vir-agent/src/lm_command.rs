//! `LanguageModel` berbasis proses eksternal (F3.5, ADR-005).
//!
//! Root RLM pada Vbeta bukan model VIR (Burn tidak ada di build inti), melainkan
//! callable yang benar-benar berjalan di sisi host: program + argumen yang
//! menerima prompt di stdin dan mencetak kode Python ke stdout.
//!
//! Ini bukan LM tiruan: keluaran sepenuhnya ditentukan program yang dipanggil.
//! Bila program tidak ada, gagal, atau keluarannya tidak valid, galat itulah
//! yang diteruskan — tidak ada teks karangan.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

use vir_common::VirError;

use crate::lm::LanguageModel;

/// Batas bita keluaran satu panggilan model (1 MiB).
const MAX_OUTPUT: usize = 1024 * 1024;

/// LM yang memanggil program eksternal sekali per prompt.
pub struct CommandLm {
    program: String,
    args: Vec<String>,
}

impl CommandLm {
    /// LM dari program + argumen: `program [args...]`; prompt lewat stdin.
    pub fn new(program: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            program: program.into(),
            args,
        }
    }

    /// LM yang menjalankan berkas Python dengan interpreter `python`.
    pub fn python(script: &Path, python: &str) -> Result<Self, VirError> {
        if !script.is_file() {
            return Err(VirError::Config(format!(
                "skrip model tidak ditemukan: {}",
                script.display()
            )));
        }
        let Some(path) = script.to_str() else {
            return Err(VirError::Config(format!(
                "jalur skrip model tidak valid: {}",
                script.display()
            )));
        };
        Ok(Self::new(python, vec![path.to_string()]))
    }

    /// Program yang dipanggil.
    pub fn program(&self) -> &str {
        &self.program
    }
}

impl LanguageModel for CommandLm {
    fn generate(&self, prompt: &str) -> Result<String, VirError> {
        let output = self.spawn_and_collect(prompt)?;
        read_output(&output, &self.program)
    }
}

impl CommandLm {
    fn spawn_and_collect(&self, prompt: &str) -> Result<Output, VirError> {
        let mut cmd = Command::new(&self.program);
        cmd.args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(VirError::Io)?;
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(prompt.as_bytes());
        }
        child.wait_with_output().map_err(VirError::Io)
    }
}

/// Verifikasi keluaran proses: status, batas bita, lalu teks UTF-8.
fn read_output(output: &Output, program: &str) -> Result<String, VirError> {
    if !output.status.success() {
        return Err(VirError::Internal(format!(
            "model '{program}' gagal (status {:?}): {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    if output.stdout.len() > MAX_OUTPUT {
        return Err(VirError::Internal(format!(
            "keluaran model '{program}' {} bita melebihi batas {MAX_OUTPUT}",
            output.stdout.len()
        )));
    }
    String::from_utf8(output.stdout.clone())
        .map_err(|e| VirError::Internal(format!("keluaran model '{program}' bukan UTF-8: {e}")))
}
