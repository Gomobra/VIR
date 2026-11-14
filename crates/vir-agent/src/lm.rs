//! Trait `LanguageModel` dan implementasi untuk tes/closure (F3.5).
//!
//! Root RLM dapat berupa model VIR (`vir-model`), callable Python
//! (`vir-py`), atau LM tiruan untuk tes. Trait ini adalah satu-satunya titik
//! sambung; implementasi konkret diletakkan di crate pemiliknya.

use vir_common::VirError;

/// Model bahasa: menghasilkan teks dari prompt.
pub trait LanguageModel {
    fn generate(&self, prompt: &str) -> Result<String, VirError>;
}

/// LM tiruan dengan urutan jawaban tetap (untuk tes deterministik).
#[derive(Debug, Clone)]
pub struct ScriptedLm {
    responses: Vec<String>,
    idx: std::cell::Cell<usize>,
}

impl ScriptedLm {
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            idx: std::cell::Cell::new(0),
        }
    }

    /// Berapa jawaban sudah dipakai.
    pub fn used(&self) -> usize {
        self.idx.get()
    }
}

impl LanguageModel for ScriptedLm {
    fn generate(&self, _prompt: &str) -> Result<String, VirError> {
        let i = self.idx.get();
        if let Some(response) = self.responses.get(i) {
            self.idx.set(i + 1);
            Ok(response.clone())
        } else {
            Ok(String::new())
        }
    }
}

/// LM berbasis closure; pembungkus tipis untuk callable eksternal.
pub struct FnLm<F> {
    f: F,
}

impl<F> FnLm<F>
where
    F: Fn(&str) -> String,
{
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F> LanguageModel for FnLm<F>
where
    F: Fn(&str) -> String,
{
    fn generate(&self, prompt: &str) -> Result<String, VirError> {
        Ok((self.f)(prompt))
    }
}

/// LM yang selalu gagal; berguna untuk menguji penanganan galat.
pub struct FailingLm {
    pub message: String,
}

impl LanguageModel for FailingLm {
    fn generate(&self, _prompt: &str) -> Result<String, VirError> {
        Err(VirError::Internal(self.message.clone()))
    }
}
