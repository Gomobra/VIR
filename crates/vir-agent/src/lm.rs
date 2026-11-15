//! Titik sambung model (F3.5).
//!
//! Trait [`LanguageModel`] adalah satu-satunya antarmuka yang dipakai RLM.
//! Implementasi konkret tinggal di modulnya masing-masing.

use vir_common::VirError;

pub use crate::lm_command::CommandLm;
pub use crate::lm_scripted::ScriptedLm;

/// Model bahasa: menghasilkan teks dari prompt.
pub trait LanguageModel {
    /// Hasilkan satu respons untuk `prompt`.
    fn generate(&self, prompt: &str) -> Result<String, VirError>;

    /// `true` bila LM ini berskrip (bukan model nyata); bawaan `false`.
    fn is_scripted(&self) -> bool {
        false
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
