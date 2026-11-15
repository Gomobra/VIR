//! LM berskrip deterministik (F3.5).
//!
//! Hanya untuk tes dan flag `--scripted` yang eksplisit. Bukan model nyata:
//! jawabannya sudah ditentukan di muka.

use vir_common::VirError;

use crate::lm::LanguageModel;

/// LM tiruan dengan urutan jawaban tetap.
///
/// Setiap jawaban diteruskan apa adanya sebagai kode Python ke REPL.
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
        match self.responses.get(i) {
            Some(response) => {
                self.idx.set(i + 1);
                Ok(response.clone())
            }
            None => Ok(String::new()),
        }
    }

    fn is_scripted(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urutan_dipakai_berurutan() {
        let lm = ScriptedLm::new(vec!["a".into(), "b".into()]);
        assert_eq!(lm.generate("p").unwrap_or_default(), "a");
        assert_eq!(lm.generate("p").unwrap_or_default(), "b");
        assert_eq!(lm.used(), 2);
        assert!(lm.is_scripted());
    }

    #[test]
    fn habis_mengembalikan_teks_kosong() {
        let lm = ScriptedLm::new(vec!["a".into()]);
        assert_eq!(lm.generate("p").unwrap_or_default(), "a");
        assert_eq!(lm.generate("p").unwrap_or_default(), "");
    }
}
