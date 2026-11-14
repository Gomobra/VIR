//! `vir.text`: tokenizer BPE tingkat byte (PRD F1.2).

use pyo3::prelude::*;
use pyo3::types::PyType;
use vir_text::{BpeTokenizer, SPECIAL_TOKENS};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(text_encode, m)?)?;
    m.add_function(wrap_pyfunction!(text_decode, m)?)?;
    m.add_function(wrap_pyfunction!(text_special_tokens, m)?)?;
    m.add_class::<Tokenizer>()?;
    Ok(())
}

/// Encode teks dengan tokenizer kosong (byte-level).
#[pyfunction]
fn text_encode(text: &str) -> Vec<u32> {
    BpeTokenizer::new().encode(text)
}

/// Decode token dengan tokenizer kosong (byte-level).
#[pyfunction]
fn text_decode(tokens: Vec<u32>) -> String {
    BpeTokenizer::new().decode(&tokens)
}

/// Daftar token khusus (indeks awal setelah 256 byte).
#[pyfunction]
fn text_special_tokens() -> Vec<String> {
    SPECIAL_TOKENS.iter().map(|s| (*s).to_string()).collect()
}

/// Tokenizer BPE yang sudah terlatih.
#[pyclass(name = "Tokenizer")]
pub struct Tokenizer {
    inner: BpeTokenizer,
}

#[pymethods]
impl Tokenizer {
    /// Tokenizer kosong: 256 byte + token khusus.
    #[new]
    fn new() -> Self {
        Self {
            inner: BpeTokenizer::new(),
        }
    }

    /// Latih BPE dari korpus. GIL dilepas selama pelatihan (operasi berat).
    #[classmethod]
    #[pyo3(signature = (texts, vocab_size = 512))]
    fn train(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        texts: Vec<String>,
        vocab_size: usize,
    ) -> Self {
        let inner = py.allow_threads(move || {
            let mut tok = BpeTokenizer::new();
            tok.train(&texts, vocab_size);
            tok
        });
        Self { inner }
    }

    /// Token id untuk teks.
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Teks dari token id.
    fn decode(&self, tokens: Vec<u32>) -> String {
        self.inner.decode(&tokens)
    }

    /// Ukuran vocab saat ini.
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Roundtrip encode→decode identik dengan teks asal.
    fn roundtrip(&self, text: &str) -> bool {
        self.inner.decode(&self.inner.encode(text)) == text
    }
}
