//! Agen: trait `LanguageModel`, REPL stateful, dan RLM rekursif.
//!
//! - [`LanguageModel`] adalah titik sambung model (F3.5); implementasi tes
//!   ada di [`ScriptedLm`] dan [`FnLm`].
//! - [`Repl`] adalah sesi CPython berumur panjang di atas `vir-sandbox`
//!   dengan injeksi `mem`/`llm_query` dan pemulihan crash (F3.2).
//! - [`run_rlm`] menjalankan loop RLM dengan anggaran dan penghentian (F3.4).

mod lm;
mod repl;
mod rlm;

pub use lm::{FailingLm, FnLm, LanguageModel, ScriptedLm};
pub use repl::{ExecOutcome, Repl, RpcRequest};
pub use rlm::{RlmConfig, RlmRun, llm_query, result, run_rlm};
