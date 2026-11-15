//! Agen: trait `LanguageModel`, REPL stateful, dan RLM rekursif.
//!
//! - [`LanguageModel`] adalah titik sambung model (F3.5). Implementasi nyata:
//!   [`CommandLm`] (program eksternal; root Vbeta menurut ADR-005).
//!   [`ScriptedLm`] hanya untuk tes dan flag `--scripted`.
//! - [`Repl`] adalah sesi CPython berumur panjang di atas `vir-sandbox`
//!   dengan injeksi `mem`/`llm_query` dan pemulihan crash (F3.2).
//! - [`run_rlm`] menjalankan loop RLM dengan anggaran dan penghentian (F3.4).
//! - [`MemBridge`] menjembatani operasi `mem` ke `vir-mfs` yang nyata (F3.3).

pub mod lm;
pub mod lm_command;
pub mod lm_scripted;
pub mod mem_bridge;
pub mod repl;
pub mod rlm;

pub use lm::{FailingLm, FnLm, LanguageModel, ResLm};
pub use lm_command::CommandLm;
pub use lm_scripted::ScriptedLm;
pub use mem_bridge::{MemBridge, MemOp};
pub use repl::{ExecOutcome, Repl, RpcRequest};
pub use rlm::{RlmConfig, RlmRun, llm_query, result, run_rlm};
