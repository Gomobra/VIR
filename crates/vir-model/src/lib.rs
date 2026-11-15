//! Inti model VIR: generator NCM (PRD F1.6), rekurensi (F2.1), halting (F2.2),
//! loop blok (F2.3), sampling, dan gerbang mini F1.8.
//!
//! | Modul | Isi |
//! |---|---|
//! | [`generator`] | F1.6 — `Down_k`/`R_k` → proyeksi → prelude/inti/coda → head per level |
//! | [`scale`] | F1.4.b — operator `Down_k`/`R_k` per level |
//! | [`recurrent`] | F2.1 — `recurrent_forward`, sampling `r`, prelude/inti/coda |
//! | [`sampling`] | F1.6 task 3 — temperature, top-k, seed; CFG dan MBR |
//! | [`loss`] | F1.6/R6 — loss berbobot level, masukan teacher forcing |
//! | [`block_ar`] | F2.3 — loop blok, EOS, batas blok/token, biaya posisi |
//! | [`smoke`] | F1.8 — gerbang mini: loss turun, determinisme, loop blok |
//! | [`smoke_host`] | F1.8 — kepala host: Adam + kuadrat terkecil |
//! | [`smoke_mini`] | F1.8 — irisan penuh: VQ→generator→dekode, memorisasi 8 sampel |
//!
//! Catatan ADR (PRD F1.4.b langkah 4): `Down_k` = *average pooling* area tanpa
//! tumpang tindih, `R_k` = interpolasi linear + konvolusi residual `Phi`
//! ([`vir_vq::multiscale`]); generator memakai operator yang sama dengan
//! tokenizer agar `ẑ^(k)` konsisten antara latihan dan generasi.

pub mod block_ar;
pub mod generator;
pub mod loss;
pub mod recurrent;
pub mod sampling;
pub mod scale;
pub mod smoke;
pub mod smoke_host;
pub mod smoke_mini;
mod smoke_mini_impl;

pub use block_ar::{BlockArConfig, BlockArCost, BlockMeta, block_ar};
pub use generator::{GenerateOutput, Generator, GeneratorConfig, replace_head};
pub use loss::{level_losses, ncm_loss, teacher_inputs};
pub use recurrent::{EntropyHalter, FixedHalter, Halter, entropy, recurrent_forward, sample_r};
pub use sampling::{
    Sampler, argmax_row, cfg_logit, mbr_select, sample_codes, sample_index, softmax_row,
    top_k_indices,
};
pub use scale::LevelScales;
pub use smoke::{full_smoke, model_smoke};
pub use smoke_host::{fit_heads, fit_heads_exact, solve_lstsq};
pub use smoke_mini::{MiniSmokeConfig, SmokeResult, run_mini_smoke, synth_samples};
