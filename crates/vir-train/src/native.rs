//! Trainer **GPU-native** (F2.5 optimasi #1): model Burn tetap di device, bobot
//! TIDAK disalin ke host tiap langkah.
//!
//! # Masalah yang dipecahkan (R1 — terukur dari kode)
//!
//! [`crate::trainer::Trainer`] memakai `ParamStore` host (`Vec<f32>`) sebagai
//! sumber kebenaran bobot. Satu langkah `VirStepFn::compute` lalu melakukan:
//!
//! ```text
//! load_store    : host f32 → tensor device   (Tensor::from_floats, SEMUA param)
//! forward + backward
//! collect_grads : grad device → host Vec<f32> (to_data().try_to_vec, SEMUA grad)
//! copy_back     : model → store untuk slot Tokenizer (satu D2H lagi)
//! ```
//!
//! Untuk model ~100M parameter (dim 384, K 12) itu ratusan MB per langkah lewat
//! PCIe, di atas biaya kernel. Jalur ini menghapus ketiga transfer tersebut:
//!
//! ```text
//! batch (host Vec<i64>, hanya data) → tensor device
//! forward → loss → loss.backward()
//! grads = GradientsParams::from_params(&mut grads, &model, &keep)
//! model = opt.step(lr, model, grads)     // AdamW + grup Muon, di device
//! ```
//!
//! | aspek | `Trainer` (ParamStore) | `NativeTrainer` |
//! |---|---|---|
//! | sumber bobot | `ParamStore` host | model di device |
//! | H2D per langkah | semua param | batch token saja |
//! | D2H per langkah | semua grad + codebook | **nol** (metrik hanya saat log) |
//! | optimizer | Muon/AdamW manual (`crate::optim`) | `ModuleOptimizer` Burn (`burn_bridge`) |
//! | checkpoint | `state.json` + `tensors.bin` | `native_state.json` + `model.safetensors` |
//!
//! Jalur host tetap dipertahankan utuh: tes byte-exact resume dan
//! [`crate::ddp`] memakainya.
//!
//! # Pembekuan tahap
//!
//! `ModuleOptimizer` hanya punya dua grup (AdamW default + Muon berdasarkan
//! predikat path) dan tidak punya konsep "beku". Karena itu gradien parameter
//! beku tidak pernah dimasukkan ke `GradientsParams` yang diberikan ke
//! `opt.step` (daftar `ParamId` non-beku di [`non_frozen_param_ids`]). Tanpa
//! grad, `ModuleOptimizerMapper` tidak menyentuh parameter tersebut sama
//! sekali — bobot dan state optimizernya tidak berubah.
//!
//! # Metrik hanya saat log (W1)
//!
//! Metrik host (per_level, perplexity, utilization, recon, codebook stats)
//! menuntut sinkronisasi device→host. Karena hanya dibaca oleh log JSONL dan
//! gerbang, metrik **tidak** dihitung tiap langkah: `native_forward` hanya
//! mengembalikan loss tracked + bahan metrik bila langkah itu memang jatuh
//! tempo (`log_every` atau evaluasi gerbang). Ekstraksi metrik (satu-satunya
//! `into_scalar`/histogram) terjadi di [`NativeTrainer::log_step`]. Efek
//! sampingnya: graf autodiff tidak ditahan antar langkah.
//!
//! # Pengukuran yang adil (W0)
//!
//! Kompilasi kernel dan autotune terjadi pada langkah pertama; tok/s yang
//! dihitung sejak konstruksi trainer karena itu terlalu rendah. [`set_warmup`]
//! menjalankan N langkah nyata (forward/backward/step) yang tidak menghitung
//! metrik, tidak menulis log, dan tidak menambah `tokens_done`. Jam tok/s
//! dimulai ulang setelah warmup, dan [`StepRecord::dt_ms`] mencatat durasi satu
//! langkah penuh agar steady-state bisa diukur dari JSONL.
//!
//! # State optimizer pada resume
//!
//! `model.safetensors` hanya memuat bobot. State AdamW/Muon milik
//! `ModuleOptimizer` **tidak** ikut; resume membangun optimizer baru (state
//! nol). Konsekuensinya langkah pertama setelah resume bukan langkah identik
//! dengan jalan tanpa henti. Jalur host (`crate::trainer::Trainer`) tetap
//! menyimpan seluruh state optimizer byte demi byte, jadi pemakainya yang
//! butuh resume identik memakai jalur itu.

use std::time::{Duration, Instant};

use burn::module::{Module, Param, ParamId};
use burn::optim::{GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use burn::tensor::{Device, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use vir_common::VirError;
use vir_vq::{Codebook, MultiscaleVq, Quantizer, tensor_accuracy, vq_loss_chunked};

use crate::burn_bridge::{BurnHybridSettings, burn_optimizer_with_paths};
use crate::checkpoint::{CHECKPOINT_VERSION, LOG_FILE};
use crate::corruption::Corruption;
use crate::data::{BatchSource, ShardBatchSource};
use crate::logging::{EventRecord, JsonlLogger, StepRecord};
use crate::model::{VirTrainConfig, VirTrainModel, train_device, train_device_dtype};
use crate::optim::ParamRole;
use crate::params;
use crate::rng::Rng;
use crate::trainer::{GateMetric, GateTracker, RunOutcome, Stage, StepOut, TrainConfig};

/// Nama berkas bobot safetensors di dalam direktori checkpoint native.
pub const MODEL_FILE: &str = "model.safetensors";
/// Nama berkas metadata di dalam direktori checkpoint native.
pub const NATIVE_STATE_FILE: &str = "native_state.json";

// ---------------------------------------------------------------------------
// Safetensors (writer + reader, tanpa dependensi baru; R7/R10)
// ---------------------------------------------------------------------------

/// Serialisasi safetensors dari daftar tensor bernama.
///
/// Format: `u64 LE` panjang header → header JSON
/// `{name: {dtype, shape, data_offsets}}` (di-pad spasi agar panjang header +
/// 8 kelipat 8) → buffer data little-endian `f32`. Backend float Burn 0.22 di
/// semua device yang dipakai di sini beraliran `f32`, jadi `F32` cukup.
pub fn write_safetensors(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Result<Vec<u8>, VirError> {
    let mut header = String::from("{");
    let mut offset = 0usize;
    for (i, (name, shape, data)) in tensors.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let dims = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            escape_json(name),
            dims,
            offset,
            offset + data.len() * 4
        ));
        offset += data.len() * 4;
    }
    header.push('}');
    while (header.len() + 8) % 8 != 0 {
        header.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + offset);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for (_, _, data) in tensors {
        for v in data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(out)
}

/// Baca safetensors menjadi `(nama, shape, nilai)`.
pub fn read_safetensors(bytes: &[u8]) -> Result<Vec<(String, Vec<usize>, Vec<f32>)>, VirError> {
    if bytes.len() < 8 {
        return Err(VirError::Internal("safetensors terlalu pendek".into()));
    }
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap_or([0; 8])) as usize;
    if 8 + hlen > bytes.len() {
        return Err(VirError::Internal(
            "header safetensors di luar berkas".into(),
        ));
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen])
        .map_err(|e| VirError::Internal(format!("header safetensors rusak: {e}")))?;
    let obj = header
        .as_object()
        .ok_or_else(|| VirError::Internal("header safetensors bukan objek".into()))?;
    let base = 8 + hlen;
    let mut out = Vec::with_capacity(obj.len());
    for (name, spec) in obj {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = spec
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|v| v as usize))
                    .collect()
            })
            .unwrap_or_default();
        let range = spec
            .get("data_offsets")
            .and_then(|s| s.as_array())
            .ok_or_else(|| VirError::Internal(format!("offsets hilang: {name}")))?;
        let start = range.first().and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let end = range.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        if start > end || end - start > bytes.len() - base || (end - start) % 4 != 0 {
            return Err(VirError::Internal(format!(
                "rentang data safetensors tidak sah: {name}"
            )));
        }
        let mut data = Vec::with_capacity((end - start) / 4);
        for i in (start..end).step_by(4) {
            let b: [u8; 4] = bytes[base + i..base + i + 4].try_into().unwrap_or([0; 4]);
            data.push(f32::from_le_bytes(b));
        }
        out.push((name.clone(), shape, data));
    }
    Ok(out)
}

/// Escape nilai untuk kunci JSON.
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Peran + pembekuan gradien
// ---------------------------------------------------------------------------

/// Kumpulkan `ParamId` parameter yang TIDAK beku pada tahap `stage` (untuk
/// `GradientsParams::from_params` — hanya non-beku yang diikutkan ke optimizer).
///
/// Path diambil dari `enter_module`/`exit_module` (derive tidak memancarkan
/// `visit_float_with_path`), jadi konsisten dengan `crate::params`.
pub fn non_frozen_param_ids(model: &VirTrainModel, stage: Stage) -> Vec<ParamId> {
    struct Finder {
        stage: Stage,
        path: Vec<String>,
        ids: Vec<ParamId>,
    }
    impl burn::module::ModuleVisitor for Finder {
        fn enter_module(&mut self, name: &str, _c: &str) {
            self.path.push(name.to_string());
        }
        fn exit_module(&mut self, _n: &str, _c: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let frozen = match params::role_for_path(&self.path.join(".")) {
                ParamRole::Tokenizer => true,
                ParamRole::Generator => self.stage != Stage::Generator,
                ParamRole::Vq => self.stage == Stage::Generator,
            };
            if !frozen {
                self.ids.push(param.id);
            }
        }
    }
    let mut f = Finder {
        stage,
        path: Vec::new(),
        ids: Vec::new(),
    };
    model.visit(&mut f);
    f.ids
}

// ---------------------------------------------------------------------------
// Langkah native
// ---------------------------------------------------------------------------

/// Konteks satu langkah native (RNG, korupsi, sumber data).
pub struct NativeStepCtx<'a> {
    /// Tahap aktif.
    pub stage: Stage,
    /// Langkah global.
    pub step: u64,
    /// Langkah sejak masuk tahap.
    pub stage_step: u64,
    /// Laju pembelajaran AdamW.
    pub lr: f64,
    /// RNG deterministik (ikut checkpoint).
    pub rng: &'a mut Rng,
    /// Korupsi aktif (ikut checkpoint).
    pub corruption: Option<&'a mut Corruption>,
}

/// Hasil satu langkah native.
pub struct NativeStepOut {
    /// Loss tracked (masih di device; dipakai `backward`).
    pub loss: Tensor<1>,
    /// Jumlah token yang diproses (`B·L`).
    pub tokens: usize,
    /// Bahan metrik; `Some` hanya pada langkah yang memang mengukur (W1).
    pub metrics: Option<RawMetrics>,
}

/// Bahan metrik satu langkah, masih di device.
///
/// Hanya ada pada langkah yang mengukur (log/gerbang), sehingga tidak ada
/// sinkronisasi device→host pada langkah biasa.
pub enum RawMetrics {
    /// Tahap 1 (VQ + decoder).
    Vq(VqMetrics),
    /// Tahap 2 (generator NCM).
    Gen(GenMetrics),
}

impl RawMetrics {
    /// Ekstrak metrik host dari bahan metrik.
    ///
    /// Satu-satunya tempat `into_scalar`/histogram kode dipanggil; hanya
    /// dijalankan dari [`NativeTrainer::log_step`] (tiap `log_every`).
    pub fn to_step_out(&self, model: &VirTrainModel) -> StepOut {
        match self {
            Self::Vq(m) => vq_metrics(m, model),
            Self::Gen(m) => gen_metrics(m, model),
        }
    }
}

/// Bahan metrik tahap 1 (`vq_forward`).
pub struct VqMetrics {
    /// Laten `z [B, L, d]`.
    pub z: Tensor<3>,
    /// Rekonstruksi kumulatif per level `ẑ^(k)`.
    pub levels: Vec<Tensor<3>>,
    /// Indeks kode per level.
    pub indices: Vec<Tensor<2, Int>>,
    /// Token hasil argmax decoder `[B, L]`.
    pub pred: Tensor<2, Int>,
    /// Token target `[B, L]`.
    pub target: Tensor<2, Int>,
    /// Token valid per level (`B·ℓ_k`).
    pub level_tokens: Vec<f32>,
}

/// Bahan metrik tahap 2 (`gen_forward`).
pub struct GenMetrics {
    /// Suku loss berbobot per level (`w_k·CE_k`).
    pub terms: Vec<Tensor<1>>,
    /// Bobot level (`w_k`), untuk `perplexity = exp(CE_k)`.
    pub weights: Vec<f32>,
    /// Kode bersih per level (target NCM).
    pub indices: Vec<Tensor<2, Int>>,
    /// Laju korupsi pada batch ini.
    pub rate: Option<f32>,
    /// Token valid per level (`B·ℓ_k`).
    pub level_tokens: Vec<f32>,
}

/// Satu langkah forward native: model (device) + batch → loss tracked.
///
/// `want_metrics` menyertakan bahan metrik (W1). Bahan itu hanya berlaku untuk
/// langkah yang jatuh tempo log/gerbang, jadi langkah biasa tidak menahan graf
/// autodiff dan tidak menyinkronkan apa pun ke host.
pub fn native_forward(
    model: &mut VirTrainModel,
    ctx: &mut NativeStepCtx<'_>,
    cfg: &VirTrainConfig,
    weights: &[f32],
    source: &mut ShardBatchSource,
    device: &Device,
    want_metrics: bool,
) -> Result<NativeStepOut, VirError> {
    let (tokens, shape) = source
        .next_batch()
        .map_err(|e| VirError::Internal(format!("batch: {e}")))?;
    if shape[1] != cfg.total_len() {
        return Err(VirError::Internal(format!(
            "L batch {} != L model {}",
            shape[1],
            cfg.total_len()
        )));
    }
    let target = Tensor::<2, Int>::from_ints(TensorData::new(tokens, shape), device);
    match ctx.stage {
        Stage::Vq => vq_forward(model, ctx, cfg, &target, want_metrics),
        Stage::Generator => gen_forward(model, ctx, cfg, weights, &target, want_metrics),
    }
}

/// Tahap 1: VQ + decoder (R4).
fn vq_forward(
    model: &mut VirTrainModel,
    ctx: &mut NativeStepCtx<'_>,
    cfg: &VirTrainConfig,
    target: &Tensor<2, Int>,
    want_metrics: bool,
) -> Result<NativeStepOut, VirError> {
    let z = model.embed_tokens(target);
    let enc = model
        .encode_with_ema(z.clone(), ctx.rng)
        .map_err(|e| VirError::Internal(format!("encode EMA: {e}")))?;
    let loss = vq_loss_chunked(
        &model.dec,
        z.clone(),
        enc.z_hat.clone(),
        target.clone(),
        cfg.commitment_cost,
        cfg.chunk,
    );
    let tokens = target.dims().iter().product::<usize>();
    let metrics = want_metrics.then(|| {
        RawMetrics::Vq(VqMetrics {
            pred: model.dec.argmax_tokens(enc.z_hat.clone()),
            z,
            levels: enc.levels,
            indices: enc.indices,
            target: target.clone(),
            level_tokens: level_tokens(cfg, target.dims()[0]),
        })
    });
    Ok(NativeStepOut {
        loss,
        tokens,
        metrics,
    })
}

/// Tahap 2: generator NCM dengan tokenizer beku (R6).
fn gen_forward(
    model: &mut VirTrainModel,
    ctx: &mut NativeStepCtx<'_>,
    cfg: &VirTrainConfig,
    weights: &[f32],
    target_tokens: &Tensor<2, Int>,
    want_metrics: bool,
) -> Result<NativeStepOut, VirError> {
    let z = model.embed_tokens(target_tokens).detach();
    let enc = model.encode(z);
    let clean = enc.indices;
    let (corrupted, rate) = corrupt(&clean, ctx, cfg);
    let inputs = teacher_inputs(model, &corrupted, cfg.dim, &target_tokens.device());
    let logits = model.generator.forward(&inputs);
    // Satu CE per level: dipakai untuk loss tracked DAN metrik (hindari hitung
    // CE dua kali — mahal pada vocab besar). `terms` = `w_k·CE_k` per level.
    let terms = vir_model::level_losses(&logits, &clean, weights);
    let loss = sum_terms(&terms, &target_tokens.device());
    let tokens = target_tokens.dims().iter().product::<usize>();
    let metrics = want_metrics.then(|| {
        RawMetrics::Gen(GenMetrics {
            terms,
            weights: weights.to_vec(),
            indices: clean,
            rate,
            level_tokens: level_tokens(cfg, target_tokens.dims()[0]),
        })
    });
    Ok(NativeStepOut {
        loss,
        tokens,
        metrics,
    })
}

/// Jumlah suku loss berbobot; nol bila daftar kosong.
fn sum_terms(terms: &[Tensor<1>], device: &Device) -> Tensor<1> {
    let mut total: Option<Tensor<1>> = None;
    for term in terms {
        total = Some(match total.take() {
            Some(acc) => acc + term.clone(),
            None => term.clone(),
        });
    }
    total.unwrap_or_else(|| Tensor::zeros([1], device))
}

/// Korupsi kode masukan (PRD F1.7 langkah 2); target tetap bersih.
fn corrupt(
    clean: &[Tensor<2, Int>],
    ctx: &mut NativeStepCtx<'_>,
    cfg: &VirTrainConfig,
) -> (Vec<Tensor<2, Int>>, Option<f32>) {
    let Some(corruption) = ctx.corruption.as_deref_mut() else {
        return (clean.to_vec(), None);
    };
    let batch = clean.first().map_or(0, |t| t.dims()[0]);
    let mut batches = Vec::with_capacity(clean.len());
    for k in 0..clean.len() {
        batches.push(corruption.next_batch(k, batch, cfg.level_len(k), cfg.codebook_size));
    }
    let out = corruption.apply_levels(clean, &batches);
    (out, Some(corruption.mean_measured()))
}

/// Masukan teacher forcing: `z̃^(k) = z̃^(k−1) + R_k(e_k(r̃_k))`.
fn teacher_inputs(
    model: &VirTrainModel,
    idx: &[Tensor<2, Int>],
    dim: usize,
    device: &Device,
) -> Vec<Tensor<3>> {
    let batch = idx.first().map_or(0, |t| t.dims()[0]);
    let ctx = model.total_len();
    let mut cur = Tensor::<3>::zeros([batch, ctx, dim], device);
    let mut out = Vec::with_capacity(idx.len());
    for (k, codes) in idx.iter().enumerate() {
        out.push(cur.clone());
        let e = model.generator.embed_codes(k, codes.clone());
        cur = cur + model.generator.reconstruct_level(k, e);
    }
    out
}

/// MSE rekonstruksi kumulatif per level: `‖z − ẑ^(k)‖²`.
fn recon_errors(z: &Tensor<3>, levels: &[Tensor<3>]) -> Vec<f32> {
    levels
        .iter()
        .map(|zh| {
            let d = zh.clone() - z.clone();
            (d.clone() * d).mean().into_scalar::<f32>()
        })
        .collect()
}

/// Jumlah posisi token per level (`B·ℓ_k`).
fn level_tokens(cfg: &VirTrainConfig, batch: usize) -> Vec<f32> {
    cfg.levels.iter().map(|l| (batch * l) as f32).collect()
}

/// Utilisasi dan perplexity codebook per level.
fn codebook_stats(vq: &MultiscaleVq<Codebook>, indices: &[Tensor<2, Int>]) -> (Vec<f32>, Vec<f32>) {
    let mut utilization = Vec::with_capacity(indices.len());
    let mut perplexity = Vec::with_capacity(indices.len());
    for (k, idx) in indices.iter().enumerate() {
        let [b, l] = idx.dims();
        let stats = vq.levels[k].quantizer.stats(&idx.clone().reshape([b * l]));
        utilization.push(stats.utilization);
        perplexity.push(stats.perplexity);
    }
    (utilization, perplexity)
}

/// Metrik tahap 1 dari bahan `vq_forward` (sinkronisasi D2H).
fn vq_metrics(m: &VqMetrics, model: &VirTrainModel) -> StepOut {
    let recon = tensor_accuracy(&m.pred, &m.target) as f32;
    let per_level = recon_errors(&m.z, &m.levels);
    let (utilization, perplexity) = codebook_stats(&model.vq, &m.indices);
    StepOut {
        loss_total: 0.0,
        per_level,
        level_tokens: m.level_tokens.clone(),
        utilization,
        perplexity,
        recon: Some(recon),
        corruption_rate: None,
    }
}

/// Metrik tahap 2 dari bahan `gen_forward` (sinkronisasi D2H).
fn gen_metrics(m: &GenMetrics, model: &VirTrainModel) -> StepOut {
    let (per_level, perplexity) = level_metrics(&m.terms, &m.weights);
    let (utilization, _) = codebook_stats(&model.vq, &m.indices);
    StepOut {
        loss_total: 0.0,
        per_level,
        level_tokens: m.level_tokens.clone(),
        utilization,
        perplexity,
        recon: None,
        corruption_rate: m.rate,
    }
}

/// Loss per level (sudah berbobot) dan perplexity `exp(CE_k)` dari suku
/// `w_k·CE_k`. Rumus identik dengan versi yang dihitung tiap langkah.
fn level_metrics(terms: &[Tensor<1>], weights: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut per_level = Vec::with_capacity(terms.len());
    let mut perplexity = Vec::with_capacity(terms.len());
    for (k, term) in terms.iter().enumerate() {
        per_level.push(term.clone().into_scalar::<f32>());
        let w = weights.get(k).copied().unwrap_or(0.0).max(f32::EPSILON);
        let ce = term.clone().div_scalar(w).into_scalar::<f32>();
        perplexity.push(ce.exp());
    }
    (per_level, perplexity)
}

// ---------------------------------------------------------------------------
// NativeTrainer
// ---------------------------------------------------------------------------

/// Trainer GPU-native: satu loop forward/backward/step di device.
pub struct NativeTrainer {
    /// Konfigurasi latihan (dua tahap, jadwal, gerbang).
    pub config: TrainConfig,
    /// Model di device (tidak pernah disalin ke host).
    pub model: VirTrainModel,
    /// Optimizer Burn (AdamW + grup Muon).
    pub opt: ModuleOptimizer,
    /// Konfigurasi model.
    pub model_cfg: VirTrainConfig,
    /// Sumber batch (data host → tensor device).
    pub source: ShardBatchSource,
    /// Device aktif.
    pub device: Device,
    /// Bobot level NCM.
    weights: Vec<f32>,
    /// Tahap saat ini.
    pub stage: Stage,
    /// Langkah global.
    pub step: u64,
    /// Langkah sejak masuk tahap.
    pub stage_step: u64,
    /// RNG deterministik (ikut checkpoint).
    pub rng: Rng,
    /// Korupsi aktif.
    pub corruption: Option<Corruption>,
    /// State gerbang.
    pub gate: GateTracker,
    /// Logger JSONL.
    logger: Option<JsonlLogger>,
    /// Direktori checkpoint.
    ckpt_dir: String,
    /// Jumlah transisi tahap.
    pub transitions: u32,
    /// Loss total langkah terakhir (host; disinkronkan saat log/akhir run).
    pub last_loss: f32,
    /// Metrik rekonstruksi terbaik (gerbang).
    pub best_recon: f32,
    /// Token yang sudah diproses kumulatif (tidak termasuk warmup).
    pub tokens_done: u64,
    /// Jumlah gradien beku yang dibuang pada langkah terakhir.
    last_dropped: usize,
    /// Tokenizer sudah disalin ke generator?
    tokenizer_loaded: bool,
    /// Jam mulai untuk takar tok/s (dimulai ulang setelah warmup).
    started: Instant,
    /// Tok/s eksponensial (EMA) agar angka stabil.
    tok_s: f64,
    /// Ambang waktu untuk langkah berikutnya (debounce pengukuran).
    last_sample: Instant,
    /// Token pada pengukuran sebelumnya.
    last_sample_tokens: u64,
    /// Cache `(tahap, ParamId non-beku)`; hanya dihitung saat tahap berubah (B11).
    keep: (Stage, Vec<ParamId>),
    /// Loss tracked langkah terakhir; di-sync saat log dan sekali di akhir run.
    last_loss_t: Option<Tensor<1>>,
    /// Sisa langkah warmup (W0); dijalankan sekali, lalu dinolkan.
    warmup: u64,
    /// EWMA durasi forward (ms) — profil fase (R1).
    fw_ms: f32,
    /// EWMA durasi backward (ms) — profil fase (R1).
    bw_ms: f32,
    /// EWMA durasi optimizer step (ms) — profil fase (R1).
    op_ms: f32,
}

impl NativeTrainer {
    /// Rakit trainer native dari konfigurasi model, sumber batch, dan device.
    ///
    /// Bobot awal identik dengan [`crate::burn_step::VirStepFn`] karena memakai
    /// [`VirTrainModel::new`] dengan seed yang sama.
    pub fn new(
        cfg: VirTrainConfig,
        source: ShardBatchSource,
        device: Device,
        stage: Stage,
    ) -> Result<Self, VirError> {
        cfg.validate()?;
        let tcfg = TrainConfig {
            stage,
            ..TrainConfig::default()
        };
        Self::with_train_config(cfg, source, device, tcfg)
    }

    /// Rakit trainer native dengan `TrainConfig` eksplisit.
    pub fn with_train_config(
        cfg: VirTrainConfig,
        source: ShardBatchSource,
        device: Device,
        tcfg: TrainConfig,
    ) -> Result<Self, VirError> {
        Self::with_train_config_dtype(cfg, source, device, tcfg, false)
    }

    /// Rakit trainer native dengan `TrainConfig` + pilihan dtype (bf16/fp32).
    ///
    /// `bf16 = true` mengonfigurasi device ke bfloat16 (tensor core) sebelum
    /// model dibuat (PRD: BF16). Default `false` = fp32.
    pub fn with_train_config_dtype(
        cfg: VirTrainConfig,
        source: ShardBatchSource,
        device: Device,
        tcfg: TrainConfig,
        bf16: bool,
    ) -> Result<Self, VirError> {
        tcfg.validate()?;
        if source.seq_len() != cfg.total_len() {
            return Err(VirError::Config(format!(
                "L batch {} != L model {}",
                source.seq_len(),
                cfg.total_len()
            )));
        }
        let device = train_device_dtype(device, cfg.seed, bf16)?;
        let model = VirTrainModel::new(&cfg, &device)?;
        let opt = burn_optimizer_with_paths(
            &BurnHybridSettings::default(),
            crate::params::two_d_param_paths(&model),
        );
        let weights = crate::loss::level_weight(&cfg.levels, cfg.alpha);
        let gate = match tcfg.gate {
            Some(g) => GateTracker::new(g.metric),
            None => GateTracker::new(GateMetric::TokenAccuracy),
        };
        let now = Instant::now();
        let stage0 = tcfg.stage;
        let seed = cfg.seed;
        let keep = (stage0, non_frozen_param_ids(&model, stage0));
        Ok(Self {
            config: tcfg,
            model,
            opt,
            model_cfg: cfg,
            source,
            device,
            weights,
            stage: stage0,
            step: 0,
            stage_step: 0,
            rng: Rng::new(seed),
            corruption: None,
            gate,
            logger: None,
            ckpt_dir: String::new(),
            transitions: 0,
            last_loss: 0.0,
            best_recon: f32::MAX,
            tokens_done: 0,
            last_dropped: 0,
            tokenizer_loaded: false,
            started: now,
            tok_s: 0.0,
            last_sample: now,
            last_sample_tokens: 0,
            keep,
            last_loss_t: None,
            warmup: 0,
            fw_ms: 0.0,
            bw_ms: 0.0,
            op_ms: 0.0,
        })
    }

    /// Konfigurasi model.
    pub fn model_config(&self) -> &VirTrainConfig {
        &self.model_cfg
    }

    /// Device aktif.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Jumlah gradien beku yang dibuang pada langkah terakhir.
    pub fn last_dropped(&self) -> usize {
        self.last_dropped
    }

    /// Buka logger + direktori checkpoint (dipakai CLI).
    pub fn open(&mut self, dir: &str) -> Result<(), VirError> {
        std::fs::create_dir_all(dir)?;
        self.ckpt_dir = dir.to_string();
        self.logger = Some(JsonlLogger::new(&format!("{dir}/{LOG_FILE}"))?);
        self.log_event("stage_start", "trainer native mulai")
    }

    /// Ganti anggaran langkah.
    pub fn set_budget(&mut self, steps: u64) {
        self.config.steps = steps;
    }

    /// Setel jumlah langkah warmup (W0).
    ///
    /// Warmup dijalankan sekali oleh [`Self::run`]: langkah nyata
    /// (forward/backward/step) untuk memicu kompilasi kernel dan autotune,
    /// tanpa metrik, log, atau `tokens_done`. Jam tok/s dimulai ulang setelah
    /// warmup, sehingga angka yang dilaporkan adalah steady-state.
    pub fn set_warmup(&mut self, warmup: u64) {
        self.warmup = warmup;
    }

    /// Logger aktif.
    pub fn logger_mut(&mut self) -> Option<&mut JsonlLogger> {
        self.logger.as_mut()
    }

    /// `true` bila gerbang VQ sudah lulus (untuk ringkasan CLI).
    pub fn gate_passed(&self) -> bool {
        self.gate.passed
    }

    /// Tok/s rata-rata sejak akhir warmup (lihat [`Self::set_warmup`]).
    pub fn tokens_per_sec(&self) -> f64 {
        let secs = self.started.elapsed().as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        self.tokens_done as f64 / secs
    }

    /// Tok/s jendela terakhir (EMA), diperbarui tiap panggilan.
    fn sample_tok_s(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_sample).as_secs_f64();
        if dt < 0.05 {
            return;
        }
        let dtok = self.tokens_done.saturating_sub(self.last_sample_tokens) as f64;
        let inst = dtok / dt;
        self.tok_s = if self.tok_s <= 0.0 {
            inst
        } else {
            0.8 * self.tok_s + 0.2 * inst
        };
        self.last_sample = now;
        self.last_sample_tokens = self.tokens_done;
    }

    /// Jalankan sampai anggaran langkah habis.
    pub fn run(&mut self) -> Result<(), VirError> {
        self.run_warmup()?;
        while self.step < self.config.steps {
            self.train_step(true)?;
        }
        self.sync_last_loss();
        Ok(())
    }

    /// Jalankan tepat `steps` langkah (abaikan anggaran global).
    pub fn run_steps(&mut self, steps: u64) -> Result<(), VirError> {
        let budget = self.config.steps;
        self.config.steps = self.step + steps;
        let out = self.run();
        self.config.steps = budget;
        out
    }

    /// Langkah warmup (W0): graf sama, tanpa penghitung/log/metrik.
    ///
    /// Berjalan sekali per trainer; sesudahnya `warmup` dinolkan agar `run` yang
    /// dipanggil lagi (mis. `run_steps` bertahap) tidak mengulang warmup.
    fn run_warmup(&mut self) -> Result<(), VirError> {
        let warmup = self.warmup;
        if warmup == 0 {
            return Ok(());
        }
        for _ in 0..warmup {
            self.train_step(false)?;
        }
        self.warmup = 0;
        self.reset_clock();
        self.log_event("warmup", &format!("{warmup} langkah warmup"))
    }

    /// Mulai pengukuran tok/s dari sini (bukan dari konstruksi trainer).
    fn reset_clock(&mut self) {
        let now = Instant::now();
        self.started = now;
        self.last_sample = now;
        self.last_sample_tokens = self.tokens_done;
        self.tok_s = 0.0;
        self.fw_ms = 0.0;
        self.bw_ms = 0.0;
        self.op_ms = 0.0;
    }

    /// Satu langkah penuh: forward → backward → `opt.step`.
    ///
    /// `measured = false` (warmup) menjalankan graf yang sama tetapi tidak
    /// menyentuh penghitung, log, metrik, atau gerbang.
    fn train_step(&mut self, measured: bool) -> Result<(), VirError> {
        let t0 = Instant::now();
        let lr = self.config.lr.lr(self.step);
        let want_metrics = measured && self.metrics_due();
        let out = self.forward(lr, want_metrics)?;
        let fwd_ms = elapsed_ms(t0);
        let tokens = out.tokens;
        let metrics = out.metrics;
        let (bwd_ms, opt_ms) = self.backward_and_step(lr, out.loss)?;
        let dt_ms = elapsed_ms(t0);
        if measured {
            self.record_times(fwd_ms, bwd_ms, opt_ms);
            self.finish_step(tokens, metrics, lr, dt_ms)?;
        }
        Ok(())
    }

    /// Perbarui EWMA durasi tiap fase (fwd/bwd/opt) — untuk R1 (profil).
    fn record_times(&mut self, fwd: f32, bwd: f32, opt: f32) {
        let a = 0.2;
        self.fw_ms = ewma(self.fw_ms, fwd, a);
        self.bw_ms = ewma(self.bw_ms, bwd, a);
        self.op_ms = ewma(self.op_ms, opt, a);
    }

    /// `true` bila langkah berikutnya harus mengukur metrik (log/gerbang).
    fn metrics_due(&self) -> bool {
        self.log_due(self.step + 1) || self.gate_due(self.stage_step + 1)
    }

    /// `true` bila langkah `step` (1-based) menulis baris log.
    fn log_due(&self, step: u64) -> bool {
        self.config.log_every > 0 && step.is_multiple_of(self.config.log_every)
    }

    /// `true` bila gerbang tahap 1 jatuh tempo pada `stage_step` (1-based).
    fn gate_due(&self, stage_step: u64) -> bool {
        match self.config.gate {
            Some(g) => self.stage == Stage::Vq && !self.gate.passed && g.due(stage_step),
            None => false,
        }
    }

    /// Forward satu langkah (device); `want_metrics` meminta bahan metrik.
    ///
    /// Memuat tokenizer ke generator bila belum (tahap 2, sekali saja) — sama
    /// seperti versi sebelumnya; tanpa ini `embed_codes` salah.
    fn forward(&mut self, lr: f64, want_metrics: bool) -> Result<NativeStepOut, VirError> {
        if self.stage == Stage::Generator && !self.tokenizer_loaded {
            let vq = &self.model.vq;
            let generator = &mut self.model.generator;
            generator
                .load_tokenizer(vq)
                .map_err(|e| VirError::Internal(format!("load tokenizer: {e}")))?;
            self.tokenizer_loaded = true;
        }
        let stage = self.stage;
        let step = self.step;
        let stage_step = self.stage_step;
        let rng = &mut self.rng;
        let corruption = self.corruption.as_mut();
        let mut ctx = NativeStepCtx {
            stage,
            step,
            stage_step,
            lr,
            rng,
            corruption,
        };
        native_forward(
            &mut self.model,
            &mut ctx,
            &self.model_cfg,
            &self.weights,
            &mut self.source,
            &self.device,
            want_metrics,
        )
    }

    /// Backward + `opt.step` di device; hanya gradien non-beku yang diikutkan.
    ///
    /// Mengembalikan `(bwd_ms, opt_ms)` untuk profil fase (R1).
    fn backward_and_step(&mut self, lr: f64, loss: Tensor<1>) -> Result<(f32, f32), VirError> {
        if !loss.is_tracked() {
            return Err(VirError::Internal(
                "loss tidak terlacak autodiff (device tanpa gradien?)".into(),
            ));
        }
        self.last_loss_t = Some(loss.clone());
        let tb = Instant::now();
        let mut grads = loss.backward();
        let bwd_ms = elapsed_ms(tb);
        self.refresh_keep();
        let grads = GradientsParams::from_params(&mut grads, &self.model, self.keep_ids());
        let to = Instant::now();
        // ModuleOptimizer::step mengonsumsi model; klon device-side (bukan PCIe).
        let model = self.model.clone();
        self.model = self.opt.step(lr, model, grads);
        let opt_ms = elapsed_ms(to);
        Ok((bwd_ms, opt_ms))
    }

    /// Segarkan cache `ParamId` non-beku bila tahap berubah (B11).
    ///
    /// Bentuk model tetap sepanjang latihan, jadi satu entri `(stage, ids)`
    /// cukup; transisi VQ → Generator adalah satu-satunya perubahan.
    fn refresh_keep(&mut self) {
        let stage = self.stage;
        if self.keep.0 != stage || self.keep.1.is_empty() {
            self.keep = (stage, non_frozen_param_ids(&self.model, stage));
        }
    }

    /// `ParamId` non-beku untuk tahap aktif (lihat [`Self::refresh_keep`]).
    fn keep_ids(&self) -> &Vec<ParamId> {
        &self.keep.1
    }

    /// Selesaikan langkah terukur: penghitung, metrik, checkpoint, gerbang.
    fn finish_step(
        &mut self,
        tokens: usize,
        metrics: Option<RawMetrics>,
        lr: f64,
        dt_ms: f32,
    ) -> Result<(), VirError> {
        self.step += 1;
        self.stage_step += 1;
        self.tokens_done += tokens as u64;
        self.sample_tok_s();
        let recon = match metrics {
            Some(raw) => self.log_step(&raw, lr, dt_ms)?,
            None => None,
        };
        self.maybe_checkpoint()?;
        self.maybe_gate(recon)
    }

    /// Tulis satu baris JSONL; metrik host diekstrak di sini (satu-satunya D2H).
    ///
    /// Mengembalikan nilai rekonstruksi (bila ada) untuk gerbang tahap 1.
    fn log_step(&mut self, raw: &RawMetrics, lr: f64, dt_ms: f32) -> Result<Option<f32>, VirError> {
        let mut out = raw.to_step_out(&self.model);
        if let Some(t) = &self.last_loss_t {
            out.loss_total = t.clone().into_scalar::<f32>();
        }
        if let Some(v) = out.recon {
            self.best_recon = self.track_best(v);
        }
        let rec = StepRecord {
            stage: self.stage.code(),
            step: self.step,
            stage_step: self.stage_step,
            lr: lr as f32,
            loss_total: out.loss_total,
            per_level: out.per_level.clone(),
            level_tokens: out.level_tokens.clone(),
            utilization: out.utilization.clone(),
            perplexity: out.perplexity.clone(),
            grad_norm: 0.0,
            corruption_rate: out.corruption_rate,
            recon: out.recon,
            dt_ms,
        };
        if let Some(log) = self.logger.as_mut() {
            log.log(&rec)?;
        }
        Ok(out.recon)
    }

    /// Simpan checkpoint bila jatuh tempo.
    fn maybe_checkpoint(&mut self) -> Result<(), VirError> {
        let due = self.config.checkpoint_every > 0
            && self.step.is_multiple_of(self.config.checkpoint_every);
        if due {
            self.save()?;
        }
        Ok(())
    }

    /// Evaluasi gerbang tahap 1 bila jatuh tempo; transisi bila lulus.
    fn maybe_gate(&mut self, recon: Option<f32>) -> Result<(), VirError> {
        let gate = self
            .config
            .gate
            .filter(|_| self.stage == Stage::Vq && !self.gate.passed);
        let Some(gate) = gate else { return Ok(()) };
        let Some(v) = recon.filter(|_| gate.due(self.stage_step)) else {
            return Ok(());
        };
        if !self.gate.update(&gate, v, self.step) {
            return Ok(());
        }
        self.log_event(
            "gate_passed",
            &format!(
                "{} = {:.4} (ambang {:.4})",
                gate.metric.name(),
                v,
                gate.threshold
            ),
        )?;
        self.enter_generator()
    }

    /// Sinkron loss tracked terakhir ke host (satu D2H per `run`).
    fn sync_last_loss(&mut self) {
        if let Some(t) = self.last_loss_t.take() {
            self.last_loss = t.into_scalar::<f32>();
        }
    }

    /// Transisi tahap 1 → tahap 2.
    ///
    /// Pada jalur native "bekukan VQ" berarti gradiennya tidak dimasukkan ke
    /// `GradientsParams` lagi; tidak ada state yang perlu dipindah.
    pub fn enter_generator(&mut self) -> Result<(), VirError> {
        self.stage = Stage::Generator;
        self.stage_step = 0;
        self.transitions += 1;
        self.log_event(
            "stage_start",
            "transisi ke generator (gradien VQ dibekukan)",
        )?;
        self.save()
    }

    /// Simpan checkpoint native atomik (`native_state.json` + `model.safetensors`).
    ///
    /// Kedua berkas ditulis ke `*.tmp` lalu `rename` (atomik per berkas).
    pub fn save(&self) -> Result<(), VirError> {
        if self.ckpt_dir.is_empty() {
            return Ok(());
        }
        let dir = self.ckpt_dir.clone();
        let state = NativeCheckpoint {
            version: CHECKPOINT_VERSION.to_string(),
            stage: self.stage,
            step: self.step,
            stage_step: self.stage_step,
            transitions: self.transitions,
            rng: self.rng.clone(),
            config: self.config.clone(),
            gate: self.gate.clone(),
            corruption: self.corruption.clone(),
            model: self.model_cfg.clone(),
            tokens_done: self.tokens_done,
        };
        let text =
            serde_json::to_string_pretty(&state).map_err(|e| VirError::Internal(e.to_string()))?;
        let blob = write_safetensors(&params::export_safetensors(&self.model)?)?;
        atomic_write(&format!("{dir}/{NATIVE_STATE_FILE}"), text.as_bytes())?;
        atomic_write(&format!("{dir}/{MODEL_FILE}"), &blob)?;
        Ok(())
    }

    /// Pulihkan trainer native dari `dir` dengan sumber data eksplisit.
    ///
    /// Model dibangun ulang dari `state.model` lalu bobotnya diisi dari
    /// `model.safetensors`. State optimizer tidak ikut (lihat catatan modul);
    /// pemanggil yang butuh resume identik memakai `crate::trainer::Trainer`.
    pub fn resume(dir: &str, source: ShardBatchSource) -> Result<Self, VirError> {
        let text = std::fs::read_to_string(format!("{dir}/{NATIVE_STATE_FILE}"))?;
        let state: NativeCheckpoint = serde_json::from_str(&text)
            .map_err(|e| VirError::Internal(format!("state native rusak: {e}")))?;
        let blob = std::fs::read(format!("{dir}/{MODEL_FILE}"))?;
        let tensors = read_safetensors(&blob)?;
        let device = train_device(Default::default(), state.model.seed);
        let model = VirTrainModel::new(&state.model, &device)?;
        let model = params::import_safetensors(model, &tensors)?;
        let opt = burn_optimizer_with_paths(
            &BurnHybridSettings::default(),
            crate::params::two_d_param_paths(&model),
        );
        let weights = crate::loss::level_weight(&state.model.levels, state.model.alpha);
        let now = Instant::now();
        let keep = (state.stage, non_frozen_param_ids(&model, state.stage));
        Ok(Self {
            config: state.config,
            model,
            opt,
            model_cfg: state.model,
            source,
            device,
            weights,
            stage: state.stage,
            step: state.step,
            stage_step: state.stage_step,
            rng: state.rng,
            corruption: state.corruption,
            gate: state.gate,
            logger: None,
            ckpt_dir: dir.to_string(),
            transitions: state.transitions,
            last_loss: f32::NAN,
            best_recon: f32::MAX,
            tokens_done: state.tokens_done,
            last_dropped: 0,
            tokenizer_loaded: false,
            started: now,
            tok_s: 0.0,
            last_sample: now,
            last_sample_tokens: state.tokens_done,
            keep,
            last_loss_t: None,
            warmup: 0,
            fw_ms: 0.0,
            bw_ms: 0.0,
            op_ms: 0.0,
        })
    }

    /// Lintas-tahap: nilai terbaik sejauh ini.
    fn track_best(&mut self, v: f32) -> f32 {
        let higher = self
            .config
            .gate
            .map(|g| g.metric.higher_is_better())
            .unwrap_or(true);
        let better = if higher {
            v > self.best_recon
        } else {
            v < self.best_recon
        };
        if better {
            self.best_recon = v;
        }
        self.best_recon
    }

    /// Tulis satu peristiwa JSONL.
    pub fn log_event(&mut self, event: &str, message: &str) -> Result<(), VirError> {
        let rec = EventRecord::new(event, self.stage.code(), self.step, message, None);
        match self.logger.as_mut() {
            Some(log) => log.log(&rec),
            None => Ok(()),
        }
    }

    /// Ringkasan hasil (untuk CLI) tanpa membangun `Trainer`.
    pub fn outcome(&self, gate_passed: bool) -> RunOutcome {
        RunOutcome {
            steps: self.step,
            stage: self.stage,
            gate_passed,
            transitioned: self.transitions > 0,
            loss: self.last_loss,
            tok_s: self.tokens_per_sec(),
            fwd_ms: self.fw_ms,
            bwd_ms: self.bw_ms,
            opt_ms: self.op_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpoint native
// ---------------------------------------------------------------------------

/// Metadata checkpoint native (bobot terpisah di `model.safetensors`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeCheckpoint {
    /// Versi format.
    pub version: String,
    /// Tahap saat checkpoint ditulis.
    pub stage: Stage,
    /// Langkah global.
    pub step: u64,
    /// Langkah sejak masuk tahap.
    pub stage_step: u64,
    /// Jumlah transisi tahap.
    pub transitions: u32,
    /// State RNG.
    pub rng: Rng,
    /// Konfigurasi latihan.
    pub config: TrainConfig,
    /// State gerbang.
    pub gate: GateTracker,
    /// Korupsi (state RNG ikut).
    pub corruption: Option<Corruption>,
    /// Konfigurasi model (bentuk bobot).
    pub model: VirTrainConfig,
    /// Token yang sudah diproses.
    pub tokens_done: u64,
}

/// Tulis `bytes` ke `path` secara atomik (`path.tmp` → rename).
fn atomic_write(path: &str, bytes: &[u8]) -> Result<(), VirError> {
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Durasi berjalan untuk instrumentasi (dipakai bench).
pub fn elapsed_since(start: Instant) -> Duration {
    start.elapsed()
}

/// Milidetik sejak `t0` (untuk `StepRecord::dt_ms`).
fn elapsed_ms(t0: Instant) -> f32 {
    t0.elapsed().as_secs_f32() * 1000.0
}

/// Exponentially-weighted moving average (profil fase, R1).
fn ewma(cur: f32, inst: f32, a: f32) -> f32 {
    if cur <= 0.0 {
        inst
    } else {
        (1.0 - a) * cur + a * inst
    }
}
