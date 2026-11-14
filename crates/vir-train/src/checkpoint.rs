//! Checkpoint atomik dan resume identik (PRD F1.7 task "checkpoint atomik dan
//! tes resume identik", langkah 7).
//!
//! Struktur satu checkpoint = satu direktori:
//!
//! | Berkas | Isi |
//! |---|---|
//! | `state.json` | metadata: versi, tahap, langkah, RNG, config, gerbang, bentuk parameter, opt |
//! | `tensors.bin` | nilai parameter + state optimizer, little-endian |
//!
//! Penulisan **atomik**: payload ditulis ke `{dir}.tmp`, lalu `rename` ke
//! `{dir}` (POSIX: atomik dalam satu filesystem). Direktori lama dipindah ke
//! `{dir}.old` lebih dulu agar `rename` direktori tidak gagal karena direktori
//! tujuan tidak kosong; bila `rename` gagal, `.old` dipulihkan.
//!
//! Resume identik: nilai parameter disimpan sebagai bit `f32` apa adanya
//! (`to_le_bytes`), RNG disimpan sebagai state 4×u64, dan seluruh state
//! optimizer ikut, sehingga langkah berikutnya sama bit demi bit.

use crate::corruption::Corruption;
use crate::logging::JsonlLogger;
use crate::optim::{
    AdamState, Hybrid, MomState, OptGroup, OptimSettings, ParamRole, ParamSlot, ParamStore,
};
use crate::rng::Rng;
use crate::trainer::{GateMetric, GateTracker, TrainConfig};
use serde::{Deserialize, Serialize};
use vir_common::VirError;

/// Versi format checkpoint (berubah bila layout tidak kompatibel).
pub const CHECKPOINT_VERSION: &str = "vir-train-1";
/// Nama berkas metadata di dalam direktori checkpoint.
pub const STATE_FILE: &str = "state.json";
/// Nama berkas payload tensor di dalam direktori checkpoint.
pub const BLOB_FILE: &str = "tensors.bin";
/// Nama berkas log JSONL di dalam direktori checkpoint.
pub const LOG_FILE: &str = "train.jsonl";

/// Metadata satu parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamMeta {
    /// Nama parameter.
    pub name: String,
    /// Baris.
    pub rows: usize,
    /// Kolom.
    pub cols: usize,
    /// Peran tahap.
    pub role: ParamRole,
    /// Grup optimizer.
    pub group: OptGroup,
    /// Status beku.
    pub frozen: bool,
}

/// Metadata optimizer (cukup untuk membangun ulang state).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptMeta {
    /// Parameter Muon.
    pub muon: crate::optim::MuonParams,
    /// Parameter AdamW.
    pub adamw: crate::optim::AdamParams,
    /// Weight decay.
    pub weight_decay: f32,
    /// Skala LR Muon.
    pub muon_lr_scale: f32,
}

/// Metadata checkpoint (tanpa array besar).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Versi format.
    pub version: String,
    /// Tahap saat checkpoint ditulis.
    pub stage: u32,
    /// Langkah global.
    pub step: u64,
    /// Langkah sejak masuk tahap.
    pub stage_step: u64,
    /// Jumlah transisi tahap yang sudah terjadi.
    pub transitions: u32,
    /// State RNG.
    pub rng: Rng,
    /// Konfigurasi latihan.
    pub config: TrainConfig,
    /// State gerbang rekonstruksi.
    pub gate: GateTracker,
    /// Korupsi (termasuk state RNG-nya) — `None` bila tidak memakai korupsi.
    pub corruption: Option<Corruption>,
    /// Metadata parameter berurutan.
    pub params: Vec<ParamMeta>,
    /// Metadata optimizer.
    pub opt: OptMeta,
    /// Panjang payload tensor (validasi saat load).
    pub blob_len: usize,
}

impl Checkpoint {
    /// JSON metadata (dipakai untuk diff/audit).
    pub fn to_json(&self) -> Result<String, VirError> {
        serde_json::to_string_pretty(self).map_err(|e| VirError::Internal(e.to_string()))
    }

    /// Metadata dari JSON.
    pub fn from_json(s: &str) -> Result<Self, VirError> {
        serde_json::from_str(s).map_err(|e| VirError::Internal(e.to_string()))
    }
}

/// Pasangan metadata + payload tensor.
#[derive(Debug, Clone)]
pub struct CheckpointFile {
    /// Metadata.
    pub checkpoint: Checkpoint,
    /// Payload little-endian: parameter, state Muon, state AdamW.
    pub blob: Vec<u8>,
}

impl CheckpointFile {
    /// Susun dari metadata dan payload.
    pub fn new(checkpoint: Checkpoint, blob: Vec<u8>) -> Self {
        Self { checkpoint, blob }
    }

    /// Tulis checkpoint secara atomik ke `dir`.
    pub fn save_atomic(&self, dir: &str) -> Result<(), VirError> {
        let tmp = format!("{dir}.tmp");
        let old = format!("{dir}.old");
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::remove_dir_all(&old);
        std::fs::create_dir_all(&tmp)?;
        std::fs::write(format!("{tmp}/{STATE_FILE}"), self.checkpoint.to_json()?)?;
        std::fs::write(format!("{tmp}/{BLOB_FILE}"), &self.blob)?;
        if std::path::Path::new(dir).exists() {
            std::fs::rename(dir, &old)?;
        }
        if let Err(e) = std::fs::rename(&tmp, dir) {
            if std::path::Path::new(&old).exists() {
                let _ = std::fs::rename(&old, dir);
            }
            return Err(VirError::Io(e));
        }
        let _ = std::fs::remove_dir_all(&old);
        Ok(())
    }

    /// Baca checkpoint dari `dir`.
    pub fn load(dir: &str) -> Result<Self, VirError> {
        let meta = std::fs::read_to_string(format!("{dir}/{STATE_FILE}"))?;
        let checkpoint = Checkpoint::from_json(&meta)?;
        let blob = std::fs::read(format!("{dir}/{BLOB_FILE}"))?;
        if blob.len() != checkpoint.blob_len {
            return Err(VirError::Internal(format!(
                "panjang payload {} != metadata {}",
                blob.len(),
                checkpoint.blob_len
            )));
        }
        if checkpoint.version != CHECKPOINT_VERSION {
            return Err(VirError::Config(format!(
                "versi checkpoint {} tidak didukung (butuh {CHECKPOINT_VERSION})",
                checkpoint.version
            )));
        }
        Ok(Self { checkpoint, blob })
    }

    /// Pulihkan `ParamStore` dan optimizer dari checkpoint.
    pub fn restore(&self) -> Result<(ParamStore, Hybrid), VirError> {
        let meta = &self.checkpoint;
        let buf = &self.blob[..];
        let mut pos = 0usize;
        let mut slots = Vec::with_capacity(meta.params.len());
        for p in &meta.params {
            let n = p.rows.saturating_mul(p.cols).max(1);
            let data = take_f32s(buf, &mut pos, n)?;
            let mut slot = ParamSlot::new(&p.name, p.rows, p.cols, p.role, data);
            slot.group = p.group;
            slot.frozen = p.frozen;
            slots.push(slot);
        }
        let store = ParamStore { slots };
        let mut muon_states = Vec::with_capacity(meta.params.len());
        for p in &meta.params {
            let tag = take_u8(buf, &mut pos)?;
            let len = p.rows.saturating_mul(p.cols).max(1);
            muon_states.push(MomState::decode(tag, len, buf, &mut pos)?);
        }
        let mut adam_states = Vec::with_capacity(meta.params.len());
        for p in &meta.params {
            let n = p.rows.saturating_mul(p.cols).max(1);
            let m = take_f32s(buf, &mut pos, n)?;
            let v = take_f32s(buf, &mut pos, n)?;
            let t = take_u64(buf, &mut pos)?;
            adam_states.push(AdamState { m, v, t });
        }
        if pos != buf.len() {
            return Err(VirError::Internal(format!(
                "payload checkpoint tidak terpakai penuh: {pos} dari {}",
                buf.len()
            )));
        }
        let settings = OptimSettings {
            muon: meta.opt.muon,
            adamw: meta.opt.adamw,
            weight_decay: meta.opt.weight_decay,
            muon_lr_scale: meta.opt.muon_lr_scale,
        };
        let hybrid = Hybrid::from_states(&store, &settings, muon_states, adam_states);
        Ok((store, hybrid))
    }

    /// Encoder parameter + state optimizer menjadi payload.
    ///
    /// State ditulis **per slot, berurutan** (termasuk state nol untuk slot
    /// dengan grup lain) supaya indeks state cocok dengan indeks slot setelah
    /// `restore`, sama seperti `Hybrid::new` yang membuat state untuk semua
    /// slot.
    pub fn encode(store: &ParamStore, opt: &Hybrid) -> Vec<u8> {
        let mut enc = Enc { b: Vec::new() };
        for s in &store.slots {
            for v in &s.data {
                enc.f32(*v);
            }
        }
        for i in 0..store.slots.len() {
            match opt.muon.states.get(i) {
                Some(st) => st.encode(&mut enc.b),
                None => MomState::zeros(store.slots[i].len(), opt.muon.params.state_bits)
                    .encode(&mut enc.b),
            }
        }
        for i in 0..store.slots.len() {
            let st = opt.adamw.states.get(i);
            let len = store.slots[i].len();
            let (m, v, t) = match st {
                Some(st) => (st.m.clone(), st.v.clone(), st.t),
                None => (vec![0.0; len], vec![0.0; len], 0),
            };
            for x in m {
                enc.f32(x);
            }
            for x in v {
                enc.f32(x);
            }
            enc.u64(t);
        }
        enc.b
    }
}

/// Posisi latihan yang ikut ke checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Progress {
    /// Tahap saat checkpoint ditulis.
    pub stage: u32,
    /// Langkah global.
    pub step: u64,
    /// Langkah sejak masuk tahap.
    pub stage_step: u64,
    /// Jumlah transisi tahap.
    pub transitions: u32,
}

/// Pembuat checkpoint dari state trainer.
pub fn build(
    config: &TrainConfig,
    progress: &Progress,
    rng: &Rng,
    gate: &GateTracker,
    corruption: Option<&Corruption>,
    store: &ParamStore,
    opt: &Hybrid,
) -> CheckpointFile {
    let params = store
        .slots
        .iter()
        .map(|s| ParamMeta {
            name: s.name.clone(),
            rows: s.rows,
            cols: s.cols,
            role: s.role,
            group: s.group,
            frozen: s.frozen,
        })
        .collect();
    let opt_meta = OptMeta {
        muon: opt.muon.params,
        adamw: opt.adamw.params,
        weight_decay: opt.settings.weight_decay,
        muon_lr_scale: opt.settings.muon_lr_scale,
    };
    let blob = CheckpointFile::encode(store, opt);
    let checkpoint = Checkpoint {
        version: CHECKPOINT_VERSION.to_string(),
        stage: progress.stage,
        step: progress.step,
        stage_step: progress.stage_step,
        transitions: progress.transitions,
        rng: rng.clone(),
        config: config.clone(),
        gate: gate.clone(),
        corruption: corruption.cloned(),
        params,
        opt: opt_meta,
        blob_len: blob.len(),
    };
    CheckpointFile::new(checkpoint, blob)
}

/// Buka logger JSONL pada direktori checkpoint (menimpa berkas lama).
pub fn open_logger(dir: &str) -> Result<JsonlLogger, VirError> {
    std::fs::create_dir_all(dir)?;
    JsonlLogger::in_dir(dir, LOG_FILE)
}

/// Encoder byte little-endian.
struct Enc {
    b: Vec<u8>,
}

impl Enc {
    fn f32(&mut self, v: f32) {
        self.b.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.b.extend_from_slice(&v.to_le_bytes());
    }
}

/// Ambil `n` byte pada `*pos` (gagal bila payload terpotong).
fn take(buf: &[u8], pos: &mut usize, n: usize) -> Result<(), VirError> {
    if *pos + n > buf.len() {
        return Err(VirError::Internal("payload checkpoint terpotong".into()));
    }
    Ok(())
}

/// Baca satu `f32` little-endian.
fn take_f32(buf: &[u8], pos: &mut usize) -> Result<f32, VirError> {
    take(buf, pos, 4)?;
    let b = [buf[*pos], buf[*pos + 1], buf[*pos + 2], buf[*pos + 3]];
    *pos += 4;
    Ok(f32::from_le_bytes(b))
}

/// Baca `n` buah `f32`.
fn take_f32s(buf: &[u8], pos: &mut usize, n: usize) -> Result<Vec<f32>, VirError> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(take_f32(buf, pos)?);
    }
    Ok(out)
}

/// Baca satu `u8`.
fn take_u8(buf: &[u8], pos: &mut usize) -> Result<u8, VirError> {
    take(buf, pos, 1)?;
    let v = buf[*pos];
    *pos += 1;
    Ok(v)
}

/// Baca satu `u64` little-endian.
fn take_u64(buf: &[u8], pos: &mut usize) -> Result<u64, VirError> {
    take(buf, pos, 8)?;
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[*pos..*pos + 8]);
    *pos += 8;
    Ok(u64::from_le_bytes(b))
}

/// Uji asap checkpoint: simpan, muat, bandingkan payload.
pub fn checkpoint_smoke() -> Result<(), VirError> {
    use crate::optim::{OptimSettings, ParamSlot};
    let path = std::env::temp_dir().join("vir_train_ckpt_smoke");
    let dir = path
        .to_str()
        .ok_or_else(|| VirError::Internal("path checkpoint tidak valid".into()))?;
    let _ = std::fs::remove_dir_all(dir);
    let mut store = ParamStore::new();
    store.add(ParamSlot::new(
        "core.q.weight",
        2,
        2,
        ParamRole::Generator,
        vec![0.5, -0.25, 1.0, 0.0],
    ));
    store.add(ParamSlot::vector(
        "norm.weight",
        ParamRole::Generator,
        vec![1.0, 2.0],
    ));
    let opt = Hybrid::new(&store, &OptimSettings::default());
    let config = TrainConfig::default();
    let gate = GateTracker::new(GateMetric::TokenAccuracy);
    let progress = Progress {
        stage: 1,
        step: 7,
        stage_step: 7,
        transitions: 0,
    };
    let file = build(
        &config,
        &progress,
        &Rng::new(1),
        &gate,
        None,
        &store,
        &opt,
    );
    file.save_atomic(dir)?;
    let loaded = CheckpointFile::load(dir)?;
    let (store2, _opt2) = loaded.restore()?;
    for (a, b) in store.slots.iter().zip(store2.slots.iter()) {
        if a.data != b.data {
            return Err(VirError::Internal("parameter tidak pulih identik".into()));
        }
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}