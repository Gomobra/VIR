//! Trainer dua tahap dengan gerbang metrik rekonstruksi (PRD F1.7).
//!
//! Tahap 1 melatih VQ-VAE sampai gerbang rekonstruksi lulus; tahap 2 membekukan
//! VQ lalu melatih generator. Satu `run` menjalankan kedua tahap: ketika gerbang
//! lulus, parameter VQ dibekukan dan tahap berganti ke generator tanpa
//! menghentikan loop (langkah global dan checkpoint tetap berlanjut).
//!
//! Batas tugas:
//!
//! ```text
//! for step in 0..steps:
//!     lr = jadwal(step)
//!     out = step_fn(ctx{stage, step, lr, rng, corruption}, store)   // forward+backward
//!     gnorm = clip_grad_norm(store, max_norm)
//!     opt.step(store, lr_muon, lr_adamw)     // Muon/AdamW hibrid, lewati yang beku
//!     store.clear_grads()
//!     log JSONL tiap log_every langkah
//!     checkpoint atomik tiap checkpoint_every langkah
//!     gate rekonstruksi (tahap 1) → transisi ke tahap 2
//! ```
//!
//! `StepFn` adalah satu-satunya tempat model menyentuh tensor: ia menerima
//! konteks (tahap, langkah, laju, RNG, korupsi) dan `ParamStore` (bobot +
//! gradien), lalu mengembalikan metrik. Dengan begitu trainer, optimizer,
//! jadwal, checkpoint, dan log bisa diuji tanpa membangun model Burn.

use crate::checkpoint::{CheckpointFile, LOG_FILE, Progress, build};
use crate::corruption::Corruption;
use crate::logging::{EventRecord, JsonlLogger, StepRecord};
use crate::optim::{Hybrid, OptimSettings, ParamStore, clip_grad_norm};
use crate::rng::Rng;
use crate::schedule::LrSchedule;
use serde::{Deserialize, Serialize};
use vir_common::VirError;

/// Tahap latihan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stage {
    /// Tahap 1: VQ-VAE (tokenizer), sampai gerbang rekonstruksi lulus.
    Vq,
    /// Tahap 2: generator NCM dengan tokenizer beku.
    Generator,
}

impl Stage {
    /// Kode numerik untuk log dan checkpoint (`1` = VQ, `2` = generator).
    pub fn code(self) -> u32 {
        match self {
            Self::Vq => 1,
            Self::Generator => 2,
        }
    }

    /// Tahap dari kode numerik.
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            1 => Some(Self::Vq),
            2 => Some(Self::Generator),
            _ => None,
        }
    }

    /// Nama singkat.
    pub fn name(self) -> &'static str {
        match self {
            Self::Vq => "vq",
            Self::Generator => "generator",
        }
    }
}

/// Metrik yang dipakai gerbang rekonstruksi.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateMetric {
    /// Akurasi token eksak (0..1), lebih tinggi lebih baik.
    TokenAccuracy,
    /// Akurasi eksak per utas (0..1), lebih tinggi lebih baik.
    ExactMatch,
    /// Loss rekonstruksi, lebih rendah lebih baik.
    ReconLoss,
    /// Utilisasi codebook (0..1), lebih tinggi lebih baik.
    Utilization,
}

impl GateMetric {
    /// `true` bila nilai besar berarti lolos.
    pub fn higher_is_better(self) -> bool {
        matches!(
            self,
            Self::TokenAccuracy | Self::ExactMatch | Self::Utilization
        )
    }

    /// Nama metrik untuk log.
    pub fn name(self) -> &'static str {
        match self {
            Self::TokenAccuracy => "token_accuracy",
            Self::ExactMatch => "exact_match",
            Self::ReconLoss => "recon_loss",
            Self::Utilization => "utilization",
        }
    }
}

/// Gerbang promosi tahap 1 → tahap 2.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReconGate {
    /// Metrik yang diukur.
    pub metric: GateMetric,
    /// Ambang: akurasi/utilisasi `>=`, loss `<=`.
    pub threshold: f32,
    /// Jumlah evaluasi lulus berturut-turut sebelum transisi (default 3).
    pub patience: u64,
    /// Abaikan evaluasi sebelum langkah ini (tahap 1).
    pub min_stage_step: u64,
    /// Evaluasi tiap N langkah (0 = setiap langkah).
    pub every: u64,
}

impl ReconGate {
    /// Gerbang akurasi (default NCM: akurasi ≥ 0,5).
    pub fn accuracy(threshold: f32, patience: u64) -> Self {
        Self {
            metric: GateMetric::TokenAccuracy,
            threshold,
            patience,
            min_stage_step: 0,
            every: 0,
        }
    }

    /// Gerbang loss rekonstruksi.
    pub fn loss(threshold: f32, patience: u64) -> Self {
        Self {
            metric: GateMetric::ReconLoss,
            threshold,
            patience,
            min_stage_step: 0,
            every: 0,
        }
    }

    /// Nilai memenuhi ambang.
    pub fn satisfied(&self, v: f32) -> bool {
        if !v.is_finite() {
            return false;
        }
        if self.metric.higher_is_better() {
            v >= self.threshold
        } else {
            v <= self.threshold
        }
    }

    /// Saatnya evaluasi pada langkah tahap tertentu.
    pub fn due(&self, stage_step: u64) -> bool {
        stage_step >= self.min_stage_step
            && (self.every == 0 || stage_step.is_multiple_of(self.every))
    }
}

/// State gerbang (lintas checkpoint).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateTracker {
    /// Nilai terbaik sejauh ini (arah sesuai metrik).
    pub best: f32,
    /// Evaluasi lulus berturut-turut.
    pub run: u64,
    /// Total evaluasi.
    pub evaluations: u64,
    /// `true` bila gerbang sudah lulus.
    pub passed: bool,
    /// Langkah saat lulus.
    pub passed_at: Option<u64>,
}

impl GateTracker {
    /// Tracker baru dengan arah sesuai metrik.
    pub fn new(metric: GateMetric) -> Self {
        Self {
            best: if metric.higher_is_better() {
                f32::MIN
            } else {
                f32::MAX
            },
            run: 0,
            evaluations: 0,
            passed: false,
            passed_at: None,
        }
    }

    /// Catat satu evaluasi; `true` bila gerbang baru saja lulus.
    pub fn update(&mut self, gate: &ReconGate, value: f32, step: u64) -> bool {
        if !value.is_finite() {
            return false;
        }
        self.evaluations += 1;
        let better = if gate.metric.higher_is_better() {
            value > self.best
        } else {
            value < self.best
        };
        if better {
            self.best = value;
        }
        if gate.satisfied(value) {
            self.run += 1;
        } else {
            self.run = 0;
        }
        if !self.passed && self.run >= gate.patience.max(1) {
            self.passed = true;
            self.passed_at = Some(step);
            return true;
        }
        false
    }

    /// Nilai terbaik sejauh ini.
    pub fn best(&self) -> f32 {
        self.best
    }
}

/// Konfigurasi latihan (tersimpan di checkpoint agar resume memakai config sama).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Seed RNG.
    pub seed: u64,
    /// Tahap awal.
    pub stage: Stage,
    /// Anggaran langkah global (tahap 1 + tahap 2).
    pub steps: u64,
    /// Panjang level `ℓ_k` (urut naik).
    pub levels: Vec<usize>,
    /// Eksponen bobot level (0 = per kode; 1 = per level).
    pub alpha: f32,
    /// Jadwal laju pembelajaran.
    pub lr: LrSchedule,
    /// Clip norma gradien global (0 = tanpa clip).
    pub grad_clip: f32,
    /// Tulis log JSONL tiap N langkah (0 = tidak).
    pub log_every: u64,
    /// Checkpoint atomik tiap N langkah (0 = tidak).
    pub checkpoint_every: u64,
    /// Gerbang rekonstruksi tahap 1 (`None` = tanpa transisi otomatis).
    pub gate: Option<ReconGate>,
    /// Korupsi kode masukan (`None` = tanpa korupsi).
    pub corruption: Option<Corruption>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            stage: Stage::Vq,
            steps: 1_000,
            levels: vir_common::ModelConfig::default().levels(),
            alpha: 0.0,
            lr: LrSchedule::cosine(3e-4, 3e-5, 500, 1_000),
            grad_clip: 1.0,
            log_every: 10,
            checkpoint_every: 500,
            gate: Some(ReconGate::accuracy(0.5, 3)),
            corruption: None,
        }
    }
}

impl TrainConfig {
    /// Validasi konfigurasi.
    pub fn validate(&self) -> Result<(), VirError> {
        if self.steps == 0 {
            return Err(VirError::Config("steps harus > 0".into()));
        }
        if self.levels.is_empty() {
            return Err(VirError::Config("levels tidak boleh kosong".into()));
        }
        if self.levels.contains(&0) {
            return Err(VirError::Config("panjang level harus > 0".into()));
        }
        if !self.levels.windows(2).all(|w| w[0] < w[1]) {
            return Err(VirError::Config("levels harus naik ketat".into()));
        }
        if !(0.0..=2.0).contains(&self.alpha) {
            return Err(VirError::Config("alpha harus di [0,2]".into()));
        }
        if self.grad_clip < 0.0 || !self.grad_clip.is_finite() {
            return Err(VirError::Config("grad_clip harus >= 0 dan hingga".into()));
        }
        self.lr.validate()?;
        Ok(())
    }

    /// Jumlah level.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Bobot level menurut `alpha`.
    pub fn weights(&self) -> Vec<f32> {
        crate::loss::level_weight(&self.levels, self.alpha)
    }
}

/// Metrik satu langkah (host; tensor sudah dikonversi oleh `StepFn`).
#[derive(Debug, Clone, Default)]
pub struct StepOut {
    /// Loss total.
    pub loss_total: f32,
    /// Loss per level (sudah berbobot).
    pub per_level: Vec<f32>,
    /// Token valid per level.
    pub level_tokens: Vec<f32>,
    /// Utilisasi codebook per level.
    pub utilization: Vec<f32>,
    /// Perplexity codebook per level.
    pub perplexity: Vec<f32>,
    /// Nilai metrik rekonstruksi untuk gerbang (bila dihitung).
    pub recon: Option<f32>,
    /// Laju korupsi sebenarnya (bila dipakai).
    pub corruption_rate: Option<f32>,
}

/// Konteks satu langkah yang diberikan ke `StepFn`.
pub struct StepCtx<'a> {
    /// Tahap saat ini.
    pub stage: Stage,
    /// Langkah global.
    pub step: u64,
    /// Langkah sejak masuk tahap.
    pub stage_step: u64,
    /// Laju pembelajaran (AdamW).
    pub lr: f32,
    /// RNG deterministik (ikut checkpoint).
    pub rng: &'a mut Rng,
    /// Korupsi aktif (ikut checkpoint), bila dikonfigurasi.
    pub corruption: Option<&'a mut Corruption>,
}

/// Satu langkah: forward + backward, isi gradien di `store`, kembalikan metrik.
pub trait StepFn {
    /// Hitung satu langkah.
    fn compute(&mut self, ctx: StepCtx<'_>, store: &mut ParamStore) -> Result<StepOut, VirError>;
}

impl<F> StepFn for F
where
    F: FnMut(StepCtx<'_>, &mut ParamStore) -> Result<StepOut, VirError>,
{
    fn compute(&mut self, ctx: StepCtx<'_>, store: &mut ParamStore) -> Result<StepOut, VirError> {
        self(ctx, store)
    }
}

/// Hasil satu panggilan [`Trainer::run`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunOutcome {
    /// Langkah yang dijalankan pada panggilan ini.
    pub steps: u64,
    /// Tahap saat selesai.
    pub stage: Stage,
    /// `true` bila gerbang lulus pada panggilan ini.
    pub gate_passed: bool,
    /// `true` bila terjadi transisi tahap.
    pub transitioned: bool,
    /// Loss total langkah terakhir.
    pub loss: f32,
    /// Throughput steady-state (token/detik) sejak akhir warmup.
    ///
    /// Diisi jalur GPU-native ([`crate::native::NativeTrainer`]); jalur host
    /// mengisi `0.0`. `0.0` juga berarti "tak diukur".
    pub tok_s: f64,
    /// EWMA durasi forward (ms) — profil fase (R1).
    pub fwd_ms: f32,
    /// EWMA durasi backward (ms).
    pub bwd_ms: f32,
    /// EWMA durasi optimizer step (ms).
    pub opt_ms: f32,
}

/// Trainer dua tahap.
#[derive(Debug)]
pub struct Trainer {
    /// Konfigurasi.
    pub config: TrainConfig,
    /// Tahap saat ini.
    pub stage: Stage,
    /// Langkah global.
    pub step: u64,
    /// Langkah sejak masuk tahap.
    pub stage_step: u64,
    /// Parameter + gradien.
    pub store: ParamStore,
    /// Optimizer hibrid.
    pub opt: Hybrid,
    /// RNG (state ikut checkpoint).
    pub rng: Rng,
    /// Korupsi aktif.
    pub corruption: Option<Corruption>,
    /// State gerbang.
    pub gate: GateTracker,
    /// Logger JSONL.
    pub logger: Option<JsonlLogger>,
    /// Direktori checkpoint.
    pub ckpt_dir: String,
    /// Jumlah transisi tahap.
    pub transitions: u32,
    /// Loss total langkah terakhir.
    pub last_loss: f32,
    /// Metrik rekonstruksi terbaik (gerbang).
    pub best_recon: f32,
}

impl Trainer {
    /// Trainer baru dari konfigurasi; log JSONL dibuka (menimpa) di `dir`.
    pub fn new(config: TrainConfig, store: ParamStore, dir: &str) -> Result<Self, VirError> {
        config.validate()?;
        let opt = Hybrid::new(&store, &OptimSettings::default());
        let rng = Rng::new(config.seed);
        let corruption = config.corruption.clone();
        std::fs::create_dir_all(dir)?;
        let logger = JsonlLogger::new(&format!("{dir}/{LOG_FILE}"))?;
        let gate = match config.gate {
            Some(g) => GateTracker::new(g.metric),
            None => GateTracker::new(GateMetric::TokenAccuracy),
        };
        let mut t = Self {
            config,
            stage: Stage::Vq,
            step: 0,
            stage_step: 0,
            store,
            opt,
            rng,
            corruption,
            gate,
            logger: Some(logger),
            ckpt_dir: dir.to_string(),
            transitions: 0,
            last_loss: 0.0,
            best_recon: f32::MAX,
        };
        t.stage = t.config.stage;
        t.log_event("stage_start", t.stage.code(), 0, "trainer mulai")?;
        Ok(t)
    }

    /// Trainer dari checkpoint (resume identik); log JSONL di-*append*.
    pub fn resume(dir: &str) -> Result<Self, VirError> {
        let file = CheckpointFile::load(dir)?;
        let meta = file.checkpoint.clone();
        let (store, opt) = file.restore()?;
        let stage = Stage::from_code(meta.stage).unwrap_or(Stage::Vq);
        std::fs::create_dir_all(dir)?;
        let logger = JsonlLogger::append(&format!("{dir}/{LOG_FILE}"))?;
        let corruption = meta.corruption.clone();
        let mut t = Self {
            config: meta.config.clone(),
            stage,
            step: meta.step,
            stage_step: meta.stage_step,
            store,
            opt,
            rng: meta.rng.clone(),
            corruption,
            gate: meta.gate.clone(),
            logger: Some(logger),
            ckpt_dir: dir.to_string(),
            transitions: meta.transitions,
            last_loss: 0.0,
            best_recon: meta.gate.best,
        };
        t.log_event(
            "resume",
            t.stage.code(),
            t.step,
            &format!("resume dari langkah {}", t.step),
        )?;
        Ok(t)
    }

    /// Ganti anggaran langkah (mis. lanjut sesi Kaggle berikutnya).
    pub fn set_budget(&mut self, steps: u64) {
        self.config.steps = steps;
    }

    /// Logger aktif.
    pub fn logger_mut(&mut self) -> Option<&mut JsonlLogger> {
        self.logger.as_mut()
    }

    /// Jalankan sampai anggaran langkah habis.
    pub fn run(&mut self, step_fn: &mut impl StepFn) -> Result<RunOutcome, VirError> {
        let mut steps = 0u64;
        let mut gate_passed = false;
        let mut transitioned = false;
        while self.step < self.config.steps {
            let lr = self.config.lr.lr_f32(self.step);
            let stage = self.stage;
            let step = self.step;
            let stage_step = self.stage_step;
            let rng = &mut self.rng;
            let corruption = self.corruption.as_mut();
            let ctx = StepCtx {
                stage,
                step,
                stage_step,
                lr,
                rng,
                corruption,
            };
            let out = step_fn.compute(ctx, &mut self.store)?;
            let gnorm = clip_grad_norm(&mut self.store, self.config.grad_clip);
            if self.store.has_grads() {
                self.opt
                    .step_schedule(&mut self.store, &self.config.lr, self.step);
            }
            self.store.clear_grads();
            self.step += 1;
            self.stage_step += 1;
            steps += 1;
            self.last_loss = out.loss_total;
            if let Some(v) = out.recon {
                self.best_recon = self.track_best(v);
            }
            if self.config.log_every > 0 && self.step.is_multiple_of(self.config.log_every) {
                self.log_step(&out, lr, gnorm)?;
            }
            if self.config.checkpoint_every > 0
                && self.step.is_multiple_of(self.config.checkpoint_every)
            {
                self.save()?;
            }
            let gate_cfg = self
                .config
                .gate
                .filter(|_| self.stage == Stage::Vq && !self.gate.passed);
            if let Some(gate) = gate_cfg
                && let Some(v) = out.recon.filter(|_| gate.due(self.stage_step))
                && self.gate.update(&gate, v, self.step)
            {
                gate_passed = true;
                self.log_event_value(
                    "gate_passed",
                    Stage::Vq.code(),
                    self.step,
                    &format!(
                        "{} = {:.4} (ambang {:.4})",
                        gate.metric.name(),
                        v,
                        gate.threshold
                    ),
                    Some(v),
                )?;
                self.enter_generator()?;
                transitioned = true;
            }
        }
        Ok(RunOutcome {
            steps,
            stage: self.stage,
            gate_passed,
            transitioned,
            loss: self.last_loss,
            tok_s: 0.0,
            fwd_ms: 0.0,
            bwd_ms: 0.0,
            opt_ms: 0.0,
        })
    }

    /// Jalankan tepat `steps` langkah (abaikan anggaran global).
    pub fn run_steps(
        &mut self,
        steps: u64,
        step_fn: &mut impl StepFn,
    ) -> Result<RunOutcome, VirError> {
        let budget = self.config.steps;
        self.config.steps = self.step + steps;
        let out = self.run(step_fn);
        self.config.steps = budget;
        out
    }

    /// Transisi tahap 1 → tahap 2: bekukan VQ, reset langkah tahap.
    pub fn enter_generator(&mut self) -> Result<(), VirError> {
        let frozen = self.store.freeze_role(crate::optim::ParamRole::Vq);
        self.stage = Stage::Generator;
        self.stage_step = 0;
        self.transitions += 1;
        self.log_event(
            "stage_start",
            Stage::Generator.code(),
            self.step,
            &format!("transisi ke generator ({frozen} parameter VQ dibekukan)"),
        )?;
        self.save()?;
        Ok(())
    }

    /// Simpan checkpoint atomik.
    pub fn save(&self) -> Result<(), VirError> {
        let progress = Progress {
            stage: self.stage.code(),
            step: self.step,
            stage_step: self.stage_step,
            transitions: self.transitions,
        };
        let file = build(
            &self.config,
            &progress,
            &self.rng,
            &self.gate,
            self.corruption.as_ref(),
            &self.store,
            &self.opt,
        );
        file.save_atomic(&self.ckpt_dir)
    }

    /// Muat ulang checkpoint terakhir (memanggil [`Self::resume`]).
    pub fn reload(&mut self) -> Result<(), VirError> {
        let restored = Self::resume(&self.ckpt_dir)?;
        self.stage = restored.stage;
        self.step = restored.step;
        self.stage_step = restored.stage_step;
        self.store = restored.store;
        self.opt = restored.opt;
        self.rng = restored.rng;
        self.corruption = restored.corruption;
        self.gate = restored.gate;
        self.transitions = restored.transitions;
        Ok(())
    }

    /// Lintas-tahap: nilai terbaik sejauh ini (arah sesuai metrik).
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

    /// Tulis satu baris JSONL per langkah.
    fn log_step(&mut self, out: &StepOut, lr: f32, gnorm: f32) -> Result<(), VirError> {
        let rec = StepRecord {
            stage: self.stage.code(),
            step: self.step,
            stage_step: self.stage_step,
            lr,
            loss_total: out.loss_total,
            per_level: out.per_level.clone(),
            level_tokens: out.level_tokens.clone(),
            utilization: out.utilization.clone(),
            perplexity: out.perplexity.clone(),
            grad_norm: gnorm,
            corruption_rate: out.corruption_rate,
            // Jalur host tidak mengukur durasi per langkah (W0 = jalur native).
            dt_ms: 0.0,
            recon: out.recon,
        };
        match self.logger.as_mut() {
            Some(log) => log.log(&rec)?,
            None => return Ok(()),
        }
        Ok(())
    }

    /// Tulis satu peristiwa JSONL.
    pub fn log_event(
        &mut self,
        event: &str,
        stage: u32,
        step: u64,
        message: &str,
    ) -> Result<(), VirError> {
        self.log_event_value(event, stage, step, message, None)
    }

    /// Tulis satu peristiwa JSONL dengan nilai metrik.
    pub fn log_event_value(
        &mut self,
        event: &str,
        stage: u32,
        step: u64,
        message: &str,
        value: Option<f32>,
    ) -> Result<(), VirError> {
        let rec = EventRecord::new(event, stage, step, message, value);
        match self.logger.as_mut() {
            Some(log) => log.log(&rec)?,
            None => return Ok(()),
        }
        Ok(())
    }
}
