//! Harness ablasi kuantizer (PRD F1.4.f): tokenizer mini + loop latihan CPU.
//!
//! Satu *arm* adalah tokenizer mini yang identik — embedding token `V×d` →
//! [`MultiscaleVq`] K level residual → [`Decoder`] — dengan satu kuantizer
//! berbeda. Yang diukur (angka nyata, bukan konstanta):
//!
//! | metrik | arti |
//! |---|---|
//! | `accuracy` | akurasi token eksak per level: `argmax` decoder == token masuk |
//! | `mse` | `‖z − ẑ^(k)‖²` per level (rate–distorsi, sebanding antar kuantizer) |
//! | `perplexity` / `utilization` | distribusi kode pada level terhalus |
//! | `dead` | entri codebook dengan hitungan EMA di bawah ambang restart |
//! | `loss_first`/`loss_last`/`non_finite` | stabilitas latihan |
//!
//! Determinisme: seed device dan RNG lintasan tetap, jadi angka sama pada CPU
//! yang sama (ADR-002 memakai angka ini sebagai bukti).

use super::Lcg;
use burn::module::{Module, Param};
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::{Device, Distribution, Int, Tensor, TensorData};
use std::fmt::Debug;
use vir_vq::codebook::{Codebook, CodebookConfig};
use vir_vq::fsq::{Fsq, FsqConfig};
use vir_vq::quantizer::Quantizer;
use vir_vq::{Decoder, EncodeOutput, MultiscaleVq, fsq_loss, level_accuracy, vq_loss_chunked};

/// Panjang blok `L` (NCM: 128).
pub const LEN: usize = 16;
/// Dimensi laten `d` (NCM: 384).
pub const DIM: usize = 32;
/// Ukuran kosakata (NCM: 8192–151680).
pub const VOCAB: usize = 64;
/// Jadwal level `ℓ_k`, naik ketat dan berakhir `L` (K = 4).
pub const SCHEDULE: [usize; 4] = [2, 4, 8, 16];
/// Kernel konvolusi `Phi` pada `R_k` (PRD F1.4.b langkah 4).
pub const PHI_KERNEL: usize = 3;
/// Ukuran batch latih.
pub const BATCH: usize = 16;
/// Jumlah deret latih. Tokenizer meng-encode corpusnya sendiri, jadi ini bukan
/// generalisasi: yang diukur adalah rekonstruksi (NCM Lampiran G).
pub const SAMPLES: usize = 64;
/// Bobot komitmen `β` (0,25 pada kode acuan NCM, PRD F1.4.d langkah 2).
pub const COMMIT_COST: f32 = 0.25;
/// Potongan sumbu waktu untuk CE (PRD F1.4.d langkah 3).
const CHUNK: usize = 8;
/// Seed lintasan (device + RNG + corpus).
const SEED: u64 = 0x5EED_0014;
/// Laju belajar AdamW untuk tokenizer mini.
const LR: f64 = 5e-3;

/// Kuantizer satu arm ablasi beserta loss yang menyertainya.
pub trait Arm {
    /// Tipe kuantizer yang dihasilkan.
    type Q: Quantizer + Module + Clone + Debug + Send;

    /// Rakit kuantizer untuk satu level di `device`.
    fn build(&self, device: &Device) -> Self::Q;

    /// `Some(β)` bila arm memakai loss komitmen R4 (VQ-EMA); `None` untuk FSQ
    /// dan variannya, yang tidak butuh loss tambahan (PRD F1.4.f).
    fn commitment(&self) -> Option<f32>;
}

/// VQ-EMA dengan `|V|` entri (baseline NCM Tabel 5).
pub struct VqArm(pub usize);

impl Arm for VqArm {
    type Q = Codebook;

    fn build(&self, device: &Device) -> Codebook {
        Codebook::new(&codebook_config(self.0), device)
    }

    fn commitment(&self) -> Option<f32> {
        Some(COMMIT_COST)
    }
}

/// FSQ dengan level `L_i` per dimensi kode; `ifsq` memilih aktivasi sigmoid.
pub struct FsqArm {
    pub levels: Vec<usize>,
    pub ifsq: bool,
}

impl Arm for FsqArm {
    type Q = Fsq;

    fn build(&self, device: &Device) -> Fsq {
        let cfg = if self.ifsq {
            FsqConfig::ifsq(self.levels.clone())
        } else {
            FsqConfig::new(self.levels.clone())
        };
        Fsq::new(&cfg, DIM, device).expect("fsq")
    }

    fn commitment(&self) -> Option<f32> {
        None
    }
}

/// Tokenizer mini: embedding token `V×d` → `MultiscaleVq` (K level) → decoder.
///
/// NCM memakai encoder 0 blok transformer (tabel embedding saja, PRD F1.4
/// langkah 1), jadi `z = embed[x]` langsung.
#[derive(Module)]
pub struct Tokenizer<Q: Quantizer + Module + Debug> {
    embed: Param<Tensor<2>>,
    pub vq: MultiscaleVq<Q>,
    pub decoder: Decoder,
    #[module(skip)]
    dim: usize,
}

impl<Q: Quantizer + Module> Tokenizer<Q> {
    /// Rakit tokenizer di atas `vq` yang sudah berisi kuantizer tiap level.
    pub fn new(vq: MultiscaleVq<Q>, device: &Device) -> Self {
        let dim = vq.levels[0].quantizer.dim();
        let embed = Param::from_tensor(Tensor::random(
            [VOCAB, dim],
            Distribution::Normal(0.0, 0.5),
            device,
        ));
        Self {
            embed,
            vq,
            decoder: Decoder::new(dim, VOCAB, device),
            dim,
        }
    }

    /// Embedding token `[B, L]` → laten `z` `[B, L, d]`.
    pub fn embed_tokens(&self, tokens: &Tensor<2, Int>) -> Tensor<3> {
        let [b, l] = tokens.dims();
        let flat = tokens.clone().reshape([b * l]);
        self.embed.val().select(0, flat).reshape([b, l, self.dim])
    }

    /// Encode multiscale penuh dari token.
    pub fn encode_tokens(&self, tokens: &Tensor<2, Int>) -> (Tensor<3>, EncodeOutput) {
        let z = self.embed_tokens(tokens);
        let out = self.vq.encode_full(z.clone());
        (z, out)
    }
}

/// Ringkasan stabilitas latihan satu arm.
#[derive(Debug, Clone, Copy)]
pub struct Stability {
    pub loss_first: f32,
    pub loss_last: f32,
    pub non_finite: usize,
}

/// Hasil satu arm ablasi (angka hasil pengukuran, bukan konstanta).
#[derive(Debug, Clone)]
pub struct ArmReport {
    pub name: String,
    pub stab: Stability,
    /// `|V_k|` per level.
    pub codes: Vec<usize>,
    /// `‖z − ẑ^(k)‖²` per level.
    pub mse: Vec<f32>,
    pub accuracy: Vec<f64>,
    pub perplexity: Vec<f32>,
    pub utilization: Vec<f64>,
    pub dead: Vec<usize>,
}

impl ArmReport {
    /// Akurasi rekonstruksi eksak pada level penuh `k = K`.
    pub fn final_accuracy(&self) -> f64 {
        self.accuracy.last().copied().unwrap_or(0.0)
    }

    /// `(utilisasi, perplexity, kode mati)` pada level terhalus.
    pub fn final_usage(&self) -> (f64, f32, usize) {
        let last = self.codes.len().saturating_sub(1);
        (
            self.utilization[last],
            self.perplexity[last],
            self.dead[last],
        )
    }
}

/// Latih satu arm lalu ukur hasilnya.
pub fn run_arm<A: Arm>(name: &str, steps: usize, arm: &A) -> ArmReport {
    let device = Device::default().autodiff();
    device.seed(SEED);
    let vq = MultiscaleVq::new(&SCHEDULE, PHI_KERNEL, |_| arm.build(&device), &device);
    let mut model = Tokenizer::new(vq, &device);
    let mut opt = AdamWConfig::new().init();
    let mut rng = Lcg::new(SEED ^ 0xA11CE);
    let corpus = synth_tokens(SAMPLES, LEN, VOCAB, SEED ^ 0xC0FFEE);
    let aux = arm.commitment();
    let mut stab = Stability {
        loss_first: 0.0,
        loss_last: 0.0,
        non_finite: 0,
    };

    for step in 0..steps {
        let tokens = batch_tensor(&corpus, step, &device);
        let (z, enc) = model.encode_tokens(&tokens);
        let loss = arm_loss(&model, &z, &enc, &tokens, aux);
        let value = loss.clone().into_scalar::<f32>();
        if !value.is_finite() {
            stab.non_finite += 1;
        }
        if step == 0 {
            stab.loss_first = value;
        }
        stab.loss_last = value;
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = opt.step(LR, model, grads);
        update_codebooks(&mut model, &enc, &mut rng);
    }

    evaluate(name, &model, &corpus, &device, stab)
}

/// Loss arm: R4 (CE + komitmen + laten) untuk VQ-EMA, CE saja untuk FSQ.
fn arm_loss<Q: Quantizer + Module>(
    model: &Tokenizer<Q>,
    z: &Tensor<3>,
    enc: &EncodeOutput,
    target: &Tensor<2, Int>,
    aux: Option<f32>,
) -> Tensor<1> {
    match aux {
        Some(beta) => vq_loss_chunked(
            &model.decoder,
            z.clone(),
            enc.final_z.clone(),
            target.clone(),
            beta,
            CHUNK,
        ),
        None => fsq_loss(&model.decoder, enc.final_z.clone(), target.clone(), CHUNK),
    }
}

/// Pembaruan EMA + restart kode mati tiap level (PRD F1.4.a langkah 2).
fn update_codebooks<Q: Quantizer + Module>(
    model: &mut Tokenizer<Q>,
    enc: &EncodeOutput,
    rng: &mut Lcg,
) {
    for (k, level) in model.vq.levels.iter_mut().enumerate() {
        let [b, l] = enc.indices[k].dims();
        let idx = enc.indices[k].clone().reshape([b * l]);
        // EMA adalah pembaruan state di luar graf: `set_require_grad(false)`
        // memutus total ke autodiff, sebab `Param::from_tensor` menolak tensor
        // non-leaf yang masih terlacak (lihat vir-train `encode_with_ema`).
        let u = enc.level_inputs[k].clone().detach().set_require_grad(false);
        level.quantizer.update(&idx, &u, rng);
    }
}

/// Ukur arm terlatih: akurasi dan galat per level + statistik kode.
fn evaluate<Q: Quantizer + Module>(
    name: &str,
    model: &Tokenizer<Q>,
    corpus: &[i64],
    device: &Device,
    stab: Stability,
) -> ArmReport {
    let tokens =
        Tensor::<2, Int>::from_ints(TensorData::new(corpus.to_vec(), [SAMPLES, LEN]), device);
    let (z, enc) = model.encode_tokens(&tokens);
    let mut mse = Vec::with_capacity(enc.z_hat.len());
    let mut accuracy = Vec::with_capacity(enc.z_hat.len());
    for z_hat in enc.z_hat.iter() {
        mse.push(recon_mse(&z, z_hat));
        let logits = model.decoder.forward(z_hat.clone());
        accuracy.push(level_accuracy(&logits, &tokens));
    }
    let (perplexity, utilization, dead) = code_stats(model, &enc.indices);
    ArmReport {
        name: name.to_string(),
        stab,
        codes: model
            .vq
            .levels
            .iter()
            .map(|level| level.quantizer.num_codes())
            .collect(),
        mse,
        accuracy,
        perplexity,
        utilization,
        dead,
    }
}

/// `(perplexity, utilisasi, kode mati)` per level dari indeks kode.
fn code_stats<Q: Quantizer + Module>(
    model: &Tokenizer<Q>,
    indices: &[Tensor<2, Int>],
) -> (Vec<f32>, Vec<f64>, Vec<usize>) {
    let mut perplexity = Vec::with_capacity(indices.len());
    let mut utilization = Vec::with_capacity(indices.len());
    let mut dead = Vec::with_capacity(indices.len());
    for (k, idx) in indices.iter().enumerate() {
        let [b, l] = idx.dims();
        let stats = model.vq.levels[k]
            .quantizer
            .stats(&idx.clone().reshape([b * l]));
        perplexity.push(stats.perplexity);
        utilization.push(f64::from(stats.utilization));
        dead.push(model.vq.levels[k].quantizer.dead_codes());
    }
    (perplexity, utilization, dead)
}

/// `‖z − ẑ^(k)‖²` rata-rata: galat rekonstruksi laten per level.
fn recon_mse(z: &Tensor<3>, z_hat: &Tensor<3>) -> f32 {
    let diff = z.clone() - z_hat.clone();
    (diff.clone() * diff).mean().into_scalar::<f32>()
}

/// Deret token sintetis deterministik: baris aritmetika + token langka.
///
/// Pola aritmetika memberi struktur yang bisa dipelajari; token langka pada
/// posisi tetap memberi ketidaksetaraan antar posisi (NCM Tabel 11).
pub fn synth_tokens(n: usize, len: usize, vocab: usize, seed: u64) -> Vec<i64> {
    let mut rng = Lcg::new(seed);
    let span = vocab.max(4) as i64;
    let mut out = Vec::with_capacity(n * len);
    for _ in 0..n {
        let start = rng.range(vocab) as i64;
        let step = 1 + rng.range(5) as i64;
        let rare = span - 1 - rng.range(4) as i64;
        for t in 0..len {
            let mut value = (start + step * t as i64).rem_euclid(span);
            if t % 5 == 3 {
                value = rare;
            }
            out.push(value);
        }
    }
    out
}

/// Batch `[B, L]` dari corpus datar (baris berputar, deterministik).
fn batch_tensor(corpus: &[i64], step: usize, device: &Device) -> Tensor<2, Int> {
    let rows = SAMPLES.max(1);
    let mut batch = Vec::with_capacity(BATCH * LEN);
    for b in 0..BATCH {
        let row = (step * BATCH + b) % rows;
        batch.extend_from_slice(&corpus[row * LEN..(row + 1) * LEN]);
    }
    Tensor::<2, Int>::from_ints(TensorData::new(batch, [BATCH, LEN]), device)
}

/// Konfigurasi codebook VQ-EMA mini (NCM Tabel 5: β = 0,25, γ = 0,99).
pub fn codebook_config(entries: usize) -> CodebookConfig {
    CodebookConfig {
        num_entries: entries,
        dim: DIM,
        ema_decay: 0.99,
        commitment_cost: COMMIT_COST,
        restart_threshold: 1.0,
        eps: 1e-5,
    }
}
