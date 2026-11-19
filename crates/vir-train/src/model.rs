//! Model latih nyata (PRD F1.7, F2.5.b): tokenizer VQ multiscale + decoder
//! (tahap 1) dan generator NCM (tahap 2) dalam satu `Module` Burn.
//!
//! ```text
//! VirTrainModel {
//!   embed: [vocab, d]              // token → laten (tahap 1)
//!   vq:    MultiscaleVq<Codebook>  // tokenizer (tahap 1, beku di tahap 2)
//!   dec:   Decoder                 // ẑ → logit token (tahap 1)
//!   generator:   Generator               // NCM (tahap 2)
//! }
//! ```
//!
//! Pembobotan peran parameter ada di [`crate::params`] (embed/vq-phi/dec = Vq,
//! codebook + salinan tokenizer di generator = Tokenizer, inti generator =
//! Generator). Model ini tidak menyimpan state optimizer; bobot otoritatif ada di
//! [`crate::optim::ParamStore`] dan disalin ke model setiap langkah.

use std::sync::{Mutex, MutexGuard};

use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{Device, TensorData};
use vir_common::{VirError, level_schedule};
use vir_model::{Generator, GeneratorConfig};
use vir_vq::{Codebook, CodebookConfig, Decoder, DecoderConfig, MultiscaleVq};

use crate::rng::Rng;

/// Konfigurasi model latih.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VirTrainConfig {
    /// Lebar laten/hidden `d`.
    pub dim: usize,
    /// Jumlah head attention (`dim % num_heads == 0`).
    pub num_heads: usize,
    /// Panjang level `ℓ_k` (naik ketat, berakhir `L`).
    pub levels: Vec<usize>,
    /// Ukuran codebook `|V_k|`.
    pub codebook_size: usize,
    /// Ukuran vocab token (harus sama dengan tokenizer shard).
    pub vocab: usize,
    /// Kernel `Phi`; `<= 1` memakai `Phi` nol.
    pub phi_kernel: usize,
    /// Blok prelude generator.
    pub prelude_blocks: usize,
    /// Blok inti generator.
    pub core_blocks: usize,
    /// Blok coda generator.
    pub coda_blocks: usize,
    /// Iterasi rekurensi saat latihan.
    pub num_loops: usize,
    /// Lebar MLP decoder (`0` = `4·dim`).
    pub decoder_hidden: usize,
    /// Bobot suku komitmen R4 (`β`).
    pub commitment_cost: f32,
    /// Eksponen bobot level NCM (`α`).
    pub alpha: f32,
    /// Potongan waktu CE decoder (`0` = satu potongan penuh).
    pub chunk: usize,
    /// Seed untuk init bobot dan RNG device (agar model identik antar-run).
    pub seed: u64,
}

impl Default for VirTrainConfig {
    fn default() -> Self {
        let base = vir_common::ModelConfig::default();
        Self {
            dim: base.latent_dim,
            num_heads: base.num_heads,
            levels: level_schedule(base.schedule, base.target_length, base.num_levels),
            codebook_size: base.codebook_size,
            vocab: 4096,
            phi_kernel: 3,
            prelude_blocks: 1,
            core_blocks: 2,
            coda_blocks: 1,
            num_loops: 2,
            decoder_hidden: 0,
            commitment_cost: base.commitment_cost,
            alpha: base.alpha_level_weight,
            chunk: 0,
            seed: base.seed,
        }
    }
}

impl VirTrainConfig {
    /// Validasi jadwal level dan dimensi (R2).
    pub fn validate(&self) -> Result<(), VirError> {
        if self.dim == 0 || self.num_heads == 0 || self.dim % self.num_heads != 0 {
            return Err(VirError::Config(format!(
                "dim={} tidak habis dibagi num_heads={}",
                self.dim, self.num_heads
            )));
        }
        if self.levels.is_empty() {
            return Err(VirError::Config("jadwal level kosong".into()));
        }
        if self.levels[0] == 0 {
            return Err(VirError::Config("panjang level harus > 0".into()));
        }
        if self.levels.windows(2).any(|w| w[0] >= w[1]) {
            return Err(VirError::Config("jadwal level harus naik ketat".into()));
        }
        if self.codebook_size == 0 || self.vocab == 0 {
            return Err(VirError::Config("codebook_size/vocab harus > 0".into()));
        }
        if self.prelude_blocks == 0 || self.core_blocks == 0 || self.coda_blocks == 0 {
            return Err(VirError::Config("jumlah blok generator harus > 0".into()));
        }
        if self.num_loops == 0 {
            return Err(VirError::Config("num_loops harus > 0".into()));
        }
        Ok(())
    }

    /// Panjang konteks `L` (level terakhir).
    pub fn total_len(&self) -> usize {
        self.levels.last().copied().unwrap_or(0)
    }

    /// Panjang level ke-`k`.
    pub fn level_len(&self, k: usize) -> usize {
        self.levels.get(k).copied().unwrap_or(0)
    }

    /// Bobot level ternormalisasi (`α`).
    pub fn weights(&self) -> Vec<f32> {
        crate::loss::level_weight(&self.levels, self.alpha)
    }
}

/// Model latih dua tahap.
#[derive(Module, Debug)]
pub struct VirTrainModel {
    /// Embedding token `[vocab, d]` (tahap 1).
    pub embed: Param<Tensor<2>>,
    /// Tokenizer VQ multiscale.
    pub vq: MultiscaleVq<Codebook>,
    /// Decoder token (tahap 1).
    pub dec: Decoder,
    /// Generator NCM (tahap 2).
    pub generator: Generator,
    #[module(skip)]
    cfg: VirTrainConfig,
}

/// Hasil enkode VQ satu langkah (tahap 1).
pub struct VqEncoded {
    /// Rekonstruksi kumulatif akhir `ẑ^(K)`.
    pub z_hat: Tensor<3>,
    /// Rekonstruksi kumulatif per level `ẑ^(k)` (`k = 0..K-1`).
    pub levels: Vec<Tensor<3>>,
    /// Indeks kode per level `[B, ℓ_k]`.
    pub indices: Vec<Tensor<2, Int>>,
}

impl VirTrainModel {
    /// Rakit model dari konfigurasi; bobot awal hanya fungsi `cfg.seed`.
    ///
    /// Seed + seluruh tarikan init dijalankan di dalam satu kunci proses dan
    /// parameter lazy dipaksa ditarik sebelum kunci lepas (lihat [`INIT_LOCK`]).
    pub fn new(cfg: &VirTrainConfig, device: &Device) -> Result<Self, VirError> {
        cfg.validate()?;
        let _guard = init_guard();
        device.seed(cfg.seed);
        let model = Self::build(cfg, device)?;
        crate::params::init_params(&model);
        Ok(model)
    }

    /// Rakit modul dengan bobot lazy dari RNG device (tanpa seed/kunci).
    fn build(cfg: &VirTrainConfig, device: &Device) -> Result<Self, VirError> {
        let cb = CodebookConfig {
            num_entries: cfg.codebook_size,
            dim: cfg.dim,
            ..CodebookConfig::default()
        };
        let vq = MultiscaleVq::new(
            &cfg.levels,
            cfg.phi_kernel,
            |_| Codebook::new(&cb, device),
            device,
        );
        let dec = Decoder::with_config(
            &DecoderConfig {
                dim: cfg.dim,
                vocab: cfg.vocab,
                hidden: cfg.decoder_hidden,
            },
            device,
        );
        let mut gcfg = GeneratorConfig::new(cfg.dim, cfg.num_heads, cfg.levels.clone());
        gcfg.codebook_size = cfg.codebook_size;
        gcfg.phi_kernel = cfg.phi_kernel;
        gcfg.prelude_blocks = cfg.prelude_blocks;
        gcfg.core_blocks = cfg.core_blocks;
        gcfg.coda_blocks = cfg.coda_blocks;
        gcfg.num_loops = cfg.num_loops;
        let generator = Generator::new(&gcfg, device)?;
        let embed = Param::from_tensor(Tensor::<2>::from_floats(
            TensorData::new(init_embed(cfg), [cfg.vocab, cfg.dim]),
            device,
        ));
        Ok(Self {
            embed,
            vq,
            dec,
            generator,
            cfg: cfg.clone(),
        })
    }

    /// Konfigurasi model.
    pub fn config(&self) -> &VirTrainConfig {
        &self.cfg
    }

    /// Lebar laten `d`.
    pub fn dim(&self) -> usize {
        self.cfg.dim
    }

    /// Panjang konteks `L`.
    pub fn total_len(&self) -> usize {
        self.cfg.total_len()
    }

    /// Jumlah level `K`.
    pub fn num_levels(&self) -> usize {
        self.cfg.levels.len()
    }

    /// Token `[B, L]` → laten `[B, L, d]`.
    ///
    /// Token di luar `[0, vocab)` dipastikan sudah ditolak pemanggil
    /// ([`crate::burn_step::VirStepFn`]).
    pub fn embed_tokens(&self, tokens: &Tensor<2, Int>) -> Tensor<3> {
        let [b, l] = tokens.dims();
        let flat = tokens.clone().reshape([b * l]);
        self.embed
            .val()
            .select(0, flat)
            .reshape([b, l, self.cfg.dim])
    }

    /// Enkode multiscale (tanpa EMA): dipakai tahap 2 untuk target kode.
    pub fn encode(&self, z: Tensor<3>) -> VqEncoded {
        let out = self.vq.encode_full(z);
        VqEncoded {
            z_hat: out.final_z,
            levels: out.z_hat,
            indices: out.indices,
        }
    }

    /// Enkode multiscale + pembaruan EMA codebook tiap level (tahap 1).
    ///
    /// Loop ini identik dengan `MultiscaleVq::encode` tetapi menyalurkan `u_k`
    /// (masukan kuantizer pra-kuantisasi) ke `Quantizer::update` supaya codebook
    /// belajar EMA (PRD F1.4.a) — `LevelOutput` tidak membawanya.
    pub fn encode_with_ema(&mut self, z: Tensor<3>, rng: &mut Rng) -> Result<VqEncoded, VirError> {
        use vir_vq::Quantizer;
        let mut z_hat: Tensor<3> = z.zeros_like();
        let mut residual = z.clone();
        let mut levels = Vec::with_capacity(self.cfg.levels.len());
        let mut indices = Vec::with_capacity(self.cfg.levels.len());
        let mut adapter = RngAdapter { rng };
        for k in 0..self.cfg.levels.len() {
            let out = self.vq.levels[k].forward(residual.clone());
            let down = self.vq.levels[k].scale.down(residual.clone());
            let [db, dl, dd] = down.dims();
            let flat = down.reshape([db * dl, dd]);
            let idx = out.indices.clone().reshape([db * dl]);
            // EMA adalah pembaruan state di luar graf: `detach` saja tidak cukup
            // karena leaf hasil detach tetap `Grad`, sehingga `sum`/`weight`
            // menjadi tensor non-leaf dan `Param::from_tensor` (`set_weight`)
            // panik "Can't convert a non leaf tensor into a tracked tensor".
            // `set_require_grad(false)` memutus total ke autodiff.
            let plain = flat.detach().set_require_grad(false);
            self.vq.levels[k]
                .quantizer
                .update(&idx, &plain, &mut adapter);
            z_hat = z_hat + out.contribution;
            residual = z.clone() - z_hat.clone();
            levels.push(z_hat.clone());
            indices.push(out.indices);
        }
        Ok(VqEncoded {
            z_hat,
            levels,
            indices,
        })
    }
}

/// Inisialisasi embedding: seragam `[-1, 1]`, sekelas dengan init codebook.
fn init_embed(cfg: &VirTrainConfig) -> Vec<f32> {
    let mut rng = Rng::new(cfg.seed ^ 0x00ED_0000);
    (0..cfg.vocab * cfg.dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect()
}

/// Kunci proses untuk inisialisasi bobot model.
///
/// RNG backend CPU bersifat **global** (`burn-ndarray` menyimpan
/// `SEED: Mutex<Option<StdRng>>` di static), dan bobot `Linear`/codebook
/// diinisialisasi *lazy* pada `val()` pertama. Akibatnya "seed → seluruh
/// tarikan" hanya reproducible bila tidak ada konstruksi lain yang menarik di
/// tengah jalan: pada runner `cargo nextest` (enam tes paralel) dua `VirStepFn`
/// dengan seed sama menghasilkan bobot awal berbeda, sehingga
/// `resume_generator_identik` gagal (slot 1 = codebook VQ beda).
///
/// Kunci ini membuat satu konstruksi model tak terpisah; pemanggil wajib
/// menyimpan penjaga sampai seluruh parameter lazy ditarik
/// ([`crate::params::init_params`]).
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// Penjaga [`INIT_LOCK`]; racun akibat panic diabaikan agar proses tetap jalan.
fn init_guard() -> MutexGuard<'static, ()> {
    INIT_LOCK.lock().unwrap_or_else(|err| err.into_inner())
}

/// Adaptor `rand::RngCore` (0.8) di atas RNG vir-train yang bisa di-checkpoint.
struct RngAdapter<'a> {
    rng: &'a mut Rng,
}

impl rand::RngCore for RngAdapter<'_> {
    fn next_u32(&mut self) -> u32 {
        self.rng.next_u64() as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for slot in dest.iter_mut() {
            *slot = (self.rng.next_u64() & 0xFF) as u8;
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

/// Device latih: autodiff bila belum, lalu di-seed agar init bobot deterministik.
///
/// Seed dilakukan **di dalam** [`INIT_LOCK`] yang sama dengan
/// [`VirTrainModel::new`], agar seed satu thread tak merusak init thread lain
/// (RNG backend CPU bersifat global).
pub fn train_device(device: Device, seed: u64) -> Device {
    let device = if device.is_autodiff() {
        device
    } else {
        device.autodiff()
    };
    let _guard = init_guard();
    device.seed(seed);
    device
}

/// Device latih dengan dtype float default dikonfigurasi (bf16 bila diminta).
///
/// `Device::configure` (Burn 0.22.1) mengunci default dtype **sebelum** tensor/
/// model dibuat. Mengembalikan error bila bf16 diminta tetapi device tak
/// mendukungnya (`supports_dtype`).
pub fn train_device_dtype(device: Device, seed: u64, bf16: bool) -> Result<Device, VirError> {
    let mut device = if device.is_autodiff() {
        device
    } else {
        device.autodiff()
    };
    let _guard = init_guard();
    if bf16 {
        if !device.supports_dtype(burn::tensor::FloatDType::BF16) {
            return Err(VirError::Internal(
                "device tidak mendukung bf16".to_string(),
            ));
        }
        device
            .configure((burn::tensor::FloatDType::BF16, burn::tensor::IntDType::I32))
            .map_err(|e| VirError::Internal(format!("configure bf16 gagal: {e}")))?;
    }
    device.seed(seed);
    Ok(device)
}
