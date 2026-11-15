//! Inti eksekusi irisan mini (PRD F1.8): VQ → generator → latih kepala →
//! generate → dekode. Dipisah dari [`crate::smoke_mini`] agar tiap berkas
//! memenuhi R3 (≤400 baris); tipe publik ada di modul induk.

use burn::prelude::*;
use burn::tensor::{Device, TensorData};
use vir_common::{DeterministicRng, VirError};
use vir_vq::{Codebook, CodebookConfig, Decoder, DecoderConfig, MultiscaleVq};

use crate::generator::{Generator, GeneratorConfig};
use crate::loss::level_losses;
use crate::sampling::Sampler;
use crate::scale::LevelScales;
use crate::smoke::{codes_equal, smoke_device};
use crate::smoke_host::{accuracy, decode_tokens, fit_heads_exact, solve_lstsq};
use crate::smoke_mini::MiniSmokeConfig;

/// Hasil antara sebelum gerbang.
pub(crate) struct Raw {
    pub(crate) levels: Vec<usize>,
    pub(crate) total_positions: usize,
    pub(crate) loss_before: Vec<f32>,
    pub(crate) loss_after: Vec<f32>,
    pub(crate) acc_before: Vec<f32>,
    pub(crate) acc_after: Vec<f32>,
    pub(crate) recon_error: Vec<f32>,
    pub(crate) exact_codes: bool,
    pub(crate) exact_tokens: bool,
    pub(crate) generated_tokens: Vec<u32>,
    pub(crate) positions: usize,
}

/// Inti irisan: VQ → generator → latih → generate → dekode.
pub(crate) fn execute(cfg: &MiniSmokeConfig) -> Result<Raw, VirError> {
    if cfg.samples.is_empty() {
        return Err(VirError::Config("sampel kosong".into()));
    }
    let device = smoke_device();
    let len = *cfg
        .levels
        .last()
        .ok_or_else(|| VirError::Config("jadwal level kosong".into()))?;
    let (vq, targets, recon_error) = encode_samples(cfg, len, &device);
    let mut model = build_generator(cfg, &vq, &device)?;
    let batch = cfg.samples.len();
    let init = build_init(
        batch,
        cfg.prefix_len + len,
        cfg.dim,
        cfg.seed ^ 0x5EED_0000,
        &device,
    );
    let (scales, inputs, target_z) = teacher_context(&model, &init, &targets, cfg.prefix_len, len);

    let before = model.forward_with(&scales, &inputs);
    let (loss_before, acc_before) = level_metrics(&before, &targets);
    let features = model.forward_features_with(&scales, &inputs, model.num_loops());
    fit_heads_exact(&mut model, &features, &targets)?;
    let after = model.forward_with(&scales, &inputs);
    let (loss_after, acc_after) = level_metrics(&after, &targets);

    let out = model.generate_from(init, cfg.prefix_len, &Sampler::greedy())?;
    let exact_codes = codes_equal(&out.codes, &targets);
    let gen_z =
        out.z_hat
            .clone()
            .slice([0..batch, cfg.prefix_len..cfg.prefix_len + len, 0..cfg.dim]);
    let samples = normalize_samples(&cfg.samples, len);
    let input_tokens: Vec<u32> = samples.iter().flatten().copied().collect();
    let dec = fit_linear_decoder(&target_z, &input_tokens, cfg.dim, cfg.vocab, &device)?;
    let generated_tokens = decode_tokens(&dec, &gen_z);
    let exact_tokens = generated_tokens == input_tokens;

    Ok(Raw {
        levels: cfg.levels.clone(),
        total_positions: cfg.levels.iter().sum(),
        loss_before,
        loss_after,
        acc_before,
        acc_after,
        recon_error,
        exact_codes,
        exact_tokens,
        generated_tokens,
        positions: out.positions,
    })
}

/// Encode sampel token → VQ multiscale; kembalikan tokenizer, target, galat rekonstruksi.
fn encode_samples(
    cfg: &MiniSmokeConfig,
    len: usize,
    device: &Device,
) -> (MultiscaleVq<Codebook>, Vec<Tensor<2, Int>>, Vec<f32>) {
    let samples = normalize_samples(&cfg.samples, len);
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
    let z = token_latents(&samples, cfg.dim, cfg.vocab, cfg.seed, device);
    let encoded = vq.encode_full(z.clone());
    let recon_error = cumulative_recon_error(&z, &encoded.z_hat);
    (vq, encoded.indices, recon_error)
}

/// Rakit generator dengan codebook dan `Phi` tokenizer beku (F1.7 tahap 2).
fn build_generator(
    cfg: &MiniSmokeConfig,
    vq: &MultiscaleVq<Codebook>,
    device: &Device,
) -> Result<Generator, VirError> {
    let gcfg = GeneratorConfig {
        dim: cfg.dim,
        num_heads: cfg.num_heads,
        levels: cfg.levels.clone(),
        prelude_blocks: 1,
        core_blocks: 2,
        coda_blocks: 1,
        num_loops: 2,
        mean_loops: 32.0,
        loop_sigma: 0.5,
        codebook_size: cfg.codebook_size,
        phi_kernel: cfg.phi_kernel,
    };
    gcfg.validate()?;
    let mut model = Generator::new(&gcfg, device)?;
    model.load_tokenizer(vq)?;
    Ok(model)
}

/// Konteks teacher: benih awal + rekonstruksi kumulatif tiap level (ctx = `P + L`).
///
/// Rekonstruksi memakai operator generator (bukan `z_hat` VQ) agar identik
/// bit demi bit dengan loop `generate`.
#[allow(clippy::type_complexity)]
fn teacher_context(
    model: &Generator,
    init: &Tensor<3>,
    targets: &[Tensor<2, Int>],
    p: usize,
    l: usize,
) -> (LevelScales, Vec<Tensor<3>>, Tensor<3>) {
    let scales = model.scales_for(init.dims()[1]);
    let [b, _, d] = init.dims();
    let mut cur = init.clone();
    let mut inputs = Vec::with_capacity(targets.len());
    for k in 0..targets.len() {
        inputs.push(cur.clone());
        let codes = model.embed_codes(k, targets[k].clone());
        cur = cur + scales.reconstruct(k, &model.phis[k], codes);
    }
    let block = cur.slice([0..b, p..p + l, 0..d]);
    (scales, inputs, block)
}

/// Galat rekonstruksi kumulatif VQ per level: `‖z − ẑ^(k)‖²`.
fn cumulative_recon_error(z: &Tensor<3>, z_hats: &[Tensor<3>]) -> Vec<f32> {
    z_hats
        .iter()
        .map(|zh| {
            let d = zh.clone() - z.clone();
            (d.clone() * d).mean().into_scalar::<f32>()
        })
        .collect()
}

/// CE per level (tanpa bobot) dan akurasi argmax.
fn level_metrics(logits: &[Tensor<3>], targets: &[Tensor<2, Int>]) -> (Vec<f32>, Vec<f32>) {
    let ones = vec![1.0f32; logits.len()];
    let loss = level_losses(logits, targets, &ones)
        .into_iter()
        .map(|t| t.into_scalar::<f32>())
        .collect();
    let acc = (0..logits.len())
        .map(|k| accuracy(&logits[k..k + 1], &targets[k..k + 1]))
        .collect();
    (loss, acc)
}

/// Benih awal per (sampel, posisi) `[B, P+L, d]` (deterministik).
///
/// Memberi identitas tiap baris pada level 0; tanpa ini level 0 melihat nol
/// untuk semua sampel sehingga tidak dapat dibedakan.
fn build_init(b: usize, ctx: usize, d: usize, seed: u64, device: &Device) -> Tensor<3> {
    let mut rng = DeterministicRng::new(seed);
    let mut data = vec![0.0f32; b * ctx * d];
    for v in data.iter_mut() {
        *v = rand::Rng::gen_range(rng.inner_mut(), -1.0f32..1.0);
    }
    Tensor::<3>::from_floats(TensorData::new(data, [b, ctx, d]), device)
}

/// Embedding token beku → laten `[B, L, d]`.
fn token_latents(
    samples: &[Vec<u32>],
    dim: usize,
    vocab: usize,
    seed: u64,
    device: &Device,
) -> Tensor<3> {
    let b = samples.len();
    let len = samples.first().map_or(0, |s| s.len());
    let hi = vocab.max(1);
    let mut rng = DeterministicRng::new(seed ^ 0x1E_0000);
    let mut emb = vec![0.0f32; hi * dim];
    for v in emb.iter_mut() {
        *v = rand::Rng::gen_range(rng.inner_mut(), -1.0f32..1.0);
    }
    let mut data = vec![0.0f32; b * len * dim];
    for (i, s) in samples.iter().enumerate() {
        for (j, &t) in s.iter().enumerate() {
            let tok = (t as usize) % hi;
            for d in 0..dim {
                data[(i * len + j) * dim + d] = emb[tok * dim + d];
            }
        }
    }
    Tensor::<3>::from_floats(TensorData::new(data, [b, len, dim]), device)
}

/// Panjangkan/pangkas tiap sampel ke `len`.
fn normalize_samples(samples: &[Vec<u32>], len: usize) -> Vec<Vec<u32>> {
    samples
        .iter()
        .map(|s| {
            let mut row = s.clone();
            row.truncate(len);
            row.resize(len, 0);
            row
        })
        .collect()
}

/// Decoder linear yang dilatih (kuadrat terkecil) untuk memetakan `ẑ → token`.
///
/// `vir-vq` `Decoder` dipakai apa adanya; MLP-nya dinolkan sehingga
/// `forward(z) = head(z)` (gelu(0) = 0), lalu `head` dipasang dari solusi
/// kuadrat terkecil. Ini melatih "kepala decoder" pada 8 sampel (F1.8).
fn fit_linear_decoder(
    z: &Tensor<3>,
    tokens: &[u32],
    dim: usize,
    vocab: usize,
    device: &Device,
) -> Result<Decoder, VirError> {
    let [b, l, _] = z.dims();
    let n = b * l;
    let x = z
        .clone()
        .reshape([n, dim])
        .to_data()
        .try_to_vec::<f32>()
        .unwrap_or_default();
    if tokens.len() != n || x.len() != n * dim {
        return Err(VirError::Internal("data dekode tidak dapat dibaca".into()));
    }
    let y = one_hot_u32(tokens, vocab);
    let w = solve_lstsq(&x, &y, n, dim, vocab)
        .ok_or_else(|| VirError::Internal("dekode tidak dapat diselesaikan".into()))?;
    let mut dec = Decoder::with_config(
        &DecoderConfig {
            dim,
            vocab,
            hidden: dim,
        },
        device,
    );
    zero_mlp(&mut dec, dim, device);
    crate::generator::replace_head(&mut dec.head, w, dim, vocab, device)?;
    Ok(dec)
}

/// Nolkan MLP decoder agar `forward(z) = head(z)`.
fn zero_mlp(dec: &mut Decoder, dim: usize, device: &Device) {
    vir_vq::replace_param(&mut dec.fc1.weight, Tensor::<2>::zeros([dim, dim], device));
    if let Some(b) = dec.fc1.bias.as_mut() {
        vir_vq::replace_param(b, Tensor::<1>::zeros([dim], device));
    }
    if let Some(b) = dec.fc2.bias.as_mut() {
        vir_vq::replace_param(b, Tensor::<1>::zeros([dim], device));
    }
}

/// One-hot `[n, vocab]` dari token.
fn one_hot_u32(tokens: &[u32], vocab: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; tokens.len() * vocab];
    for (i, &t) in tokens.iter().enumerate() {
        let t = t as usize;
        if t < vocab {
            y[i * vocab + t] = 1.0;
        }
    }
    y
}
