//! Jembatan `ParamStore` ↔ modul Burn (F2.5.b).
//!
//! `ParamStore` adalah sumber kebenaran bobot (optimizer, clipping, checkpoint
//! bekerja di sana). Modul Burn hanya menampung tensor untuk forward/backward.
//! Tiga arah jembatan, semuanya lewat satu jalur jelajah `Module` yang sama
//! sehingga urutan slot store = urutan parameter model:
//!
//! | arah | fungsi | mekanisme |
//! |---|---|---|
//! | model → store | [`build_store`], [`copy_back`] | `ModuleVisitor` |
//! | store → model | [`load_store`] | `ModuleMapper` |
//! | model → grads | [`collect_grads`] | `ModuleVisitor` + `Gradients` |
//!
//! Nama slot adalah path modul berbatas titik (mis.
//! `generator.core.0.wq.weight`),
//! sehingga checkpoint bisa diinspeksi dan peran tahap diturunkan dari path.

use std::collections::HashSet;

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::prelude::*;
use burn::tensor::{Gradients, TensorData};
use vir_common::VirError;

use crate::model::VirTrainModel;
use crate::optim::{ParamRole, ParamSlot, ParamStore};
use crate::trainer::Stage;

/// Satu parameter hasil jelajah modul.
struct Found {
    name: String,
    rows: usize,
    cols: usize,
    role: ParamRole,
    data: Vec<f32>,
}

/// Peran tahap dari path modul (PRD F1.7 dua tahap).
///
/// - `codebook`/`quantizer`/`phis` → [`ParamRole::Tokenizer`] (dibekukan;
///   codebook belajar EMA, salinan tokenizer di generator tidak dilatih).
/// - `generator.*` (mis. `generator.core.0.wq.weight`) → [`ParamRole::Generator`].
/// - sisanya (embed, `vq.*.phi`, decoder) → [`ParamRole::Vq`].
///
/// Segmen pertama path adalah nama field pada [`VirTrainModel`], jadi generator
/// dikenali dari radix `generator` — bukan dari potongan `gen.` yang tidak
/// pernah muncul pada path nyata.
pub fn role_for_path(path: &str) -> ParamRole {
    role_for(path)
}

/// Peran tahap dari path modul (PRD F1.7 dua tahap).
fn role_for(path: &str) -> ParamRole {
    if path.contains("codebook") || path.contains("quantizer") || path.contains("phis") {
        return ParamRole::Tokenizer;
    }
    match path.split('.').next().unwrap_or_default() {
        "generator" => ParamRole::Generator,
        _ => ParamRole::Vq,
    }
}

/// Kumpulan parameter dari satu jelajah `Module`.
#[derive(Default)]
struct Finder {
    path: Vec<String>,
    found: Vec<Found>,
}

impl Finder {
    /// Path modul saat ini, berbatas titik.
    fn name(&self) -> String {
        self.path.join(".")
    }
}

impl ModuleVisitor for Finder {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let tensor = param.val();
        let shape = tensor.dims();
        let data: Vec<f32> = tensor.to_data().try_to_vec::<f32>().unwrap_or_default();
        let cols = shape.last().copied().unwrap_or(1).max(1);
        let rows = (shape.iter().product::<usize>().max(1) / cols).max(1);
        let name = self.name();
        self.found.push(Found {
            role: role_for(&name),
            name,
            rows,
            cols,
            data,
        });
    }
}

/// Jelajahi model; `Err` bila tidak ada parameter atau nama tidak unik.
fn scan(model: &VirTrainModel) -> Result<Vec<Found>, VirError> {
    let mut finder = Finder::default();
    model.visit(&mut finder);
    if finder.found.is_empty() {
        return Err(VirError::Internal("model tanpa parameter float".into()));
    }
    let mut seen: HashSet<&str> = HashSet::with_capacity(finder.found.len());
    for f in &finder.found {
        if !seen.insert(f.name.as_str()) {
            return Err(VirError::Internal(format!(
                "path parameter ganda: {}",
                f.name
            )));
        }
    }
    Ok(finder.found)
}

/// Kunci beku satu slot menurut peran dan tahap awal.
fn frozen_for(role: ParamRole, stage: Stage) -> bool {
    match role {
        ParamRole::Tokenizer => true,
        ParamRole::Generator => stage != Stage::Generator,
        ParamRole::Vq => stage == Stage::Generator,
    }
}

/// Selaraskan status beku store dengan tahap aktif (idempoten).
///
/// Tahap Vq: generator beku, Vq aktif. Tahap Generator: Vq beku, generator
/// aktif. Slot `Tokenizer` (codebook + salinan tokenizer) selalu beku terhadap
/// optimizer — codebook belajar EMA, bukan gradien.
pub fn sync_frozen(stage: Stage, store: &mut ParamStore) {
    for slot in store.slots.iter_mut() {
        slot.frozen = frozen_for(slot.role, stage);
    }
}

/// Bangun `ParamStore` dari bobot model sekarang (urutan jelajah modul).
pub fn build_store(model: &VirTrainModel, stage: Stage) -> Result<ParamStore, VirError> {
    let found = scan(model)?;
    let mut store = ParamStore::new();
    for f in &found {
        let mut slot = ParamSlot::new(&f.name, f.rows, f.cols, f.role, f.data.clone());
        slot.frozen = frozen_for(f.role, stage);
        store.add(slot);
    }
    Ok(store)
}

/// Paksa inisialisasi parameter lazy, tanpa menyalin nilainya.
///
/// Bobot `Linear`/codebook dibuat lazy oleh Burn (`Param::uninitialized`) dan
/// baru ditarik dari RNG device pada `val()` pertama. Inisialisasi model wajib
/// menarik semuanya sebelum kunci RNG dilepas, agar bobot awal hanya bergantung
/// pada seed (lihat `INIT_LOCK` di [`crate::model`]).
pub fn init_params(model: &VirTrainModel) {
    struct ForceInit;

    impl ModuleVisitor for ForceInit {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let _ = param.val();
        }
    }

    model.visit(&mut ForceInit);
}

/// Muat parameter model dari store.
///
/// Slot beku dimuat tanpa gradien (leaf `Requirement::None`) sehingga tidak
/// ikut graf autodiff; slot terlatih dimuat sebagai leaf `Grad` — `require_grad`
/// hanya sah pada leaf, dan `Param::from_tensor` di jalur EMA menolak non-leaf.
struct Loader<'a> {
    store: &'a ParamStore,
    index: usize,
    bad: usize,
}

impl ModuleMapper for Loader<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let Some(slot) = self.store.slots.get(self.index) else {
            self.bad += 1;
            return param;
        };
        self.index += 1;
        let tensor = param.val();
        let shape = tensor.dims();
        if slot.data.len() != shape.iter().product::<usize>() {
            self.bad += 1;
            return param;
        }
        let device = tensor.device();
        let value = Tensor::<D>::from_floats(TensorData::new(slot.data.clone(), shape), &device);
        let value = if slot.frozen {
            value
        } else {
            value.require_grad()
        };
        param.map(|_| value)
    }
}

/// Salin bobot store ke model (konsumsi model, kembalikan model baru).
pub fn load_store(model: VirTrainModel, store: &ParamStore) -> Result<VirTrainModel, VirError> {
    let mut loader = Loader {
        store,
        index: 0,
        bad: 0,
    };
    let model = model.map(&mut loader);
    if loader.bad != 0 {
        return Err(VirError::Internal(format!(
            "{} parameter gagal dimuat dari store",
            loader.bad
        )));
    }
    if loader.index != store.slots.len() {
        return Err(VirError::Internal(format!(
            "store punya {} slot, model punya {} parameter",
            store.slots.len(),
            loader.index
        )));
    }
    Ok(model)
}

/// Kumpulan gradien satu parameter.
struct Collector<'a> {
    grads: &'a Gradients,
    out: Vec<Vec<f32>>,
}

impl ModuleVisitor for Collector<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let tensor = param.val();
        let grad = tensor
            .grad(self.grads)
            .unwrap_or_else(|| tensor.zeros_like());
        self.out
            .push(grad.to_data().try_to_vec::<f32>().unwrap_or_default());
    }
}

/// Gradien tiap parameter model, urutan sama dengan [`build_store`].
pub fn collect_grads(model: &VirTrainModel, grads: &Gradients) -> Vec<Vec<f32>> {
    let mut collector = Collector {
        grads,
        out: Vec::new(),
    };
    model.visit(&mut collector);
    collector.out
}

/// Tulis gradien ke slot store yang tidak beku; kembalikan jumlah slot terisi.
pub fn write_grads(store: &mut ParamStore, grads: &[Vec<f32>]) -> Result<usize, VirError> {
    if grads.len() != store.slots.len() {
        return Err(VirError::Internal(format!(
            "gradien {} != slot {}",
            grads.len(),
            store.slots.len()
        )));
    }
    let mut filled = 0;
    for (slot, g) in store.slots.iter_mut().zip(grads.iter()) {
        if slot.frozen || slot.grad.len() != g.len() {
            continue;
        }
        slot.grad.copy_from_slice(g);
        filled += 1;
    }
    Ok(filled)
}

/// Salin parameter model kembali ke slot `Tokenizer` (bobot yang diperbarui di
/// model: EMA codebook dan salinan tokenizer generator).
///
/// Slot itu beku terhadap optimizer, tetapi wajib ikut checkpoint; tanpa salinan
/// ini resume kehilangan codebook hasil EMA.
pub fn copy_back(model: &VirTrainModel, store: &mut ParamStore) -> Result<usize, VirError> {
    let found = scan(model)?;
    if found.len() != store.slots.len() {
        return Err(VirError::Internal(
            "jumlah parameter model berubah sejak store dibangun".into(),
        ));
    }
    let mut copied = 0;
    for (f, slot) in found.iter().zip(store.slots.iter_mut()) {
        if f.name != slot.name {
            return Err(VirError::Internal(format!(
                "urutan parameter berubah: {} != {}",
                f.name, slot.name
            )));
        }
        if slot.role != ParamRole::Tokenizer {
            continue;
        }
        if slot.data.len() == f.data.len() && slot.data != f.data {
            slot.data.copy_from_slice(&f.data);
            copied += 1;
        }
    }
    Ok(copied)
}

/// Ekspor bobot model ke daftar `(path, shape, data)` untuk safetensors/checkpoint.
///
/// Path = nama modul berbatas titik; shape = `[rows, cols]`; data = f32 baris.
/// Urutan = urutan parameter model (dari `scan`), sehingga `import_safetensors`
/// dapat memuatnya kembali lewat `load_store` (yang memetakan per indeks).
pub fn export_safetensors(
    model: &VirTrainModel,
) -> Result<Vec<(String, Vec<usize>, Vec<f32>)>, VirError> {
    Ok(scan(model)?
        .into_iter()
        .map(|f| (f.name, vec![f.rows, f.cols], f.data))
        .collect())
}

/// Impor bobot dari daftar `(path, shape, data)` ke model (via `load_store`).
///
/// Urutan tensors harus sama dengan urutan parameter model (hasil `scan`).
/// `load_store` memakai shape asli tiap param; slot hanya butuh `data` + peran.
pub fn import_safetensors(
    model: VirTrainModel,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
) -> Result<VirTrainModel, VirError> {
    let mut store = ParamStore::new();
    for (name, shape, data) in tensors {
        let cols = shape.last().copied().unwrap_or(1).max(1);
        let rows = if cols > 0 { data.len() / cols } else { 0 };
        let role = role_for_path(name);
        store.add(ParamSlot::new(name, rows, cols.max(1), role, data.clone()));
    }
    load_store(model, &store)
}
