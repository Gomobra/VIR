//! Muon Newton–Schulz **batched** (F2.5): batch NS lintas param 2D se-bentuk.
//!
//! Muon standar (burn-optim) menjalankan Newton–Schulz per-param → ±N·20
//! luncuran matmul kecil per langkah (launch-bound). Modul ini menumpuk param
//! 2D se-bentuk jadi `[N, m, n]` lalu satu NS ter-batch → ±20 luncuran per
//! kelompok bentuk. Numerik setara (NS per-matrik independen).
//!
//! Dua-pass: `visit` (kumpul grad 2D by ParamId) → batch NS → `map` (tulis-balik).

use std::collections::HashMap;

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::tensor::Tensor;

use crate::model::VirTrainModel;

/// Iterasi Newton–Schulz (PRD F1.7 langkah 5).
const NS_STEPS: usize = 5;
/// Koefisien quintik Muon (PRD:604).
const NS_A: f32 = 3.4445;
const NS_B: f32 = -4.7750;
const NS_C: f32 = 2.0315;
/// Momentum Muon (PRD:604).
const MUON_MOMENTUM: f32 = 0.95;

/// Newton–Schulz quintik ter-batch pada `[N, m, n]` (per-matrik independen).
pub fn batched_newton_schulz(x: Tensor<3>) -> Tensor<3> {
    let [n, m, k] = x.dims();
    let trans = m > k;
    let x = if trans { x.transpose() } else { x };
    // Normalisasi Frobenius per-matrik (per-batch).
    let norm = x
        .clone()
        .powf_scalar(2.0)
        .sum_dim(2)
        .sum_dim(1)
        .sqrt()
        .clamp_min(1e-7);
    let norm = norm.reshape([n, 1, 1]);
    let mut x = x / norm;
    for _ in 0..NS_STEPS {
        let a = x.clone().matmul(x.clone().transpose()); // [N, m, m]
        let ax = a.clone().matmul(x.clone()); // [N, m, k]
        let aax = a.matmul(ax.clone()); // [N, m, k]
        x = x.mul_scalar(NS_A) + ax.mul_scalar(NS_B) + aax.mul_scalar(NS_C);
    }
    let _ = n;
    if trans { x.transpose() } else { x }
}

/// Kunci momentum (ParamId numerik + bentuk).
type MomKey = (u64, Vec<usize>);

/// Muon ter-batch: momentum per param 2D + NS ter-batch per langkah.
pub struct BatchedMuon {
    buf: HashMap<MomKey, Tensor<2>>,
}

impl BatchedMuon {
    pub fn new() -> Self {
        Self {
            buf: HashMap::new(),
        }
    }

    /// Hitung update Muon ter-batch untuk param 2D dari `grads`.
    ///
    /// Mengembalikan peta `id numerik → update` (siap ditulis-balik via `map`).
    pub fn compute(
        &mut self,
        model: &VirTrainModel,
        grads: &GradientsParams,
        lr: f64,
    ) -> HashMap<u64, Tensor<2>> {
        // Pass 1: kumpulkan (id, grad) untuk param 2D.
        let mut collector = GradCollector {
            grads,
            collected: Vec::new(),
        };
        model.visit(&mut collector);
        // Kelompokkan per bentuk.
        let mut groups: HashMap<Vec<usize>, Vec<(u64, Tensor<2>)>> = HashMap::new();
        for (id, g) in collector.collected {
            let shape = g.dims();
            groups.entry(shape.to_vec()).or_default().push((id, g));
        }
        // Pass 2: momentum + NS ter-batch per kelompok.
        let mut updates = HashMap::new();
        for (shape, items) in groups {
            let mut bufs: Vec<Tensor<2>> = Vec::with_capacity(items.len());
            for (id, g) in &items {
                let key: MomKey = (*id, shape.clone());
                let buf = self
                    .buf
                    .entry(key)
                    .or_insert_with(|| Tensor::<2>::zeros([shape[0], shape[1]], &g.device()));
                *buf = buf.clone().mul_scalar(MUON_MOMENTUM) + g.clone();
                bufs.push(buf.clone());
            }
            let stacked = Tensor::stack(bufs, 0); // [N, m, n]
            let upd = batched_newton_schulz(stacked).mul_scalar(lr as f32);
            let [_, m, k] = upd.dims();
            for (i, (id, _)) in items.iter().enumerate() {
                let t = upd.clone().slice([i..i + 1, 0..m, 0..k]).squeeze_dim(0);
                updates.insert(*id, t);
            }
        }
        updates
    }
}

/// Visitor: kumpulkan (id, grad) untuk param 2D.
struct GradCollector<'a> {
    grads: &'a GradientsParams,
    collected: Vec<(u64, Tensor<2>)>,
}

impl ModuleVisitor for GradCollector<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if D == 2 {
            let id = param.id;
            if let Some(g) = self.grads.get::<2>(id) {
                self.collected.push((id.val(), g));
            }
        }
    }
}

/// Applier: tulis-balik update Muon ke param 2D via `map`.
struct Applier<'a> {
    updates: &'a HashMap<u64, Tensor<2>>,
}

impl ModuleMapper for Applier<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        if D == 2 {
            let id = param.id.val();
            if let Some(new_t) = self.updates.get(&id) {
                let dt = new_t.clone().into_dispatch();
                return param.map(move |_| Tensor::<D>::from_dispatch(dt.clone()));
            }
        }
        param
    }
}

/// Terapkan update Muon ter-batch ke model (pass 2).
pub fn apply_muon(model: VirTrainModel, updates: &HashMap<u64, Tensor<2>>) -> VirTrainModel {
    let mut applier = Applier { updates };
    model.map(&mut applier)
}
