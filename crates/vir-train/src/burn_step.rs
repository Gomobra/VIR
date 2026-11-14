//! Jembatan nyata model Burn → trainer (F1.7 StepFn konkret).

use crate::optim::ParamStore;
use crate::trainer::{StepCtx, StepFn, StepOut};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::Device;
use vir_common::VirError;

/// Model mini nyata (Linear) untuk membuktikan jalur Burn→ParamStore.
#[derive(Module, Debug)]
pub struct MiniModel {
    pub fc: Linear,
}

impl MiniModel {
    pub fn new(in_dim: usize, out_dim: usize, device: &Device) -> Self {
        Self {
            fc: LinearConfig::new(in_dim, out_dim).init(device),
        }
    }
}

/// StepFn konkret: forward + MSE + backward, tulis grad ke ParamStore.
pub struct BurnStepFn {
    pub model: MiniModel,
    pub x: Tensor<2>,
    pub y: Tensor<2>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl BurnStepFn {
    pub fn new(in_dim: usize, out_dim: usize, batch: usize, device: &Device) -> Self {
        let model = MiniModel::new(in_dim, out_dim, device);
        let x = Tensor::random(
            [batch, in_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let y = Tensor::random(
            [batch, out_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        Self {
            model,
            x,
            y,
            in_dim,
            out_dim,
        }
    }
}

impl StepFn for BurnStepFn {
    fn compute(&mut self, _ctx: StepCtx<'_>, store: &mut ParamStore) -> Result<StepOut, VirError> {
        // forward
        let pred = self.model.fc.forward(self.x.clone());
        let diff = pred - self.y.clone();
        let loss = (diff.clone() * diff).mean();
        let loss_val = loss.clone().into_scalar::<f32>();

        // backward (autodiff)
        let grads = loss.backward();
        let w = self.model.fc.weight.val();
        let grad_w = w.grad(&grads).unwrap_or_else(|| Tensor::zeros_like(&w));
        // tulis ke ParamStore (slot 0)
        let flat: Vec<f32> = grad_w.into_data().try_to_vec::<f32>().unwrap_or_default();
        if let Some(slot) = store.slots.first_mut() {
            slot.grad = flat;
        }

        Ok(StepOut {
            loss_total: loss_val,
            ..StepOut::default()
        })
    }
}
