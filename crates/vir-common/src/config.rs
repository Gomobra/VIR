use crate::error::VirError;
use crate::schedule::level_schedule as make_schedule;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LevelScheduleKind {
    Linear,
    Quadratic,
    Sqrt,
    Log,
    Exp,
    Cosine,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub seed: u64,
    pub target_length: usize,
    pub num_levels: usize,
    pub schedule: LevelScheduleKind,
    pub codebook_size: usize,
    pub latent_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub dropout: f32,
    pub rope_base: f32,
    pub nested_dropout_prob: f32,
    pub ema_decay: f32,
    pub commitment_cost: f32,
    pub alpha_level_weight: f32,
    pub strict_paper: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            target_length: 128,
            num_levels: 12,
            schedule: LevelScheduleKind::Linear,
            codebook_size: 4096,
            latent_dim: 384,
            hidden_dim: 768,
            num_heads: 12,
            dropout: 0.1,
            rope_base: 10_000.0,
            nested_dropout_prob: 0.5,
            ema_decay: 0.99,
            commitment_cost: 0.25,
            alpha_level_weight: 0.0,
            strict_paper: false,
        }
    }
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), VirError> {
        if self.target_length == 0 {
            return Err(VirError::Config("target_length harus > 0".into()));
        }
        if self.num_levels == 0 {
            return Err(VirError::Config("num_levels harus > 0".into()));
        }
        if self.num_levels > self.target_length {
            return Err(VirError::Config(
                "num_levels tidak boleh melebihi target_length".into(),
            ));
        }
        if self.hidden_dim == 0 || self.hidden_dim % self.num_heads != 0 {
            return Err(VirError::Config(
                "hidden_dim harus habis dibagi num_heads".into(),
            ));
        }
        let head_dim = self.hidden_dim / self.num_heads;
        if self.strict_paper && head_dim != 64 {
            return Err(VirError::Config(format!(
                "strict_paper: head_dim harus 64, dapat {head_dim}"
            )));
        }
        let levels = make_schedule(self.schedule, self.target_length, self.num_levels);
        // naik ketat
        for w in levels.windows(2) {
            if !(w[0] < w[1]) {
                return Err(VirError::Config("jadwal level tidak naik ketat".into()));
            }
        }
        if *levels.last().unwrap_or(&0) != self.target_length {
            return Err(VirError::Config(
                "level terakhir harus = target_length".into(),
            ));
        }
        if self.codebook_size == 0 || self.latent_dim == 0 {
            return Err(VirError::Config(
                "codebook_size/latent_dim harus > 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.nested_dropout_prob) {
            return Err(VirError::Config("nested_dropout_prob di [0,1]".into()));
        }
        if self.ema_decay <= 0.0 || self.ema_decay >= 1.0 {
            return Err(VirError::Config("ema_decay di (0,1)".into()));
        }
        Ok(())
    }

    pub fn levels(&self) -> Vec<usize> {
        make_schedule(self.schedule, self.target_length, self.num_levels)
    }
}
