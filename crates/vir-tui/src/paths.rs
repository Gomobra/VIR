//! Path berkas yang dipantau TUI; dapat ditimpa lewat environment.

use crate::settings;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppPaths {
    pub metrics: String,
    pub logs: String,
    pub eval: String,
    pub rlm: String,
    pub generate: String,
    pub memory: String,
    pub settings: String,
}

impl Default for AppPaths {
    fn default() -> Self {
        Self {
            metrics: std::env::var("VIR_METRICS").unwrap_or_else(|_| "metrics.jsonl".into()),
            logs: std::env::var("VIR_LOG").unwrap_or_else(|_| "logs/vir.log".into()),
            eval: std::env::var("VIR_EVAL").unwrap_or_else(|_| "eval.jsonl".into()),
            rlm: std::env::var("VIR_RLM_EVENTS").unwrap_or_else(|_| "rlm-events.jsonl".into()),
            generate: std::env::var("VIR_GENERATE_EVENTS")
                .unwrap_or_else(|_| "generate-events.jsonl".into()),
            memory: std::env::var("VIR_MFS_PATH").unwrap_or_else(|_| "memori.db".into()),
            settings: settings::default_path().to_string(),
        }
    }
}
