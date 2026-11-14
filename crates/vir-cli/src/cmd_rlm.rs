//! `vir rlm` — RLM dengan LM berskrip (PRD F3.4).

use clap::Args;
use vir_agent::{RlmConfig, ScriptedLm, run_rlm};

use crate::CliError;

#[derive(Args)]
pub(crate) struct Args {
    /// Ringkasan tugas untuk root RLM
    #[arg(long)]
    pub context: String,
}

pub(crate) fn run(args: &Args) -> Result<(), CliError> {
    let lm = ScriptedLm::new(vec!["print('FINAL(ok)')".to_string()]);
    let cfg = RlmConfig {
        max_depth: 2,
        max_calls: 2,
        max_steps: 2,
    };
    let mut mem = |_path: &str,
                   _args: &serde_json::Value|
     -> Result<serde_json::Value, vir_common::VirError> { Ok(serde_json::Value::Null) };
    let out = run_rlm(&lm, &args.context, &cfg, &mut mem).map_err(CliError::other)?;
    match &out.final_answer {
        Some(answer) => println!("final: {answer}"),
        None => println!("final: (tidak ada jawaban)"),
    }
    println!("steps: {}, calls: {}", out.steps, out.calls_used);
    Ok(())
}