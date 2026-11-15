//! `vir rlm` — RLM dengan LM nyata dan MFS nyata (PRD F3.4, F3.6).
//!
//! Root Vbeta (ADR-005) adalah proses eksternal yang benar-benar berjalan:
//! `--model <program> [args...]` menerima prompt lewat stdin dan mencetak kode
//! Python ke stdout, atau `--model <berkas.py>` yang dijalankan `python3`.
//! Tanpa `--model`, perintah gagal dengan galat yang jelas — tidak ada jawaban
//! karangan. `--scripted` hanya untuk demonstrasi dan selalu ditandai.

use std::path::{Path, PathBuf};

use clap::Args;
use serde_json::Value as JsonValue;
use vir_agent::{CommandLm, LanguageModel, MemBridge, RlmConfig, RlmRun, ScriptedLm, run_rlm};
use vir_common::VirError;

use crate::CliError;

/// Interpreter untuk skrip model `.py`.
const DEFAULT_PYTHON: &str = "python3";
/// Jembatan memori; `None` bila `--mem` tidak diberikan.
type Bridge = Option<MemBridge>;

#[derive(Args)]
pub(crate) struct Args {
    /// Ringkasan tugas untuk root RLM
    #[arg(long)]
    pub context: String,
    /// Root LM: `program [args...]` atau berkas `.py` (dijalankan `python3`)
    #[arg(long = "model", value_name = "PATH|COMMAND")]
    pub model: Option<String>,
    /// Argumen tambahan untuk perintah model
    #[arg(long = "model-arg", value_name = "ARG")]
    pub model_args: Vec<String>,
    /// Basis data MFS yang dibuka untuk operasi `mem`
    #[arg(long, value_name = "PATH")]
    pub mem: Option<String>,
    /// Muat konteks panjang dari berkas sebagai variabel REPL
    #[arg(long, value_name = "PATH")]
    pub context_file: Option<PathBuf>,
    /// Batas kedalaman `llm_query`
    #[arg(long, default_value_t = 3)]
    pub max_depth: usize,
    /// Batas jumlah panggilan model
    #[arg(long, default_value_t = 8)]
    pub max_calls: usize,
    /// Batas jumlah langkah root
    #[arg(long, default_value_t = 8)]
    pub max_steps: usize,
    /// Pakai LM berskrip (demonstrasi; keluaran ditandai "scripted")
    #[arg(long)]
    pub scripted: bool,
    /// Cetak jejak sebagai JSON
    #[arg(long)]
    pub json: bool,
}

pub(crate) fn run(args: &Args) -> Result<(), CliError> {
    let lm = build_lm(args)?;
    let bridge = open_bridge(args.mem.as_deref())?;
    let context = load_context(args)?;
    let cfg = RlmConfig {
        max_depth: args.max_depth,
        max_calls: args.max_calls,
        max_steps: args.max_steps,
    };
    let out = run_rlm(&lm, &context, &cfg, &mut |payload: &JsonValue| {
        dispatch(&bridge, payload)
    })?;
    report(&out, &lm, &bridge, args.json);
    Ok(())
}

/// Pilih LM sesuai argumen; tanpa `--model` → galat eksplisit (bukan karangan).
fn build_lm(args: &Args) -> Result<Box<dyn LanguageModel>, CliError> {
    if args.scripted {
        let lm = ScriptedLm::new(vec!["print('FINAL(scripted)')".to_string()]);
        return Ok(Box::new(lm));
    }
    let Some(spec) = args.model.clone() else {
        return Err(CliError::other(
            "root model belum dipilih: pakai --model <program|berkas.py> \
             (ADR-005: root Vbeta adalah callable eksternal, bukan model VIR)",
        ));
    };
    if spec.ends_with(".py") {
        let lm = CommandLm::python(Path::new(&spec), DEFAULT_PYTHON)?;
        return Ok(Box::new(lm));
    }
    Ok(Box::new(CommandLm::new(spec, args.model_args.clone())))
}

/// Buka MFS bila `--mem` diberikan.
fn open_bridge(path: Option<&str>) -> Result<Bridge, CliError> {
    match path {
        Some(path) => Ok(Some(MemBridge::open(Path::new(path))?)),
        None => Ok(None),
    }
}

/// Teruskan payload `mem` ke MFS; tanpa `--mem` menjadi galat yang jelas.
fn dispatch(bridge: &Bridge, payload: &JsonValue) -> Result<JsonValue, VirError> {
    match bridge {
        Some(store) => store.call(payload),
        None => Err(VirError::Config(
            "operasi `mem` butuh --mem <path>: tidak ada memori host".into(),
        )),
    }
}

/// Susun konteks: `--context` plus isi `--context-file` bila ada.
fn load_context(args: &Args) -> Result<String, CliError> {
    let Some(path) = &args.context_file else {
        return Ok(args.context.clone());
    };
    let text = std::fs::read_to_string(path).map_err(CliError::other)?;
    Ok(format!(
        "{}\n\n--- {} ---\n{text}",
        args.context,
        path.display()
    ))
}

/// Cetak jejak langkah sesuai format keluaran.
fn report(out: &RlmRun, lm: &dyn LanguageModel, bridge: &Bridge, json: bool) {
    let scripted = lm.is_scripted();
    let mem = match bridge {
        Some(store) => store.origin().display().to_string(),
        None => "(tidak ada)".to_string(),
    };
    if json {
        println!(
            "{}",
            serde_json::json!({
                "final": out.final_answer,
                "steps": out.steps,
                "calls_used": out.calls_used,
                "recovered": out.recovered,
                "mem": mem,
                "scripted": scripted,
            })
        );
        return;
    }
    match &out.final_answer {
        Some(answer) => println!("final: {answer}"),
        None => println!("final: (tidak ada jawaban)"),
    }
    println!(
        "steps: {}, calls: {}, recovered: {}",
        out.steps, out.calls_used, out.recovered
    );
    println!("mem: {mem}");
    if scripted {
        println!("catatan: LM berskrip (--scripted) — bukan model nyata");
    }
}
