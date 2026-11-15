use clap::{Parser, Subcommand};
use vir_agent::{RlmConfig, ScriptedLm, run_rlm};
use vir_common::build_info;
use vir_mfs::MemoryStore;

mod smoke;

#[derive(Parser)]
#[command(name = "vir", version, about = "VIR Vbeta CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Smoke {
        #[arg(long)]
        json: bool,
    },
    Tui,
    Mem {
        #[arg(long)]
        path: String,
        #[arg(long)]
        write: Option<String>,
        #[arg(long)]
        read: Option<i64>,
        #[arg(long)]
        find: Option<String>,
    },
    Rlm {
        #[arg(long)]
        context: String,
    },
    Version,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Smoke { json } => smoke::run(json)?,
        Commands::Tui => {
            vir_tui::run()?;
        }
        Commands::Mem {
            path,
            write,
            read,
            find,
        } => {
            let store = MemoryStore::open(&path)?;
            if let Some(content) = write {
                let vid = store.write("/cli", &content)?;
                println!("wrote version {}", vid);
            }
            if let Some(vid) = read {
                println!("{}", store.read(vid)?);
            }
            if let Some(q) = find {
                for (id, content, score) in store.find(&q, 5)? {
                    println!("{}: {} (score {:.4})", id, content, score);
                }
            }
        }
        Commands::Rlm { context } => {
            let lm = ScriptedLm::new(vec!["print('FINAL(ok)')".into()]);
            let cfg = RlmConfig {
                max_depth: 2,
                max_calls: 2,
                max_steps: 2,
            };
            let mut mem = |_path: &str,
                           _args: &serde_json::Value|
             -> Result<serde_json::Value, vir_common::VirError> {
                Ok(serde_json::Value::Null)
            };
            let r = run_rlm(&lm, &context, &cfg, &mut mem)?;
            println!("final: {:?}", r.final_answer);
            println!("steps: {}, calls: {}", r.steps, r.calls_used);
        }
        Commands::Version => println!("{}", build_info()),
    }
    Ok(())
}
