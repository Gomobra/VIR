use clap::{Parser, Subcommand};
use vir_agent::{RlmConfig, ScriptedLm, run_rlm};
use vir_common::build_info;
use vir_data::synth_arithmetic;
use vir_mfs::MemoryStore;
use vir_text::BpeTokenizer;

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

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Smoke { json } => {
            let texts: Vec<String> = synth_arithmetic(50).into_iter().collect();
            let mut tok = BpeTokenizer::new();
            tok.train(&texts, 280);
            let sample = "1 + 2 = 3";
            let enc = tok.encode(sample);
            let dec = tok.decode(&enc);
            let roundtrip_ok = dec == sample;
            if json {
                println!(
                    "{{\"status\":\"{}\",\"vocab_size\":{},\"roundtrip_ok\":{},\"build\":\"{}\"}}",
                    if roundtrip_ok { "ok" } else { "fail" },
                    tok.vocab_size(),
                    roundtrip_ok,
                    build_info()
                );
            } else {
                println!(
                    "smoke: vocab_size={} roundtrip_ok={}",
                    tok.vocab_size(),
                    roundtrip_ok
                );
            }
            if !roundtrip_ok {
                std::process::exit(1);
            }
        }
        Commands::Tui => {
            if let Err(e) = vir_tui::run() {
                eprintln!("tui error: {e}");
                std::process::exit(1);
            }
        }
        Commands::Mem {
            path,
            write,
            read,
            find,
        } => {
            let store =
                MemoryStore::open(&path).unwrap_or_else(|_| MemoryStore::open_in_memory().unwrap());
            if let Some(content) = write {
                let vid = store.write("/cli", &content).unwrap();
                println!("wrote version {}", vid);
            }
            if let Some(vid) = read {
                println!("{}", store.read(vid).unwrap());
            }
            if let Some(q) = find {
                for (id, content, score) in store.find(&q, 5).unwrap() {
                    println!("{}: {} (score {:.4})", id, content, score);
                }
            }
        }
        Commands::Rlm { context } => {
            let lm = ScriptedLm::new(vec!["print('FINAL(ok)')".into()]);
            let cfg = RlmConfig {
                max_depth: 2,
                max_calls: 2,
            };
            let r = run_rlm(&lm, &context, &cfg).unwrap();
            println!("final: {:?}", r.final_answer);
            println!("steps: {}, calls: {}", r.steps, r.calls_used);
        }
        Commands::Version => println!("{}", build_info()),
    }
}
