use clap::{Parser, Subcommand};
use vir_common::build_info;
use vir_data::synth_arithmetic;
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
        Commands::Version => println!("{}", build_info()),
    }
}
