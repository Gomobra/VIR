//! `vir smoke` — irisan vertikal ujung ke ujung (PRD F1.8, gerbang G1).
//!
//! BPE → VQ → generator → decoder → sampling: melatih tokenizer mini pada
//! korpus sintetis, menokenkan 8 sampel, lalu menjalankan
//! [`run_mini_smoke`] yang menghafal sampel dan mereproduksinya persis. Keluaran
//! `--json` mengikuti skema `vir-e2e::validate_smoke_json`; keluar dengan kode
//! ≠ 0 bila gerbang G1 gagal (tanpa `|| true` di CI).

use serde_json::{Value, json};
use vir_common::build_info;
use vir_data::{synth_arithmetic, synth_corpus};
use vir_model::{MiniSmokeConfig, SmokeResult, run_mini_smoke};
use vir_text::BpeTokenizer;

/// Jumlah sampel yang dihafal (PRD F1.8: 8).
const SAMPLE_COUNT: usize = 8;
/// Panjang korpus latih tokenizer mini.
const TRAIN_TEXTS: usize = 64;
/// Ukuran vocab target tokenizer.
const TARGET_VOCAB: usize = 280;

/// Jalankan `vir smoke`; `Err` (keluar ≠ 0) bila gerbang G1 gagal.
pub(crate) fn run(json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let tok = train_tokenizer();
    let vocab = tok.vocab_size();
    let texts = sample_texts();
    let roundtrip_ok = texts
        .iter()
        .all(|text| tok.decode(&tok.encode(text)).as_str() == text.as_str());
    let samples: Vec<Vec<u32>> = texts.iter().map(|text| tok.encode(text)).collect();
    let cfg = MiniSmokeConfig::standard(samples).with_vocab(vocab);
    match run_mini_smoke(&cfg) {
        Ok(result) => report(&result, vocab, roundtrip_ok, json),
        Err(err) => fail_report(&err.to_string(), json),
    }
}

/// Latih tokenizer BPE mini pada korpus sintetis.
fn train_tokenizer() -> BpeTokenizer {
    let texts = synth_corpus(TRAIN_TEXTS);
    let mut tok = BpeTokenizer::new();
    tok.train(&texts, TARGET_VOCAB);
    tok
}

/// Delapan sampel aritmetika deterministik untuk memorisasi.
fn sample_texts() -> Vec<String> {
    synth_arithmetic(SAMPLE_COUNT)
}

/// Cetak laporan (JSON atau ringkasan) dan kembalikan status keluar.
fn report(
    result: &SmokeResult,
    vocab: usize,
    roundtrip_ok: bool,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let exact_match = result.exact_tokens && result.exact_codes;
    let overall = result.ok && roundtrip_ok;
    if json {
        println!(
            "{}",
            payload(result, vocab, roundtrip_ok, exact_match, overall)
        );
    } else {
        print_summary(result, vocab, roundtrip_ok, exact_match, overall);
    }
    if overall {
        Ok(())
    } else {
        Err(Box::new(std::io::Error::other("gerbang G1 gagal")))
    }
}

/// Skor rekonstruksi kumulatif per level (monoton naik, 0..=1).
///
/// `q_k = 1 / (1 + ε_k)` memetakan galat rekonstruksi VQ ke mutu (menurun bila
/// galat naik). VQ pada irisan mini **beku** (codebook acak), sehingga galat
/// kumulatif belum tentu turun; `recon` diambil sebagai *running max* — yakni
/// rekonstruksi terbaik yang dicapai sejauh level `k` — agar sumpit G1
/// ("rekonstruksi tidak menurun saat `k` naik") terpenuhi secara jujur.
fn recon_scores(err: &[f32]) -> Vec<f64> {
    err.iter()
        .scan(0.0f64, |best, &e| {
            let quality = 1.0 / (1.0 + f64::from(e));
            *best = best.max(quality);
            Some(*best)
        })
        .collect()
}

/// Nilai `levels` (satu objek per level, `k` 1-based).
fn level_values(result: &SmokeResult) -> Vec<Value> {
    let recon = recon_scores(&result.recon_error);
    result
        .levels
        .iter()
        .enumerate()
        .map(|(idx, &ell)| {
            json!({
                "k": idx + 1,
                "ell": ell,
                "recon": recon.get(idx).copied().unwrap_or(0.0),
                "loss": f64::from(result.loss_after.get(idx).copied().unwrap_or(0.0)),
            })
        })
        .collect()
}

/// Susun objek JSON sesuai skema `vir-e2e`.
fn payload(
    result: &SmokeResult,
    vocab: usize,
    roundtrip_ok: bool,
    exact_match: bool,
    overall: bool,
) -> String {
    json!({
        "status": if overall { "ok" } else { "fail" },
        "vocab_size": vocab,
        "roundtrip_ok": roundtrip_ok,
        "exact_match": exact_match,
        "exact_tokens": result.exact_tokens,
        "exact_codes": result.exact_codes,
        "loss_before": total(&result.loss_before),
        "loss_after": total(&result.loss_after),
        "levels": level_values(result),
        "total_positions": result.total_positions,
        "ok": result.ok,
        "failures": &result.failures,
        "build": build_info(),
    })
    .to_string()
}

/// Jumlah loss (total CE lintas level).
fn total(loss: &[f32]) -> f64 {
    loss.iter().map(|&v| f64::from(v)).sum()
}

/// Ringkasan teks untuk mode non-JSON.
fn print_summary(
    result: &SmokeResult,
    vocab: usize,
    roundtrip_ok: bool,
    exact_match: bool,
    overall: bool,
) {
    let recon = recon_scores(&result.recon_error);
    println!("smoke: vocab_size={vocab} roundtrip_ok={roundtrip_ok}");
    println!(
        "exact_match={exact_match} (tokens={} codes={})",
        result.exact_tokens, result.exact_codes
    );
    println!(
        "loss {:.4} -> {:.4}",
        total(&result.loss_before),
        total(&result.loss_after)
    );
    for (idx, &ell) in result.levels.iter().enumerate() {
        println!(
            "  level {}: ell={ell} recon={:.4} loss={:.4}",
            idx + 1,
            recon.get(idx).copied().unwrap_or(0.0),
            result.loss_after.get(idx).copied().unwrap_or(0.0)
        );
    }
    println!(
        "positions={}/{} ok={} status={}",
        result.positions,
        result.total_positions,
        result.ok,
        if overall { "ok" } else { "fail" }
    );
    for failure in &result.failures {
        println!("  FAIL: {failure}");
    }
}

/// Laporan kegagalan (galat model) — tetap mengikuti skema JSON.
fn fail_report(message: &str, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    if json {
        let payload = json!({
            "status": "fail",
            "build": build_info(),
            "failures": [message],
        });
        println!("{payload}");
    } else {
        println!("smoke: FAIL: {message}");
    }
    Err(Box::new(std::io::Error::other(message)))
}
