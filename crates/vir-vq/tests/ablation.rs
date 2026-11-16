//! F1.4.f — ablasi kuantizer (PRD F1.4.f, gerbang G1).
//!
//! Tiga kuantizer dilatih pada tokenizer mini yang identik (CPU, deterministik)
//! lalu dibandingkan pada tiga kriteria PRD:
//!
//! 1. **Akurasi rekonstruksi eksak** — `argmax` decoder == token masuk, per
//!    level (`acc@1..acc@K`), plus `mse` laten sebagai ukuran rate–distorsi yang
//!    sebanding antar kuantizer dengan jumlah bit berbeda.
//! 2. **Utilisasi / distribusi kode** — `util_K`, `ppl_K`, dan `dead_K` (kode
//!    mati menurut hitungan EMA; FSQ/iFSQ selalu 0 karena tanpa codebook).
//! 3. **Stabilitas latihan** — `loss0 → lossN` dan `nonfinite`.
//!
//! Arm: VQ-EMA dengan codebook seukuran data (32), VQ-EMA seukuran NCM (4096),
//! FSQ `[7,5,5,5,5]` (4375 kode), dan iFSQ `[16,16,16]` (4096 kode).
//! Tabel angka nyata dicetak `println!` (dan terlihat di log CI bila ada arm
//! yang gagal); ADR-002 memakainya untuk memilih kuantizer default Vbeta.

mod support;

use std::fs::OpenOptions;
use std::io::Write;

use support::ablation::{
    ArmReport, BATCH, DIM, FsqArm, LEN, SAMPLES, SCHEDULE, VOCAB, VqArm, run_arm,
};

/// Langkah latihan tiap arm (cukup untuk konvergensi pada data mini).
const STEPS: usize = 300;
/// Ambang akurasi rekonstruksi eksak pada level penuh. NCM Tabel 1 melaporkan
/// 98,79–100% pada skala penuh; mini diturunkan karena `d`, `K`, `L`, dan `|V|`
/// jauh lebih kecil (tugas lebih mudah, kapasitas jauh lebih kecil).
const MIN_ACCURACY: f64 = 0.9;
/// Selisih akurasi minimal level penuh terhadap level pertama (G1: "rekonstruksi
/// membaik per level").
const MIN_LEVEL_GAIN: f64 = 0.05;

#[test]
fn ablasi_vq_ema_vs_fsq_vs_ifsq() {
    // VQ-EMA: codebook seukuran data (32) dan seukuran NCM (4096).
    // FSQ `[7,5,5,5,5]` = 4375 kode (PRD F1.4.f langkah 2); iFSQ `[16,16,16]`
    // = 4096 kode, 4 bit per dimensi (langkah 3).
    let arms = vec![
        run_arm("vq-e32", STEPS, &VqArm(32)),
        run_arm("vq-e4096", STEPS, &VqArm(4096)),
        run_arm(
            "fsq-7555",
            STEPS,
            &FsqArm {
                levels: vec![7, 5, 5, 5, 5],
                ifsq: false,
            },
        ),
        run_arm(
            "ifsq-16",
            STEPS,
            &FsqArm {
                levels: vec![16, 16, 16],
                ifsq: true,
            },
        ),
    ];
    emit(&render(&arms));

    for arm in &arms {
        assert_arm(arm);
    }
    assert_usage(&arms);
}

/// Setiap arm wajib stabil dan merekonstruksi eksak (kriteria 1 dan 3).
fn assert_arm(arm: &ArmReport) {
    assert_eq!(
        arm.stab.non_finite, 0,
        "{} menghasilkan loss non-finite",
        arm.name
    );
    assert!(
        arm.stab.loss_last < arm.stab.loss_first,
        "{} loss tidak turun: {:.4} -> {:.4}",
        arm.name,
        arm.stab.loss_first,
        arm.stab.loss_last
    );
    let accuracy = arm.final_accuracy();
    assert!(
        accuracy >= MIN_ACCURACY,
        "{} akurasi level penuh {accuracy:.4} < {MIN_ACCURACY}",
        arm.name
    );
    let first = arm.accuracy.first().copied().unwrap_or(0.0);
    assert!(
        accuracy >= first + MIN_LEVEL_GAIN,
        "{} rekonstruksi tidak membaik per level: {first:.4} -> {accuracy:.4}",
        arm.name
    );
}

/// Kriteria 2: utilisasi terukur, dan kuantizer tanpa codebook bebas kode mati.
fn assert_usage(arms: &[ArmReport]) {
    for arm in arms {
        let (util, perplexity, dead) = arm.final_usage();
        assert!(
            (0.0..=1.0).contains(&util),
            "{} utilisasi {util} di luar [0,1]",
            arm.name
        );
        assert!(
            perplexity >= 1.0,
            "{} perplexity {perplexity} < 1",
            arm.name
        );
        let no_codebook = arm.name.starts_with("fsq") || arm.name.starts_with("ifsq");
        if no_codebook {
            assert_eq!(
                dead, 0,
                "{} melaporkan kode mati padahal tanpa codebook",
                arm.name
            );
        }
    }
}

/// Tabel ablasi: satu baris per arm (dipakai `println!` dan pesan assert).
fn render(arms: &[ArmReport]) -> String {
    let mut out = format!(
        "ablasi F1.4.f | L={LEN} K={} d={DIM} V={VOCAB} B={BATCH} S={SAMPLES}",
        SCHEDULE.len()
    );
    out.push_str(
        "\narm             |V|/level      loss0    lossN  nonfinite \
         mse@1    mse@2    mse@3    mse@4   acc@1   acc@2   acc@3   acc@4   \
         ppl_K  util_K  dead_K",
    );
    for arm in arms {
        let (util, ppl, dead) = arm.final_usage();
        let mses: Vec<String> = arm.mse.iter().map(|v| format!("{v:7.4}")).collect();
        let accs: Vec<String> = arm.accuracy.iter().map(|v| format!("{v:6.2}")).collect();
        out.push_str(&format!(
            "\n{:<16} {:<12} {:7.3} {:8.3} {:8}  {}  {}  {:6.1} {:6.3} {:6}",
            arm.name,
            codes_label(&arm.codes),
            arm.stab.loss_first,
            arm.stab.loss_last,
            arm.stab.non_finite,
            mses.join(" "),
            accs.join(" "),
            ppl,
            util,
            dead,
        ));
    }
    out
}

/// `|V_k|` per level sebagai satu label `a/b/c`.
fn codes_label(codes: &[usize]) -> String {
    codes
        .iter()
        .map(usize::to_string)
        .collect::<Vec<String>>()
        .join("/")
}

/// Cetak laporan ke stdout **dan** ke fd 2 proses.
///
/// `cargo test` menelan `println!` ketika tes lulus, sedangkan CI menjalankan
/// `cargo test` tanpa `--nocapture`; menulis langsung ke `/dev/stderr` membuat
/// angka ablasi tetap terlihat di log CI (bukti G1) tanpa membuat tes gagal.
fn emit(text: &str) {
    println!("{text}");
    let Ok(mut err) = OpenOptions::new().write(true).open("/dev/stderr") else {
        return;
    };
    if writeln!(err, "{text}").is_err() {
        eprintln!("ablation: laporan hanya tercetak di stdout");
    }
}
