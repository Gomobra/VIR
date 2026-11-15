//! Gerbang G1 (PRD F1.8): irisan vertikal mini menghafal 8 sampel lalu
//! mereproduksinya persis. Tes ini menjalankan `run_mini_smoke` dengan benih
//! tetap dan memeriksa: loss turun per level, akurasi 1, Σℓ_k sesuai F1.1,
//! kode hasil generate == kode target, dan token hasil dekode == token masukan.

use vir_model::smoke_mini::{MiniSmokeConfig, run_mini_smoke, synth_samples};

#[test]
fn overfit_eight_samples_reproduce_exactly() {
    let samples = synth_samples(8, 8, 16, 42);
    let cfg = MiniSmokeConfig::standard(samples).with_vocab(16);
    let r = run_mini_smoke(&cfg).expect("irisan mini harus berjalan");

    assert!(r.ok, "gerbang G1 gagal: {:?}", r.failures);
    assert!(r.exact_codes, "kode generate != kode target");
    assert!(r.exact_tokens, "token dekode != token masukan");

    // Σℓ_k = 1 + 4 + 8 (jadwal linear L=8 K=3, round-half-even) dan posisi
    // generator sama.
    assert_eq!(r.total_positions, 13);
    assert_eq!(r.positions, r.total_positions);

    // Loss turun dan akurasi 1 di tiap level.
    for k in 0..r.levels.len() {
        assert!(
            r.loss_after[k] < r.loss_before[k],
            "loss level {k} tidak turun: {} → {}",
            r.loss_before[k],
            r.loss_after[k]
        );
        assert!(
            r.acc_after[k] > 0.999,
            "akurasi level {k} < 1: {}",
            r.acc_after[k]
        );
    }

    // Rekonstruksi kumulatif VQ dihitung dan dilaporkan (VQ mini beku, F1.4.b).
    assert!(r.recon_error.iter().all(|v| v.is_finite()));

    // Token hasil dekode sama persis dengan token masukan.
    let input: Vec<u32> = synth_samples(8, 8, 16, 42).into_iter().flatten().collect();
    assert_eq!(r.generated_tokens, input);
}
