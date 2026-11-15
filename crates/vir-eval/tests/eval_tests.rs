//! Tes F4.5/F5.4 untuk `vir-eval`: estimator pass@k, rekonstruksi, metrik
//! generatif (PPL, entropi, ROUGE, BLEU), tabel metrik, dan panjang konteks
//! efektif. Semua nilai harapan dihitung dari definisi (PRD R15, F4.5, F5.4).

use vir_eval::{
    MetricTable, TokenDist, bleu, effective_context_length, fmt_num, generative_ppl, match_score,
    pass_at_k, pass_at_k_curve, recon_accuracy, rouge_l, rouge_n, unigram_entropy,
};

/// Bandingkan dua f64 dengan toleransi ketat.
fn approx(got: f64, want: f64) {
    assert!((got - want).abs() < 1e-9, "dapat {got}, harap {want}");
}

#[test]
fn pass_at_k_nilai_dasar() {
    approx(pass_at_k(10, 5, 1), 0.5); // 1 - C(5,1)/C(10,1)
    approx(pass_at_k(10, 0, 1), 0.0); // tanpa benar
    approx(pass_at_k(10, 10, 1), 1.0); // semua benar
    approx(pass_at_k(5, 3, 2), 0.9); // 1 - C(2,2)/C(5,2)
    approx(pass_at_k(4, 2, 0), 1.0); // k = 0
}

#[test]
fn recon_accuracy_akurat() {
    approx(recon_accuracy(&[1, 2, 3, 4], &[1, 2, 9, 4]), 0.75);
    approx(recon_accuracy(&[1, 2, 3], &[1, 9, 3]), 2.0 / 3.0);
    approx(recon_accuracy(&[], &[]), 0.0);
}

#[test]
fn unigram_entropi_dasar() {
    approx(unigram_entropy(&[]), 0.0);
    approx(unigram_entropy(&[1, 1, 1, 1]), 0.0);
    approx(unigram_entropy(&[1, 2]), 1.0);
    approx(unigram_entropy(&[1, 1, 2, 2]), 1.0);
}

#[test]
fn generative_ppl_dengan_referensi() {
    let dist = TokenDist::from_counts(&[(1, 1), (2, 1)]);
    approx(generative_ppl(&[1, 2], &dist), 2.0);
    approx(generative_ppl(&[], &dist), 1.0);
    assert!(generative_ppl(&[3], &dist).is_infinite());
}

#[test]
fn token_dist_prob_dan_entropi() {
    let dist = TokenDist::from_counts(&[(1, 1), (2, 1)]);
    approx(dist.prob(1), 0.5);
    approx(dist.entropy_bits(), 1.0);
    assert_eq!(dist.total(), 2);

    let smoothed = TokenDist::smoothed(&[(1, 1)], 1.0);
    approx(smoothed.prob(1), 2.0 / 3.0);
    approx(smoothed.prob(2), 1.0 / 3.0);
    approx(smoothed.prob(1) + smoothed.prob(2), 1.0);
}

#[test]
fn rouge_n_dan_rouge_l() {
    approx(rouge_n(&[1, 2, 3], &[1, 2, 3], 1), 1.0);
    approx(rouge_n(&[1, 2, 3, 4], &[1, 2], 1), 2.0 / 3.0);
    approx(rouge_n(&[1, 2, 3], &[1, 2, 3], 0), 0.0);

    approx(rouge_l(&[1, 2, 3], &[1, 2, 3]), 1.0);
    approx(rouge_l(&[1, 2, 3, 4], &[1, 3]), 2.0 / 3.0);
}

#[test]
fn bleu_dasar() {
    approx(bleu(&[1, 2, 3, 4], &[1, 2, 3, 4]), 1.0);
    approx(bleu(&[], &[1]), 0.0);
    approx(bleu(&[1, 2, 3, 4], &[9]), 0.0);
}

#[test]
fn match_score_multiset() {
    approx(match_score(&[1, 2, 3], &[1, 2, 3]), 1.0);
    approx(match_score(&[1, 2, 3], &[1, 9]), 1.0 / 3.0);
    approx(match_score(&[], &[]), 1.0);
    approx(match_score(&[], &[1]), 0.0);
    approx(match_score(&[1, 1, 2], &[1]), 1.0 / 3.0);
}

#[test]
fn effective_context_length_terbesar_yang_lulus() {
    let got = effective_context_length(&[4, 8, 16], &[0.9, 0.5, 0.95], 0.856);
    assert_eq!(got, Some(16));

    let none = effective_context_length(&[4, 8], &[0.1, 0.2], 0.856);
    assert_eq!(none, None);

    // Ambang bersifat ketat: skor sama dengan ambang tidak lolos.
    let strict = effective_context_length(&[10], &[0.856], 0.856);
    assert_eq!(strict, None);
}

#[test]
fn fmt_num_empat_desimal() {
    assert_eq!(fmt_num(0.5), "0.5000");
    assert_eq!(fmt_num(1.23456), "1.2346");
}

#[test]
fn pass_at_k_curve_sesuai_k() {
    let curve = pass_at_k_curve(10, 5, &[1, 2]);
    assert_eq!(curve.len(), 2);
    assert_eq!(curve[0].0, 1);
    approx(curve[0].1, 0.5);
    assert_eq!(curve[1].0, 2);
}

#[test]
fn metric_table_render_markdown() {
    let mut table = MetricTable::new(&["A", "B"]);
    table.push_row(&["1".to_string()]);
    assert_eq!(table.len(), 1);
    assert!(!table.is_empty());
    let lines: Vec<&str> = table.render().lines().collect();
    assert_eq!(lines[0], "| A | B |");
    assert_eq!(lines[1], "| --- | --- |");
    assert_eq!(lines[2], "| 1 |  |");
}
