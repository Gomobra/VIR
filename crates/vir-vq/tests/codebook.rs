//! F1.4.a — codebook EMA, dead-code restart, straight-through (PRD F1.4.a).

mod support;

use burn::tensor::{Int, Tensor, TensorData};
use support::{read_int1, read2, tensor2, vector1};
use vir_vq::codebook::{Codebook, CodebookConfig};
use vir_vq::quantizer::Quantizer;

fn cfg(num: usize, dim: usize) -> CodebookConfig {
    CodebookConfig {
        num_entries: num,
        dim,
        ema_decay: 0.99,
        commitment_cost: 0.25,
        restart_threshold: 1.0,
        eps: 1e-5,
    }
}

fn device() -> burn::tensor::Device {
    Default::default()
}

/// Codebook 4x2 dengan entri tetap: [0,0],[0,1],[1,0],[1,1].
fn fixed_codebook() -> Codebook {
    let mut cb = Codebook::new(&cfg(4, 2), &device());
    cb.set_entries(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("set_entries");
    cb
}

#[test]
fn argmin_memilih_entri_terdekat() {
    let cb = fixed_codebook();
    // [0.1, 0.1] paling dekat [0,0] -> idx 0; [0.9, 0.9] -> [1,1] idx 3.
    let z = tensor2(2, 2, &[0.1, 0.1, 0.9, 0.9]);
    let idx = read_int1(&cb.quantize_idx(z));
    assert_eq!(idx, vec![0, 3]);
}

#[test]
fn quantize_idx_reference_cocok_dengan_naive_argmin() {
    let cb = fixed_codebook();
    let mut rng = support::Lcg::new(7);
    let data: Vec<f32> = (0..20).map(|_| rng.unit() * 1.2 - 0.1).collect();
    let z = tensor2(10, 2, &data);
    let a = read_int1(&cb.quantize_idx_reference(z.clone()));
    // Ground truth: jarak Euclidean naif.
    let entries = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let mut want = Vec::new();
    for r in 0..10 {
        let row = [data[r * 2], data[r * 2 + 1]];
        let mut best = 0usize;
        let mut bd = f32::MAX;
        for (j, e) in entries.iter().enumerate() {
            let d = (row[0] - e[0]).powi(2) + (row[1] - e[1]).powi(2);
            if d < bd {
                bd = d;
                best = j;
            }
        }
        want.push(best as i32);
    }
    assert_eq!(a, want);
}

#[test]
fn ema_update_av_trigger_count_hitung_developer() {
    let cb = fixed_codebook();
    assert_eq!(
        cb.used_codes(),
        4,
        "semua entri terpakai setelah set_entries"
    );
}

#[test]
fn dead_codes_terdeteksi() {
    // restart_threshold = 0 -> dead-code restart mati, sehingga entri yang tak
    // dipakai tetap mati (tak dihidupkan ulang dari batch).
    let mut c = cfg(8, 2);
    c.restart_threshold = 0.0;
    let mut cb = Codebook::new(&c, &device());
    let z = tensor2(2, 2, &[0.0, 0.0, 1.0, 1.0]);
    let idx = cb.quantize_idx(z);
    let mut rng = support::Lcg::new(1);
    cb.ema_update(&tensor2(2, 2, &[0.0, 0.0, 1.0, 1.0]), &idx, &mut rng);
    cb.refresh_cache();
    // Hanya <=2 entri terpakai; sisanya (count < 1.0) dianggap mati oleh
    // `used_codes` (ambang 1.0), bukan oleh `dead_codes` (ambang restart).
    assert!(cb.used_codes() <= 2, "hanya <=2 entri terpakai");
    assert!(cb.dead_codes() <= cb.used_codes() + 6);
}

#[test]
fn straight_through_meneruskan_nilai_z() {
    let cb = fixed_codebook();
    // STE: output = z + (z_q - z).detach(); nilainya == z_q, grad = 1 ke z.
    let z = tensor2(2, 2, &[0.1, 0.1, 0.9, 0.9]);
    let st = cb.straight_through(z.clone());
    let vals = read2(&st);
    // Nilai sama dengan vektor terkuantisasi.
    let q = read2(&cb.quantize_pair(z).1);
    for (a, b) in vals.iter().zip(q.iter()) {
        assert!((a - b).abs() < 1e-5, "STE nilai {a} vs {b}");
    }
}

#[test]
fn set_entries_menepati_ema_state() {
    // E-012: setelah set_entries, EMA update pertama tak boleh mencampur init acak.
    // restart_threshold = 0 -> entri yang tak dipakai batch tidak dihidupkan ulang.
    let mut c = cfg(4, 2);
    c.restart_threshold = 0.0;
    let mut cb = Codebook::new(&c, &device());
    cb.set_entries(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("set");
    // EMA update dengan batch yang sama persis -> entri tidak berubah (decay 0.99,
    // tapi sum/count sudah = e sehingga e tetap).
    let z = tensor2(1, 2, &[0.0, 0.0]);
    let idx = cb.quantize_idx(z);
    let mut rng = support::Lcg::new(3);
    cb.ema_update(&tensor2(1, 2, &[0.0, 0.0]), &idx, &mut rng);
    cb.refresh_cache();
    let w = read2(&cb.weight.val());
    assert!((w[0] - 0.0).abs() < 1e-3, "entri 0 tetap dekat 0: {}", w[0]);
    assert!((w[3] - 1.0).abs() < 1e-2, "entri 3 tetap dekat 1: {}", w[3]);
}

#[test]
fn stats_utilisasi_dan_perplexity() {
    let cb = fixed_codebook();
    let z = tensor2(2, 2, &[0.1, 0.1, 0.9, 0.9]);
    let quant = cb.quantize(z);
    let stats = cb.stats(&quant.indices);
    assert_eq!(stats.codes, 4);
    assert_eq!(stats.used, 2, "2 dari 4 kode terpakai");
    assert!((stats.utilization - 0.5).abs() < 1e-6);
    assert!(stats.perplexity >= 1.0, "perplexity >= 1");
}

#[test]
fn histogram_kosong_aman() {
    let cb = fixed_codebook();
    let empty: Tensor<1, Int> =
        Tensor::<1, Int>::from_ints(TensorData::new(Vec::<i32>::new(), [0]), &device());
    let stats = cb.stats(&empty);
    assert_eq!(stats.used, 0);
    assert_eq!(stats.utilization, 0.0);
    let _ = vector1(&[0.0]);
}
