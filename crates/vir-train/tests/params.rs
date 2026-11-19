//! Tes `two_d_param_paths` (F2.5): Muon hanya pada bobot blok transformer.

use vir_train::model::{VirTrainConfig, VirTrainModel, train_device};
use vir_train::params::two_d_param_paths;

fn cfg() -> VirTrainConfig {
    VirTrainConfig {
        dim: 16,
        num_heads: 2,
        levels: vec![2, 4],
        codebook_size: 32,
        vocab: 4,
        phi_kernel: 1,
        prelude_blocks: 1,
        core_blocks: 2,
        coda_blocks: 1,
        num_loops: 1,
        decoder_hidden: 16,
        commitment_cost: 0.25,
        alpha: 0.0,
        chunk: 0,
        seed: 5,
    }
}

#[test]
fn two_d_paths_hanya_bobot_blok_dan_tidak_kosong() {
    let device = train_device(Default::default(), 5);
    let model = VirTrainModel::new(&cfg(), &device).expect("model");
    let paths = two_d_param_paths(&model);
    assert!(
        !paths.is_empty(),
        "Muon harus punya param (blok transformer)"
    );
    for p in &paths {
        assert!(p.ends_with(".weight"), "hanya bobot: {p}");
        let is_block = p.contains("prelude")
            || p.contains("core")
            || p.contains("coda")
            || p.contains(".wqkv.")
            || p.contains(".wo.")
            || p.contains(".w_gate_up.")
            || p.contains(".w_down.");
        assert!(is_block, "harus bobot blok, dapat: {p}");
        // Head/embedding/codebook tidak boleh masuk Muon.
        assert!(!p.contains("head"), "head bukan blok: {p}");
        assert!(!p.starts_with("embed"), "embed bukan blok: {p}");
    }
}
