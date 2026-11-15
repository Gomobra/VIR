//! Tes nyata AdamW, clipping, klasifikasi grup, dan state momentum 8-bit
//! (PRD F1.7 langkah 4–5).

use vir_train::optim::AdamParams;
use vir_train::{
    AdamState, AdamW, Hybrid, MomState, OptGroup, OptimSettings, ParamRole, ParamSlot, ParamStore,
    classify, clip_grad_norm,
};

/// Store dengan satu slot pada indeks 0.
fn store_with(slot: ParamSlot) -> ParamStore {
    let mut store = ParamStore::new();
    store.add(slot);
    store
}

/// `½ Σ (w − t)²` untuk satu slot.
fn quad_loss(data: &[f32], target: &[f32]) -> f32 {
    data.iter()
        .zip(target.iter())
        .map(|(w, t)| 0.5 * (w - t) * (w - t))
        .sum()
}

#[test]
fn adamw_reduces_quadratic_loss() {
    let target = [1.0f32, -1.0, 0.5];
    let slot = ParamSlot::vector("norm.weight", ParamRole::Generator, vec![5.0, -3.0, 2.0]);
    let mut store = store_with(slot);
    let mut opt = AdamW::from_store(&store, &AdamParams::default());

    let first = quad_loss(&store.slots[0].data, &target);
    for _ in 0..300 {
        let g: Vec<f32> = store.slots[0]
            .data
            .iter()
            .zip(target.iter())
            .map(|(w, t)| w - t)
            .collect();
        store.set_grad(0, &g);
        assert!(opt.step(0, &mut store.slots[0], 0.05, 0.0));
    }
    let last = quad_loss(&store.slots[0].data, &target);
    assert!(first > 1.0, "loss awal {first} terlalu kecil untuk diuji");
    assert!(last < 1e-4, "loss akhir {last} tidak cukup kecil");
    assert!(last < first * 1e-3, "loss tidak turun cukup jauh");
}

#[test]
fn adamw_zero_grad_is_a_no_op() {
    let slot = ParamSlot::vector("norm.bias", ParamRole::Generator, vec![1.0, 2.0]);
    let mut store = store_with(slot);
    let mut opt = AdamW::from_store(&store, &AdamParams::default());
    let before = store.slots[0].data.clone();
    store.set_grad(0, &[0.0, 0.0]);
    assert!(opt.step(0, &mut store.slots[0], 0.1, 0.0));
    assert_eq!(store.slots[0].data, before, "gradien nol mengubah bobot");
}

#[test]
fn adamw_state_tracks_steps_and_bias() {
    let mut state = AdamState::zeros(2);
    assert_eq!(state.t, 0);
    state.m = vec![1.0, -1.0];
    state.v = vec![0.25, 0.25];
    assert_eq!(state.m.len(), 2);
}

#[test]
fn classify_routes_matrices_to_muon_and_rest_to_adamw() {
    assert_eq!(classify("block.q.weight", 768, 768), OptGroup::Muon);
    assert_eq!(classify("block.ffn.weight", 3072, 768), OptGroup::Muon);
    assert_eq!(classify("embed.token.weight", 4096, 384), OptGroup::AdamW);
    assert_eq!(classify("head.weight", 4096, 768), OptGroup::AdamW);
    assert_eq!(classify("norm.weight", 768, 1), OptGroup::AdamW);
    assert_eq!(classify("core.q.weight", 768, 1), OptGroup::AdamW);
}

#[test]
fn hybrid_updates_only_unfrozen_slots() {
    let mut store = ParamStore::new();
    store.add(ParamSlot::new(
        "core.q.weight",
        4,
        4,
        ParamRole::Vq,
        vec![0.0; 16],
    ));
    store.add(ParamSlot::vector(
        "norm.weight",
        ParamRole::Vq,
        vec![1.0, 1.0],
    ));
    assert_eq!(store.slots[0].group, OptGroup::Muon);
    assert_eq!(store.slots[1].group, OptGroup::AdamW);
    store.set_grad(0, &[0.01; 16]);
    store.set_grad(1, &[1.0, -1.0]);

    let mut opt = Hybrid::new(&store, &OptimSettings::default());
    assert_eq!(opt.step_single(&mut store, 1e-2), 2);
    assert!(store.slots[1].data[0] < 1.0 && store.slots[1].data[1] > 1.0);

    let frozen = store.freeze_role(ParamRole::Vq);
    assert_eq!(frozen, 2, "freeze_role harus membekukan dua slot");
    store.set_grad(0, &[0.01; 16]);
    store.set_grad(1, &[1.0, -1.0]);
    assert_eq!(opt.step_single(&mut store, 1e-2), 0, "slot beku diupdate");
}

#[test]
fn clip_grad_norm_scales_global_norm() {
    let mut store = ParamStore::new();
    store.add(ParamSlot::vector(
        "norm.weight",
        ParamRole::Generator,
        vec![0.0, 0.0],
    ));
    store.add(ParamSlot::vector(
        "norm.bias",
        ParamRole::Generator,
        vec![0.0, 0.0],
    ));
    store.set_grad(0, &[3.0, 0.0]);
    store.set_grad(1, &[0.0, 4.0]);

    let norm = clip_grad_norm(&mut store, 2.0);
    assert!((norm - 5.0).abs() < 1e-5, "norma awal {norm} != 5");
    let after = store.grad_norm();
    assert!(
        (after - 2.0).abs() < 1e-5,
        "norma setelah clip {after} != 2"
    );
    assert!((store.slots[0].grad[0] - 1.2).abs() < 1e-5);
    assert!((store.slots[1].grad[1] - 1.6).abs() < 1e-5);

    let small = clip_grad_norm(&mut store, 10.0);
    assert!((small - 2.0).abs() < 1e-5, "clip longgar mengubah gradien");
}

#[test]
fn mom_state_8bit_quantizes_within_half_step() {
    let mut st = MomState::zeros(4, 8);
    st.store(&[0.0, 127.0, -63.5, 0.25]);
    assert!((st.value(1) - 127.0).abs() <= 0.51, "nilai 127 tidak pulih");
    assert!(
        (st.value(2) + 63.5).abs() <= 0.51,
        "nilai -63,5 tidak pulih"
    );

    let mut buf = Vec::new();
    st.encode(&mut buf);
    // `decode` menerima buffer setelah tag (pos = 1), sama seperti `restore`.
    let decoded = MomState::decode(1, 4, &buf, &mut 1).expect("decode state 8-bit");
    assert_eq!(decoded.len(), 4);
    assert!((decoded.value(1) - st.value(1)).abs() < 1e-6);
}

#[test]
fn mom_state_f32_keeps_exact_values() {
    let mut st = MomState::zeros(3, 32);
    st.store(&[0.5, -0.25, 1.0]);
    assert!((st.value(0) - 0.5).abs() < 1e-6);
    assert!((st.value(2) - 1.0).abs() < 1e-6);
}
