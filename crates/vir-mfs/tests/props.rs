//! F3.3.a — tes properti: urutan operasi acak (write/read/forget/restore) tak
//! pernah mengubah isi mentah, versi lama tetap utuh, dan hash selalu cocok.

use std::collections::HashSet;

use proptest::prelude::*;
use vir_mfs::{MemoryStore, ReadTarget, forget, read, restore};

const PATHS: [&str; 3] = ["/episodik/a", "/semantik/b", "/kerja/c"];
const TEXTS: [&str; 4] = ["catatan satu", "catatan dua", "nasi goreng", "rust memori"];

#[derive(Debug, Clone, Copy)]
enum Op {
    Write(usize, usize),
    Read(usize),
    Forget(usize),
    Restore(usize),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    proptest::prop_oneof![
        (0usize..PATHS.len(), 0usize..TEXTS.len()).prop_map(|(p, t)| Op::Write(p, t)),
        (0usize..PATHS.len()).prop_map(Op::Read),
        (0usize..PATHS.len()).prop_map(Op::Forget),
        (0usize..PATHS.len()).prop_map(Op::Restore),
    ]
}

#[derive(Default)]
struct Model {
    versions: Vec<Vec<i64>>,
    forgotten: Vec<HashSet<i64>>,
}

impl Model {
    fn new() -> Self {
        Self {
            versions: vec![Vec::new(); PATHS.len()],
            forgotten: vec![HashSet::new(); PATHS.len()],
        }
    }

    fn push(&mut self, path: usize, vid: i64) {
        self.versions[path].push(vid);
    }

    fn latest(&self, path: usize) -> i64 {
        *self.versions[path].last().expect("minimal satu versi")
    }

    fn is_forgotten(&self, path: usize, vid: i64) -> bool {
        self.forgotten[path].contains(&vid)
    }
}

fn apply(store: &MemoryStore, ops: &[Op]) -> Model {
    let mut model = Model::new();
    for (i, path) in PATHS.iter().enumerate() {
        let vid = store.write(path, "awal").expect("tulis awal");
        model.push(i, vid);
    }
    for op in ops {
        match *op {
            Op::Write(p, t) => {
                let vid = store.write(PATHS[p], TEXTS[t]).expect("tulis");
                model.push(p, vid);
            }
            Op::Read(p) => check_read(store, &model, p),
            Op::Forget(p) => {
                let vid = forget(store, PATHS[p]).expect("lupa");
                model.forgotten[p].insert(vid);
            }
            Op::Restore(p) => {
                let vid = restore(store, PATHS[p]).expect("pulihkan");
                model.forgotten[p].remove(&vid);
            }
        }
    }
    model
}

fn check_read(store: &MemoryStore, model: &Model, p: usize) {
    let latest = model.latest(p);
    match read(store, ReadTarget::Path(PATHS[p])) {
        Ok(hit) => {
            assert_eq!(hit.version_id, latest, "baca mengembalikan versi terbaru");
            assert!(!model.is_forgotten(p, latest));
        }
        Err(_) => assert!(
            model.is_forgotten(p, latest),
            "hanya versi lupa yang gagal dibaca"
        ),
    }
}

fn assert_invariants(store: &MemoryStore, model: &Model) {
    for (p, path) in PATHS.iter().enumerate() {
        check_history(store, model, p, path);
        check_latest(store, model, p, path);
    }
}

fn check_history(store: &MemoryStore, model: &Model, p: usize, path: &str) {
    let ids = store.history(path).expect("riwayat");
    assert_eq!(ids.len(), model.versions[p].len(), "jumlah versi {path}");
    for vid in &model.versions[p] {
        assert!(
            store.verify(*vid).expect("verifikasi"),
            "hash versi {vid} cocok"
        );
        let content = store.read(*vid).expect("isi mentah");
        assert_eq!(
            store.hash_of(*vid).expect("hash"),
            vir_mfs::hex_sha256(content.as_bytes()),
            "hash versi {vid} sesuai isi"
        );
        assert_eq!(
            store.is_forgotten(*vid).expect("status"),
            model.is_forgotten(p, *vid)
        );
    }
}

fn check_latest(store: &MemoryStore, model: &Model, p: usize, path: &str) {
    let latest = model.latest(p);
    if model.is_forgotten(p, latest) {
        assert!(
            store.read_path(path).is_err(),
            "{path} yang dilupakan tak terbaca"
        );
        return;
    }
    assert_eq!(
        store.read_path(path).expect("baca terbaru"),
        store.read(latest).expect("isi versi"),
        "{path} terbaca sebagai versi terbaru"
    );
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(48))]

    #[test]
    fn invarian_terpenuhi_untuk_urutan_operasi_acak(
        ops in prop::collection::vec(op_strategy(), 0..24usize)
    ) {
        let store = MemoryStore::open_in_memory().expect("buka toko");
        let model = apply(&store, &ops);
        assert_invariants(&store, &model);
    }
}
