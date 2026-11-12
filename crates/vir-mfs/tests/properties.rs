//! Tes properti dan recall untuk `vir-mfs` (PRD F3.3.a/b).

use proptest::prelude::*;
use vir_mfs::{
    Embedder, HashNgramEmbedder, LinkKind, MemoryStore, SearchMode, SearchOptions, hex_sha256,
};

fn path_strategy() -> impl Strategy<Value = String> {
    ("[a-z]{1,6}", 0usize..3).prop_map(|(name, depth)| {
        let mut path = format!("/semantik/{name}");
        for i in 0..depth {
            path.push_str(&format!("/d{i}"));
        }
        path
    })
}

fn text_strategy() -> impl Strategy<Value = String> {
    prop::collection::vec(prop::char::range('a', 'z'), 1..48)
        .prop_map(|chars| chars.into_iter().collect())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Membaca tidak mengubah isi, versi, atau hash; hanya menambah log akses.
    #[test]
    fn read_tidak_mengubah_isi(content in text_strategy(), path in path_strategy()) {
        let store = MemoryStore::open_in_memory().unwrap();
        let vid = store.write(&path, &content).unwrap();
        let versions_before = store.history(&path).unwrap().len();
        let hash_before = store.hash_of(vid).unwrap();
        let first = store.read(vid).unwrap();
        let second = store.read(vid).unwrap();
        prop_assert_eq!(first, content.clone());
        prop_assert_eq!(second, content);
        prop_assert_eq!(store.hash_of(vid).unwrap(), hash_before);
        prop_assert_eq!(store.history(&path).unwrap().len(), versions_before);
        prop_assert!(store.verify(vid).unwrap());
        prop_assert!(store.access_count(vid).unwrap() >= 2);
    }

    /// Setiap versi lama tetap utuh setelah banyak tulisan baru.
    #[test]
    fn versi_lama_utuh(
        contents in prop::collection::vec(text_strategy(), 1..8),
        path in path_strategy(),
    ) {
        let store = MemoryStore::open_in_memory().unwrap();
        let mut written: Vec<(i64, String)> = Vec::new();
        for content in &contents {
            let vid = store.write(&path, content).unwrap();
            written.push((vid, content.clone()));
        }
        let history = store.history(&path).unwrap();
        prop_assert_eq!(history.len(), contents.len());
        for (vid, content) in &written {
            prop_assert_eq!(&store.read(*vid).unwrap(), content);
            prop_assert!(store.verify(*vid).unwrap());
            prop_assert_eq!(&store.hash_of(*vid).unwrap(), &hex_sha256(content.as_bytes()));
        }
        let info = store.versions(&path).unwrap();
        for (i, v) in info.iter().enumerate() {
            prop_assert_eq!(v.supersedes.is_some(), i > 0);
        }
    }

    /// Urutan acak write/forget/restore/tag tidak merusak hash atau isi.
    #[test]
    fn hash_cocok_setelah_operasi_acak(
        paths in prop::collection::vec(path_strategy(), 1..4),
        contents in prop::collection::vec(text_strategy(), 1..10),
        flags in prop::collection::vec(any::<bool>(), 1..10),
        tag_indices in prop::collection::vec(0usize..8, 0..4),
    ) {
        let store = MemoryStore::open_in_memory().unwrap();
        let mut written: Vec<(i64, String)> = Vec::new();
        for (path, content) in paths.iter().zip(contents.iter()) {
            let vid = store.write(path, content).unwrap();
            written.push((vid, content.clone()));
        }
        for ((vid, _), forget) in written.iter().zip(flags.iter()) {
            if *forget {
                store.forget(*vid).unwrap();
            } else {
                store.restore(*vid).unwrap();
            }
        }
        for (n, idx) in tag_indices.iter().enumerate() {
            if let Some((vid, _)) = written.get(*idx) {
                let path = store.version_info(*vid).unwrap().path;
                let fid = store.file_id(&path).unwrap().unwrap();
                store.tag(fid, &format!("tag{n}")).unwrap();
            }
        }
        for (vid, content) in &written {
            prop_assert_eq!(&store.read(*vid).unwrap(), content);
            prop_assert_eq!(&store.hash_of(*vid).unwrap(), &hex_sha256(content.as_bytes()));
        }
        let count = store.entries().unwrap().len();
        prop_assert!(count <= paths.len());
    }
}

/// Recall@k: 1 bila target ada di hasil, 0 bila tidak.
fn hit_at(hits: &[vir_mfs::SearchHit], target: i64) -> f64 {
    if hits.iter().any(|h| h.version_id == target) {
        1.0
    } else {
        0.0
    }
}

#[test]
fn recall_hibrida_tidak_kalah_dari_mode_tunggal() {
    let topics = ["alpha", "beta", "gamma", "delta", "epsilon"];
    let store = MemoryStore::open_in_memory().unwrap();
    let mut relevant = Vec::new();
    for (i, topic) in topics.iter().enumerate() {
        let path = format!("/episodik/e{i}");
        let text = format!("catatan {topic} tentang memori dan sistem {i}");
        relevant.push(store.write(&path, &text).unwrap());
    }
    for i in 0..15 {
        let text = format!("bising {i} tanpa topik khusus");
        store.write(&format!("/episodik/n{i}"), &text).unwrap();
    }
    store.ensure_vectors().unwrap();
    let k = 3;
    for (topic, target) in topics.iter().zip(relevant.iter()) {
        let keyword = store.search_fts(topic, k).unwrap();
        let vector = store.search_vector(topic, k).unwrap();
        let opts = SearchOptions {
            query: (*topic).to_string(),
            limit: k,
            mode: SearchMode::Hybrid,
            ..SearchOptions::default()
        };
        let hybrid = store.search(&opts).unwrap();
        let (rk, rv, rh) = (
            hit_at(&keyword, *target),
            hit_at(&vector, *target),
            hit_at(&hybrid, *target),
        );
        assert!(rh > 0.0, "hibrida tidak menemukan {topic}");
        assert!(rh >= rk, "hibrida {rh} < kata kunci {rk} untuk {topic}");
        assert!(rh >= rv, "hibrida {rh} < vektor {rv} untuk {topic}");
        assert!(hybrid[0].fts_rank.is_some() || hybrid[0].vector_rank.is_some());
    }
}

#[test]
fn duplikat_ditautkan_bukan_dihapus() {
    let store = MemoryStore::open_in_memory().unwrap();
    let a = store
        .write("/episodik/a", "memori episodik yang sama")
        .unwrap();
    let b = store
        .write("/episodik/b", "memori episodik yang sama")
        .unwrap();
    let links = store.links_from(b).unwrap();
    assert!(links.contains(&(a, LinkKind::DuplicateOf)));
    assert_eq!(store.read(a).unwrap(), "memori episodik yang sama");
    assert_eq!(store.read(b).unwrap(), "memori episodik yang sama");
    assert_eq!(store.entries().unwrap().len(), 2);
}

#[test]
fn konsolidasi_tidak_mengubah_checksum_sumber() {
    let store = MemoryStore::open_in_memory().unwrap();
    let source = store
        .write("/episodik/e1", "baris pertama\nbaris kedua\nbaris ketiga")
        .unwrap();
    let before = store.hash_of(source).unwrap();
    let before_count = store.entries().unwrap().len();
    let report = store
        .consolidate(
            source,
            "/semantik/ringkas",
            &vir_mfs::HeadlineSummarizer::default(),
        )
        .unwrap();
    assert_eq!(store.hash_of(source).unwrap(), before);
    assert_eq!(store.version_info(source).unwrap().source, None);
    assert_eq!(
        store.version_info(report.version_id).unwrap().source,
        Some(source)
    );
    assert_eq!(store.entries().unwrap().len(), before_count + 1);
    assert!(
        store
            .links_from(report.version_id)
            .unwrap()
            .contains(&(source, LinkKind::DerivedFrom))
    );
    assert_eq!(
        store.read(report.version_id).unwrap(),
        "baris pertama baris kedua baris ketiga"
    );
}

#[test]
fn embedder_bawaan_deterministik_dan_terindeks() {
    let store = MemoryStore::open_in_memory().unwrap();
    let vid = store.write("/semantik/x", "memori deterministik").unwrap();
    assert!(store.vector_of(vid).unwrap().is_some());
    assert_eq!(
        HashNgramEmbedder::default().embed("memori deterministik"),
        HashNgramEmbedder::default().embed("memori deterministik")
    );
    assert_eq!(store.reindex().unwrap(), 1);
}
