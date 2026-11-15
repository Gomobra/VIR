//! F3.3.b — filter path/tag/waktu dan hasil pencarian hibrida berskor.

use std::time::{SystemTime, UNIX_EPOCH};

use vir_mfs::{MemoryStore, SearchHit, SearchMode, SearchOptions};

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("jam sistem")
        .as_secs() as i64
}

fn store() -> MemoryStore {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    store
        .write("/episodik/makan", "nasi goreng enak")
        .expect("tulis 1");
    store
        .write("/semantik/resep", "resep nasi goreng lengkap")
        .expect("tulis 2");
    store
}

fn opts(query: &str) -> SearchOptions {
    SearchOptions {
        query: query.to_string(),
        ..SearchOptions::default()
    }
}

fn paths(hits: &[SearchHit]) -> Vec<&str> {
    hits.iter().map(|h| h.path.as_str()).collect()
}

#[test]
fn filter_path_prefix() {
    let store = store();
    let mut o = opts("nasi goreng");
    o.path_prefix = Some("/episodik".to_string());
    let hits = store.search(&o).expect("cari");
    assert_eq!(paths(&hits), vec!["/episodik/makan"]);
    o.path_prefix = Some("/semantik".to_string());
    assert_eq!(
        paths(&store.search(&o).expect("cari")),
        vec!["/semantik/resep"]
    );
    o.path_prefix = Some("/episodik/makan".to_string());
    assert_eq!(
        paths(&store.search(&o).expect("cari")),
        vec!["/episodik/makan"]
    );
}

#[test]
fn filter_namespace() {
    let store = store();
    let mut o = opts("nasi goreng");
    o.namespace = Some("semantik".to_string());
    assert_eq!(
        paths(&store.search(&o).expect("cari")),
        vec!["/semantik/resep"]
    );
    o.namespace = Some("ruang-hampa".to_string());
    assert!(store.search(&o).expect("cari").is_empty());
}

#[test]
fn filter_tag() {
    let store = store();
    let file_id = vir_mfs::tag(&store, "/semantik/resep", "resep").expect("tag");
    assert!(file_id > 0);
    assert_eq!(
        store.files_with_tag("resep").expect("berkas bertag"),
        vec![file_id]
    );
    let mut o = opts("nasi goreng");
    o.tags = vec!["resep".to_string()];
    assert_eq!(
        paths(&store.search(&o).expect("cari")),
        vec!["/semantik/resep"]
    );
    o.tags = vec!["tidak-ada".to_string()];
    assert!(store.search(&o).expect("cari").is_empty());
}

#[test]
fn filter_waktu() {
    let store = store();
    let t = now();
    let mut o = opts("nasi goreng");
    o.since = Some(t + 3_600);
    assert!(
        store.search(&o).expect("cari").is_empty(),
        "versi masa depan tak ada"
    );
    o.since = Some(t - 3_600);
    o.until = Some(t + 3_600);
    assert_eq!(store.search(&o).expect("cari").len(), 2);
    o.until = Some(t - 3_600);
    assert!(store.search(&o).expect("cari").is_empty());
}

#[test]
fn filter_kosong_dan_limit_nol() {
    let store = store();
    let mut o = opts("nasi goreng");
    o.limit = 0;
    assert!(store.search(&o).expect("cari").is_empty());
    assert!(store.search(&opts("   ")).expect("cari").is_empty());
    assert!(store.search_fts("", 5).expect("cari").is_empty());
}

#[test]
fn yang_dilupakan_tersembunyi_kecuali_diminta() {
    let store = store();
    let vid = store
        .current_version("/episodik/makan")
        .expect("terkini")
        .expect("ada versi");
    store.forget(vid).expect("lupa");
    let hits = store.search(&opts("nasi goreng")).expect("cari");
    assert_eq!(
        paths(&hits),
        vec!["/semantik/resep"],
        "versi yang dilupakan disembunyikan"
    );
    let mut o = opts("nasi goreng");
    o.include_forgotten = true;
    assert_eq!(store.search(&o).expect("cari").len(), 2);
}

#[test]
fn find_hibrida_mengembalikan_skor() {
    let store = store();
    let hits = vir_mfs::find(&store, "nasi goreng", 5).expect("cari");
    assert_eq!(hits.len(), 2);
    for hit in &hits {
        assert!(hit.score > 0.0, "skor harus positif: {hit:?}");
        assert!(hit.fts_rank.is_some(), "peringkat FTS terisi");
        assert!(hit.vector_rank.is_some(), "peringkat vektor terisi");
        assert!(!hit.content.is_empty());
    }
    assert!(hits[0].score >= hits[1].score, "hasil terurut menurun");
}

#[test]
fn mode_kata_kunci_tanpa_vektor() {
    let store = store();
    let mut o = opts("nasi goreng");
    o.mode = SearchMode::Keyword;
    let hits = store.search(&o).expect("cari");
    assert_eq!(hits.len(), 2);
    assert!(hits.iter().all(|h| h.vector_rank.is_none()));
    assert!(hits.iter().all(|h| h.fts_rank.is_some()));
}
