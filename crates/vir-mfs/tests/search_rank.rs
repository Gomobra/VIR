//! F3.3.b — pencarian: FTS5 BM25, vektor n-gram hash, fusi RRF (contoh hitung
//! tangan), dan dedup kosinus yang menaut (bukan menghapus).

use vir_mfs::{FusionConfig, LinkKind, MemoryStore, age_factor, rank_score, rrf_fuse};

#[test]
fn fts5_bm25_menemukan_dokumen() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let vid = store
        .write("/episodik/kucing", "kucing belang tidur di sofa")
        .expect("tulis kucing");
    store
        .write("/episodik/kapal", "kapal berlayar di laut")
        .expect("tulis kapal");
    let hits = store.search_fts("kucing", 5).expect("cari fts");
    assert_eq!(hits.len(), 1, "hanya satu dokumen memuat kata kunci");
    assert_eq!(hits[0].version_id, vid);
    assert_eq!(hits[0].path, "/episodik/kucing");
    assert!(hits[0].score > 0.0, "skor BM25 harus positif");
    assert!(hits[0].content.contains("kucing"));
}

#[test]
fn vektor_menemukan_dokumen_mirip() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let mirip = store
        .write("/semantik/memori", "catatan tentang rust dan memori")
        .expect("tulis mirip");
    store
        .write("/semantik/masak", "resep nasi goreng pedas")
        .expect("tulis lain");
    let hits = store.search_vector("rust memori", 5).expect("cari vektor");
    assert_eq!(hits.first().map(|h| h.version_id), Some(mirip));
    assert!(
        hits[0].vector_rank.is_some(),
        "peringkat vektor harus terisi"
    );
    assert!(hits[0].score > 0.0);
}

#[test]
fn rrf_contoh_hitung_tangan() {
    // Dokumen 10: peringkat 1 di FTS dan peringkat 3 di vektor.
    let fts = [10_i64, 11];
    let vectors = [12_i64, 13, 10];
    let fused = rrf_fuse(&[(&fts[..], 1.0), (&vectors[..], 1.0)], 60.0);
    let skor = |id: i64| fused.iter().find(|(i, _)| *i == id).map(|(_, s)| *s);
    let harap = 1.0 / 61.0 + 1.0 / 63.0;
    assert!(
        (skor(10).expect("id 10") - harap).abs() < 1e-9,
        "1/61 + 1/63 = 0,03227"
    );
    assert!(
        (harap - 0.03227).abs() < 1e-4,
        "contoh PRD: 1/61 + 1/63 ≈ 0,03227"
    );
    assert_eq!(fused[0].0, 10, "dokumen dengan dua daftar menang");
    assert_eq!(fused[1].0, 12, "peringkat 1 satu daftar menyusul");
    assert!((fused[2].1 - 1.0 / 62.0).abs() < 1e-9);
    assert!((fused[3].1 - 1.0 / 62.0).abs() < 1e-9);
    assert_eq!(fused[2].0, 11, "seri diselesaikan id menaik");
    assert_eq!(fused[3].0, 13);
}

#[test]
fn rrf_menghormati_bobot_dan_konstanta() {
    let fts = [10_i64, 11];
    let vectors = [12_i64, 13, 10];
    let berbobot = rrf_fuse(&[(&fts[..], 2.0), (&vectors[..], 1.0)], 60.0);
    let skor = |id: i64| berbobot.iter().find(|(i, _)| *i == id).map(|(_, s)| *s);
    assert!((skor(10).expect("id 10") - (2.0 / 61.0 + 1.0 / 63.0)).abs() < 1e-9);
}

#[test]
fn rrf_menghormati_konstanta_k() {
    let ids = [10_i64, 11];
    let kecil = rrf_fuse(&[(&ids[..], 1.0)], 10.0);
    let besar = rrf_fuse(&[(&ids[..], 1.0)], 60.0);
    assert!((kecil[0].1 - 1.0 / 11.0).abs() < 1e-9);
    assert!((besar[0].1 - 1.0 / 61.0).abs() < 1e-9);
}

#[test]
fn faktor_usia_dan_akses_memengaruhi_skor() {
    let cfg = FusionConfig::default();
    assert!((age_factor(0.0, 30.0) - 1.0).abs() < 1e-9);
    assert!((age_factor(30.0, 30.0) - 0.5).abs() < 1e-9);
    assert!(rank_score(1.0, 0.0, 0, &cfg) > rank_score(1.0, 60.0, 0, &cfg));
    assert!(rank_score(1.0, 0.0, 5, &cfg) > rank_score(1.0, 0.0, 0, &cfg));
    assert!((rank_score(1.0, 0.0, 0, &cfg) - 1.0).abs() < 1e-9);
}

#[test]
fn dedup_kosinus_menaut_bukan_menghapus() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let v1 = store
        .write("/episodik/a", "rust memori aman")
        .expect("tulis 1");
    let out = store
        .write_version("/episodik/b", "rust memori aman", None)
        .expect("tulis 2");
    let (dup, score) = out.duplicate_of.expect("duplikat harus terdeteksi");
    assert_eq!(dup, v1);
    assert!(score > 0.99, "kosinus kembar harus ~1, dapat {score}");
    assert!(store.version_exists(out.version_id).expect("versi ada"));
    assert_eq!(
        store.read(out.version_id).expect("baca"),
        "rust memori aman"
    );
    assert!(
        store
            .links_from(out.version_id)
            .expect("tautan")
            .contains(&(v1, LinkKind::DuplicateOf)),
        "duplikat harus menaut versi asli"
    );
    assert!(
        store
            .dedup_check("rust memori aman")
            .expect("cek")
            .is_some(),
        "isi kembar terdeteksi sebelum ditulis"
    );
}

#[test]
fn dedup_di_bawah_ambang_tidak_menaut() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    store
        .write("/episodik/a", "rust memori aman")
        .expect("tulis 1");
    let out = store
        .write_version("/episodik/b", "resep kue lapis", None)
        .expect("tulis 2");
    assert!(out.duplicate_of.is_none());
    assert!(store.dedup_check("resep kue lapis").expect("cek").is_none());
    assert!(store.links_from(out.version_id).expect("tautan").is_empty());
}
