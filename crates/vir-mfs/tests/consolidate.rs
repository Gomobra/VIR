//! F3.3.c — konsolidasi non-destruktif: sumber tetap utuh, hasilnya versi baru
//! dengan jejak asal.

use vir_mfs::{ConsolidationReport, HeadlineSummarizer, LinkKind, MemoryStore};

fn siapkan_sumber(store: &MemoryStore) -> i64 {
    store
        .write(
            "/episodik/panjang",
            "epilog panjang sekali tentang memori\nbaris kedua",
        )
        .expect("tulis sumber")
}

fn konsolidasi(store: &MemoryStore, src: i64) -> ConsolidationReport {
    store
        .consolidate(
            src,
            "/semantik/kartu",
            &HeadlineSummarizer { max_chars: 10 },
        )
        .expect("konsolidasi")
}

#[test]
fn konsolidasi_menulis_versi_baru_dengan_asal() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let src = siapkan_sumber(&store);
    let report = konsolidasi(&store, src);
    assert_ne!(
        report.source_version, report.version_id,
        "hasil harus versi baru"
    );
    assert_eq!(
        store
            .version_info(report.version_id)
            .expect("info hasil")
            .source,
        Some(src)
    );
    assert_eq!(
        store.read(report.version_id).expect("ringkasan"),
        "epilog pan"
    );
}

#[test]
fn konsolidasi_menaut_derived_from() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let src = siapkan_sumber(&store);
    let report = konsolidasi(&store, src);
    assert!(
        store
            .links_from(report.version_id)
            .expect("tautan")
            .contains(&(src, LinkKind::DerivedFrom))
    );
    assert!(store.verify(report.version_id).expect("verifikasi"));
}

#[test]
fn konsolidasi_tidak_mengubah_isi_sumber() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let src = siapkan_sumber(&store);
    let sebelum = store.read(src).expect("baca sumber");
    let hash_sblm = store.hash_of(src).expect("hash sumber");
    let report = konsolidasi(&store, src);
    assert_ne!(report.source_version, report.version_id);
    assert_eq!(
        store.read(src).expect("baca sumber"),
        sebelum,
        "isi sumber tak berubah"
    );
    assert_eq!(
        store.hash_of(src).expect("hash"),
        hash_sblm,
        "hash sumber tetap"
    );
}

#[test]
fn pending_episodik_berkurang_setelah_konsolidasi() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let a = store
        .write("/episodik/satu", "baris satu\nbaris dua")
        .expect("tulis a");
    let b = store
        .write("/episodik/dua", "baris tiga\nbaris empat")
        .expect("tulis b");
    assert_eq!(
        store.pending_episodes("episodik").expect("tertunda"),
        vec![a, b]
    );

    let reports = store
        .consolidate_pending(
            "episodik",
            "/semantik/kartu",
            &HeadlineSummarizer::default(),
        )
        .expect("konsolidasi");

    assert_eq!(reports.len(), 2);
    assert!(
        store
            .pending_episodes("episodik")
            .expect("tertunda")
            .is_empty()
    );
    assert_eq!(store.read(a).expect("isi a"), "baris satu\nbaris dua");
    assert_eq!(store.read(b).expect("isi b"), "baris tiga\nbaris empat");
    assert!(store.verify(a).expect("verifikasi"));
    assert!(store.verify(b).expect("verifikasi"));
}

#[test]
fn konsolidasi_sumber_tak_ada_gagal() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let err = store
        .consolidate(4_242, "/semantik/kartu", &HeadlineSummarizer::default())
        .expect_err("sumber tak ada harus gagal");
    assert!(!err.to_string().is_empty());
}
