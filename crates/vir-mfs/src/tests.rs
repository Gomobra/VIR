//! Tes putih (butuh akses `conn`): isi mentah di basis data tidak pernah
//! berubah, dan hash yang tak cocok terdeteksi saat dibaca.

use rusqlite::params;

use crate::MemoryStore;

fn raw_contents(store: &MemoryStore) -> Vec<String> {
    let mut stmt = store
        .conn
        .prepare("SELECT content FROM versions ORDER BY id")
        .expect("siapkan kueri");
    let rows = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .expect("kueri");
    rows.map(|r| r.expect("baris")).collect()
}

#[test]
fn isi_mentah_tak_berubah_karena_write_forget_restore() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let v1 = store.write("/semantik/a", "satu").expect("tulis 1");
    let v2 = store.write("/semantik/a", "dua").expect("tulis 2");
    store.forget(v1).expect("lupa");
    store.restore(v1).expect("pulihkan");
    store.forget(v2).expect("lupa");
    assert_eq!(
        raw_contents(&store),
        vec!["satu".to_string(), "dua".to_string()]
    );
    assert!(store.verify(v1).expect("verifikasi"));
    assert!(store.verify(v2).expect("verifikasi"));
}

#[test]
fn baca_menambah_log_akses_tanpa_mengubah_isi() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let vid = store.write("/semantik/a", "baca aku").expect("tulis");
    assert_eq!(store.access_count(vid).expect("akses"), 0);
    let sebelum = raw_contents(&store);
    let hash_sblm = store.hash_of(vid).expect("hash");
    assert_eq!(store.read(vid).expect("baca"), "baca aku");
    assert_eq!(raw_contents(&store), sebelum, "isi mentah tak berubah");
    assert_eq!(store.hash_of(vid).expect("hash"), hash_sblm, "hash tetap");
    assert_eq!(
        store.access_count(vid).expect("akses"),
        1,
        "akses masuk log terpisah"
    );
}

#[test]
fn hash_tak_cocok_terdeteksi_saat_dibaca() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let vid = store.write("/semantik/a", "asli").expect("tulis");
    let n = store
        .conn
        .execute(
            "UPDATE versions SET content = 'rusak' WHERE id = ?1",
            params![vid],
        )
        .expect("ubah isi");
    assert_eq!(n, 1);
    assert!(!store.verify(vid).expect("verifikasi"));
    assert!(
        store.read(vid).is_err(),
        "baca harus menolak isi yang berubah"
    );
}
