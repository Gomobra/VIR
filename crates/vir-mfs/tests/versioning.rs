//! F3.3.a — penyimpanan dan versi: `write` selalu menambah, isi mentah tak
//! berubah, `forget` hanya menandai, `restore` melepas, hash selalu cocok.

use vir_mfs::{MemoryStore, hex_sha256};

#[test]
fn write_selalu_menambah_versi_baru() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let v1 = store
        .write("/semantik/catatan", "versi satu")
        .expect("tulis 1");
    let v2 = store
        .write("/semantik/catatan", "versi dua")
        .expect("tulis 2");
    assert_ne!(v1, v2, "setiap write harus id versi baru");
    assert_eq!(
        store.history("/semantik/catatan").expect("riwayat"),
        vec![v1, v2]
    );
    let info = store.version_info(v2).expect("info v2");
    assert_eq!(info.supersedes, Some(v1), "versi baru menaut versi lama");
    assert_eq!(
        store.current_version("/semantik/catatan").expect("terkini"),
        Some(v2)
    );
    assert_eq!(
        store.read_path("/semantik/catatan").expect("baca"),
        "versi dua"
    );
}

#[test]
fn versi_lama_tetap_utuh_setelah_write_baru() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let v1 = store
        .write("/semantik/catatan", "isi lama")
        .expect("tulis 1");
    let _ = store
        .write("/semantik/catatan", "isi baru")
        .expect("tulis 2");
    assert_eq!(
        store.read(v1).expect("baca v1"),
        "isi lama",
        "isi lama tak boleh berubah"
    );
    assert!(
        store.verify(v1).expect("verifikasi v1"),
        "hash v1 tetap cocok"
    );
    assert_eq!(
        store.version_info(v1).expect("info v1").content_len,
        "isi lama".len() as i64
    );
}

#[test]
fn hash_tersimpan_sama_dengan_sha256_isi() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let vid = store.write("/semantik/catatan", "abc").expect("tulis");
    assert_eq!(store.hash_of(vid).expect("hash"), hex_sha256(b"abc"));
    assert!(store.verify(vid).expect("verifikasi"));
    assert_eq!(store.read(vid).expect("baca"), "abc");
}

#[test]
fn forget_hanya_menandai_dan_restore_melepas() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let vid = store
        .write("/kerja/sementara", "isi sementara")
        .expect("tulis");
    store.forget(vid).expect("lupa");
    assert!(store.is_forgotten(vid).expect("status"));
    assert!(
        store.read_path("/kerja/sementara").is_err(),
        "path yang dilupakan gagal dibaca"
    );
    assert_eq!(
        store.read(vid).expect("baca by id"),
        "isi sementara",
        "isi tetap ada"
    );
    assert!(store.verify(vid).expect("verifikasi"));
    store.restore(vid).expect("pulihkan");
    assert!(!store.is_forgotten(vid).expect("status"));
    assert_eq!(
        store.read_path("/kerja/sementara").expect("baca"),
        "isi sementara"
    );
}

#[test]
fn forget_tidak_menghapus_dari_riwayat() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let v1 = store.write("/semantik/catatan", "satu").expect("tulis 1");
    let v2 = store.write("/semantik/catatan", "dua").expect("tulis 2");
    store.forget(v1).expect("lupa v1");
    assert_eq!(
        store.history("/semantik/catatan").expect("riwayat"),
        vec![v1, v2]
    );
    assert_eq!(store.read(v1).expect("v1 masih terbaca"), "satu");
    assert!(store.is_forgotten(v1).expect("status"));
    assert!(!store.is_forgotten(v2).expect("status"));
}

#[test]
fn operasi_pada_versi_tak_ada_gagal() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    assert!(store.forget(4_242).is_err());
    assert!(store.restore(4_242).is_err());
    assert!(store.read(4_242).is_err());
    assert!(!store.verify(4_242).expect("verifikasi"));
    assert!(store.read_path("/tidak/ada").is_err());
}

#[test]
fn isi_melebihi_batas_ditolak() {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    let besar = "x".repeat(vir_mfs::MAX_CONTENT_BYTES + 1);
    assert!(store.write("/semantik/besar", &besar).is_err());
}

#[test]
fn data_tetap_setelah_buka_ulang_basis_berkas() {
    // Jalur `open_connection` (WAL + PRAGMA) yang tak tersentuh toko in-memory.
    let dir = std::env::temp_dir().join(format!("vir-mfs-persist-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("buat dir");
    let path = dir.join("mem.db").to_str().expect("path utf8").to_string();
    let vid = {
        let store = MemoryStore::open(&path).expect("buka berkas");
        store
            .write("/semantik/persist", "tetap ada")
            .expect("tulis")
    };
    {
        let store = MemoryStore::open(&path).expect("buka ulang");
        assert_eq!(store.read(vid).expect("baca"), "tetap ada");
        assert!(store.verify(vid).expect("verifikasi"));
    }
    let _ = std::fs::remove_dir_all(&dir);
}
