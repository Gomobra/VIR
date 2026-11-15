//! F3.3 — operasi `ops.rs`: ls/tree/find/read/write/link/tag/history/diff/
//! forget/restore benar-benar berjalan di atas toko nyata.

use vir_mfs::{
    LinkKind, MemoryStore, ReadTarget, diff, find, forget, history, link, links, ls, neighbors,
    read, restore, tag, tree, untag, write,
};

fn tulis(store: &MemoryStore, path: &str, content: &str) -> i64 {
    write(store, path, content).expect("tulis").version_id
}

fn store() -> MemoryStore {
    let store = MemoryStore::open_in_memory().expect("buka toko");
    tulis(&store, "/episodik/a", "apa kabar");
    tulis(&store, "/episodik/sub/b", "dalam sub");
    tulis(&store, "/semantik/c", "filsafat");
    store
}

#[test]
fn ls_menelusuri_prefiks_segmen() {
    let store = store();
    let semua: Vec<String> = ls(&store, "/")
        .expect("ls")
        .into_iter()
        .map(|e| e.path)
        .collect();
    assert_eq!(semua, vec!["/episodik/a", "/episodik/sub/b", "/semantik/c"]);
    let ep: Vec<String> = ls(&store, "/episodik")
        .expect("ls")
        .into_iter()
        .map(|e| e.path)
        .collect();
    assert_eq!(ep, vec!["/episodik/a", "/episodik/sub/b"]);
    let satu: Vec<String> = ls(&store, "/episodik/a")
        .expect("ls")
        .into_iter()
        .map(|e| e.path)
        .collect();
    assert_eq!(satu, vec!["/episodik/a"]);
    assert!(ls(&store, "/tidak/ada").expect("ls").is_empty());
}

#[test]
fn ls_menyembunyikan_yang_dilupakan() {
    let store = store();
    forget(&store, "/semantik/c").expect("lupa");
    let semua: Vec<String> = ls(&store, "/")
        .expect("ls")
        .into_iter()
        .map(|e| e.path)
        .collect();
    assert!(!semua.contains(&"/semantik/c".to_string()));
    restore(&store, "/semantik/c").expect("pulihkan");
    assert!(
        ls(&store, "/")
            .expect("ls")
            .iter()
            .any(|e| e.path == "/semantik/c")
    );
}

#[test]
fn tree_membangun_ruang_nama() {
    let store = store();
    let root = tree(&store, "/").expect("tree");
    assert_eq!(root.path, "/");
    let nama: Vec<&str> = root.children.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(nama, vec!["episodik", "semantik"]);
    let ep = tree(&store, "/episodik").expect("tree");
    assert_eq!(
        (ep.name.as_str(), ep.path.as_str()),
        ("episodik", "/episodik")
    );
    let anak: Vec<&str> = ep.children.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(anak, vec!["a", "sub"]);
    let sub = ep
        .children
        .iter()
        .find(|c| c.name == "sub")
        .expect("simpul sub");
    assert!(sub.entry.is_none(), "direktori belakangan bukan berkas");
    assert!(sub.children.iter().any(|c| c.entry.is_some()));
    assert!(
        tree(&store, "/tidak/ada")
            .expect("tree")
            .children
            .is_empty()
    );
}

#[test]
fn read_mengembalikan_isi_dan_hash() {
    let store = store();
    let v1 = tulis(&store, "/episodik/a", "versi satu");
    let r = read(&store, ReadTarget::Path("/episodik/a")).expect("baca");
    assert_eq!(r.version_id, v1);
    assert_eq!(r.content, "versi satu");
    assert_eq!(r.hash, vir_mfs::hex_sha256(b"versi satu"));
    assert_eq!(r.path, "/episodik/a");
    let v2 = tulis(&store, "/episodik/a", "versi dua");
    assert_eq!(
        read(&store, ReadTarget::Path("/episodik/a"))
            .expect("baca")
            .version_id,
        v2
    );
    assert_eq!(
        read(&store, ReadTarget::Version(v1)).expect("baca").content,
        "versi satu"
    );
    assert!(read(&store, ReadTarget::Version(9_999)).is_err());
}

#[test]
fn history_menunjukkan_rantai_versi() {
    let store = store();
    let v1 = tulis(&store, "/episodik/riwayat", "satu");
    let v2 = tulis(&store, "/episodik/riwayat", "dua");
    let h = history(&store, "/episodik/riwayat").expect("riwayat");
    let ids: Vec<i64> = h.iter().map(|i| i.version_id).collect();
    assert_eq!(ids, vec![v1, v2]);
    assert_eq!(h[0].supersedes, None);
    assert_eq!(h[1].supersedes, Some(v1));
    assert_eq!(h[1].content_len, 3);
    assert!(h.iter().all(|i| !i.deleted));
    assert!(history(&store, "/tidak/ada").expect("riwayat").is_empty());
}

#[test]
fn diff_menghitung_baris_berubah() {
    let store = store();
    let v1 = tulis(&store, "/episodik/a", "satu\ndua\ntiga");
    let v2 = tulis(&store, "/episodik/a", "satu\nDUA\ntiga");
    let d = diff(&store, v1, v2).expect("diff");
    assert_eq!((d.added, d.removed), (1, 1));
    assert!(d.changed());
    assert_eq!(d.summary(), "+1 -1");
    assert!(d.unified().contains("+ DUA"));
    let sama = diff(&store, v1, v1).expect("diff");
    assert!(sama.unchanged());
    assert_eq!(sama.added, 0);
}

#[test]
fn link_dan_tetangga_graf() {
    let store = store();
    let v1 = tulis(&store, "/episodik/a", "satu");
    let v2 = tulis(&store, "/episodik/a", "dua");
    link(&store, v1, v2, LinkKind::DerivedFrom).expect("taut");
    assert!(
        links(&store, v1)
            .expect("tautan")
            .contains(&(v2, LinkKind::DerivedFrom))
    );
    assert_eq!(neighbors(&store, v1).expect("tetangga"), vec![v2]);
    assert_eq!(neighbors(&store, v2).expect("tetangga"), vec![v1]);
    assert!(link(&store, v1, v1, LinkKind::DerivedFrom).is_err());
    assert!(link(&store, v1, 9_999, LinkKind::DerivedFrom).is_err());
}

#[test]
fn tag_dan_untag_melalui_path_atau_id() {
    let store = store();
    let fid = tag(&store, "/episodik/a", "penting").expect("tag");
    assert_eq!(fid, store.file_id("/episodik/a").expect("id").expect("ada"));
    assert_eq!(store.tags(fid).expect("tag"), vec!["penting".to_string()]);
    tag(&store, &fid.to_string(), "uji").expect("tag id");
    assert_eq!(
        store.tags(fid).expect("tag"),
        vec!["penting".to_string(), "uji".to_string()]
    );
    untag(&store, "/episodik/a", "penting").expect("untag");
    assert_eq!(store.tags(fid).expect("tag"), vec!["uji".to_string()]);
    assert!(untag(&store, "/episodik/a", "penting").is_err());
    assert!(tag(&store, "/episodik/a", "   ").is_err());
    assert!(tag(&store, "/tidak/ada", "x").is_err());
}

#[test]
fn forget_dan_restore_melalui_ops() {
    let store = store();
    let vid = forget(&store, "/semantik/c").expect("lupa");
    assert!(store.is_forgotten(vid).expect("status"));
    assert!(read(&store, ReadTarget::Path("/semantik/c")).is_err());
    assert_eq!(
        read(&store, ReadTarget::Version(vid))
            .expect("baca")
            .content,
        "filsafat"
    );
    restore(&store, &vid.to_string()).expect("pulihkan");
    assert!(!store.is_forgotten(vid).expect("status"));
    assert_eq!(
        read(&store, ReadTarget::Path("/semantik/c"))
            .expect("baca")
            .content,
        "filsafat"
    );
    assert!(forget(&store, "/tidak/ada").is_err());
    assert!(forget(&store, "9_999").is_err());
}

#[test]
fn find_mengembalikan_hasil_ber_skor() {
    let store = store();
    tulis(&store, "/episodik/catatan", "filsafat memori");
    let hits = find(&store, "filsafat", 5).expect("cari");
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h.score > 0.0));
    assert!(hits.iter().any(|h| h.path == "/semantik/c"));
    assert!(
        find(&store, "kata-yang-tak-ada", 5)
            .expect("cari")
            .is_empty()
    );
}
