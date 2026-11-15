//! Tes jembatan MFS dan loop RLM (F3.4, F3.7).
//!
//! Membuktikan: `mem` menyentuh MFS nyata (bukan `null`), dan loop RLM hanya
//! melaporkan `FINAL(...)` bila model benar-benar mencetaknya.

use serde_json::json;
use vir_common::VirError;

use crate::lm::ScriptedLm;
use crate::mem_bridge::MemBridge;
use crate::rlm::{RlmConfig, run_rlm};

fn cfg() -> RlmConfig {
    RlmConfig {
        max_depth: 2,
        max_calls: 4,
        max_steps: 4,
    }
}

fn bridge() -> MemBridge {
    MemBridge::open_in_memory().expect("buka memori")
}

#[test]
fn mem_read_mengembalikan_isi_nyata() {
    let bridge = bridge();
    let vid = bridge.write_file("/n/catatan", "ascii").expect("tulis");
    let value = bridge
        .call(&json!({ "op": "read", "path": "/n/catatan" }))
        .expect("baca");
    assert_eq!(value["content"], "ascii");
    assert_eq!(value["version_id"], vid);
}

#[test]
fn mem_read_id_versi_lewat_jembatan() {
    let bridge = bridge();
    let vid = bridge.write_file("/n/catatan", "versi").expect("tulis");
    let value = bridge
        .call(&json!({ "op": "read", "path": vid.to_string() }))
        .expect("baca id");
    assert_eq!(value["content"], "versi");
}

#[test]
fn mem_write_lalu_find_menemukan_isi() {
    let bridge = bridge();
    bridge
        .call(&json!({ "op": "write", "path": "/n/catatan", "content": "hashing sharding" }))
        .expect("tulis");
    let value = bridge
        .call(&json!({ "op": "find", "query": "hashing" }))
        .expect("cari");
    let hits = value.as_array().expect("array hasil");
    assert!(!hits.is_empty(), "pencarian MFS harus menemukan isi nyata");
    assert_eq!(hits[0]["content"], "hashing sharding");
}

#[test]
fn mem_ls_menyebut_berkas() {
    let bridge = bridge();
    bridge
        .call(&json!({ "op": "write", "path": "/n/catatan", "content": "x" }))
        .expect("tulis");
    let value = bridge
        .call(&json!({ "op": "ls", "path": "/" }))
        .expect("daftar");
    let items = value.as_array().expect("array");
    assert!(items.iter().any(|i| i["path"] == "/n/catatan"));
}

#[test]
fn operasi_tak_dikenal_jadi_galat() {
    let err = bridge()
        .call(&json!({ "op": "entah" }))
        .expect_err("operasi tak dikenal harus gagal");
    assert!(!err.to_string().is_empty());
}

#[test]
fn read_tanpa_target_jadi_galat() {
    let err = bridge()
        .call(&json!({ "op": "read" }))
        .expect_err("read tanpa path/id harus gagal");
    let _: &VirError = &err;
}

#[test]
fn rlm_final_hanya_dari_model() {
    let lm = ScriptedLm::new(vec!["print('FINAL(ok)')".to_string()]);
    let bridge = bridge();
    let mut mem = |payload: &serde_json::Value| bridge.call(payload);
    let out = run_rlm(&lm, "tugas", &cfg(), &mut mem).expect("rlm");
    assert_eq!(out.final_answer.as_deref(), Some("FINAL(ok)"));
    assert_eq!(out.steps, 1);
}

#[test]
fn rlm_mem_membaca_mfs_nyata() {
    let lm = ScriptedLm::new(vec!["print(mem('find', query='alpha'))".to_string()]);
    let bridge = bridge();
    bridge
        .call(&json!({ "op": "write", "path": "/n/a", "content": "alpha beta" }))
        .expect("tulis");
    let mut mem = |payload: &serde_json::Value| bridge.call(payload);
    let out = run_rlm(&lm, "tugas", &cfg(), &mut mem).expect("rlm");
    let text = out.final_answer.unwrap_or_default();
    assert!(
        text.contains("alpha beta"),
        "keluaran REPL harus memuat isi MFS nyata: {text}"
    );
}

#[test]
fn rlm_tanpa_final_melaporkan_none() {
    let lm = ScriptedLm::new(vec!["x = 1".to_string()]);
    let bridge = bridge();
    let mut mem = |payload: &serde_json::Value| bridge.call(payload);
    let out = run_rlm(&lm, "tugas", &cfg(), &mut mem).expect("rlm");
    assert!(out.final_answer.is_none());
    assert!(out.steps >= 1);
}
