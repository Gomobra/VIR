//! Uji integrasi F1.8: jalankan `vir smoke --json` sebagai proses nyata dan
//! validasi keluarannya terhadap skema serta gerbang G1.
//!
//! Bila binary `vir` belum dibangun, tes dilewati dengan pesan jelas (bukan
//! `todo!`). Job `e2e` membangun `vir-cli` lebih dulu sehingga uji ini benar
//!-benar berjalan. Pipeline penuh (BPE → VQ → generator → decoder → sampling)
//! diisi oleh model-eng/bind-eng; validator di `vir-e2e` sudah menegakkan field
//! gerbang begitu field-nya muncul.

use std::path::PathBuf;
use std::process::Command;

use vir_e2e::validate_smoke_json;

/// Cari binary `vir`: `VIR_BIN` bila diset, selain itu target debug/release.
fn vir_bin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("VIR_BIN") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    ["target/debug/vir", "target/release/vir"]
        .iter()
        .map(|rel| root.join(rel))
        .find(|path| path.exists())
}

#[test]
fn smoke_json_matches_schema_and_gate() {
    let Some(bin) = vir_bin() else {
        assert!(
            std::env::var("VIR_E2E_REQUIRE_BIN").is_err(),
            "binary `vir` wajib ada (VIR_E2E_REQUIRE_BIN diset) tetapi tidak ditemukan"
        );
        eprintln!(
            "vir-e2e: binary `vir` tidak ditemukan; lewati uji CLI \
             (bangun dengan `cargo build -p vir-cli`)."
        );
        return;
    };
    let out = Command::new(&bin)
        .args(["smoke", "--json"])
        .output()
        .expect("jalankan `vir smoke --json`");
    assert!(
        out.status.success(),
        "`vir smoke` gagal: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout smoke harus UTF-8");
    let report = validate_smoke_json(stdout.trim()).expect("skema `vir smoke` valid");
    report.gate_g1_ok().expect("gerbang G1 lulus");
}
