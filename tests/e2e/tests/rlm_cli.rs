//! Uji integrasi F3.4: `vir rlm` memakai LM nyata (proses eksternal) dan
//! memori MFS nyata — bukan jawaban `FINAL(ok)` yang di-hard-code.
//!
//! Skrip model dan basis data MFS dibuat di direktori sementara; jalankan
//! dengan `VIR_E2E_REQUIRE_BIN=1` agar CI gagal bila binary `vir` hilang.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

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

/// Direktori kerja sementara di bawah `std::env::temp_dir()`.
struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("vir-rlm-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("buat direktori sementara");
        Self { dir }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

/// Skrip model nyata: panggilan pertama menjalankan `mem('find', ...)`, panggilan
/// kedua membaca hasilnya dari transkrip dan mencetak `FINAL(...)`. Membuktikan
/// RLM benar-benar memanggil MFS nyata (bukan `FINAL(ok)` yang di-hard-code).
const MODEL_SCRIPT: &str = r#"import sys

transcript = sys.stdin.read()

# Transkrip sudah memuat hasil `find` dari MFS -> selesai.
if "alpha beta" in transcript:
    print("print('FINAL(ditemukan: alpha beta)')")
else:
    print("print(mem('find', query='alpha'))")
"#;

#[test]
fn rlm_mem_membaca_mfs_nyata() {
    let Some(bin) = vir_bin() else {
        assert!(
            std::env::var("VIR_E2E_REQUIRE_BIN").is_err(),
            "binary `vir` wajib ada (VIR_E2E_REQUIRE_BIN diset) tetapi tidak ditemukan"
        );
        eprintln!("vir-e2e: binary `vir` tidak ditemukan; lewati uji RLM.");
        return;
    };
    let scratch = Scratch::new("mfs");
    let db = scratch.path("memori.db");
    let script = scratch.path("model.py");
    fs::write(&script, MODEL_SCRIPT).expect("tulis skrip model");
    let db_text = db.to_str().expect("jalur db").to_string();
    let script_text = script.to_str().expect("jalur skrip").to_string();

    let seeded = Command::new(&bin)
        .args(["mem", "--path", &db_text, "--write", "alpha beta"])
        .output()
        .expect("seed MFS lewat `vir mem --write`");
    assert!(
        seeded.status.success(),
        "seed MFS gagal: {}",
        String::from_utf8_lossy(&seeded.stderr)
    );

    let out = Command::new(&bin)
        .args([
            "rlm",
            "--context",
            "uji memori",
            "--model",
            &script_text,
            "--mem",
            &db_text,
            "--json",
        ])
        .output()
        .expect("jalankan `vir rlm`");
    assert!(
        out.status.success(),
        "`vir rlm` gagal: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout UTF-8");
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("JSON keluar");
    let final_answer = value["final"].as_str().unwrap_or_default();
    assert!(
        final_answer.contains("alpha beta"),
        "`vir rlm` tidak membaca MFS nyata (final={final_answer:?}); stdout: {stdout}"
    );
    assert_eq!(value["scripted"], false, "LM proses nyata bukan scripted");
    assert_eq!(
        value["mem"].as_str(),
        Some(db_text.as_str()),
        "laporan harus menyebut basis data MFS yang dibuka"
    );
    assert!(value["steps"].as_u64().unwrap_or(0) >= 1);
}

#[test]
fn rlm_tanpa_model_gagal_tegas() {
    let Some(bin) = vir_bin() else {
        assert!(
            std::env::var("VIR_E2E_REQUIRE_BIN").is_err(),
            "binary `vir` wajib ada (VIR_E2E_REQUIRE_BIN diset) tetapi tidak ditemukan"
        );
        return;
    };
    let out = Command::new(&bin)
        .args(["rlm", "--context", "uji", "--mem", ":memory:"])
        .output()
        .expect("jalankan `vir rlm`");
    assert!(
        !out.status.success(),
        "`vir rlm` tanpa --model tidak boleh mengaku berhasil"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--model"),
        "galat harus menyebut --model: {stderr}"
    );
}
