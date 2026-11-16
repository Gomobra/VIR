//! Kemampuan kernel untuk tes isolasi (F3.1 butir 7).
//!
//! Tes di `isolation.rs` memerlukan kernel Linux dengan Landlock dan/atau
//! seccomp. Runner CI memenuhi syarat itu, tetapi host lain tidak selalu.
//! Modul ini memusatkan deteksinya supaya alasan `SKIP` selalu memuat *apa* yang
//! hilang (ABI, seccomp, alasan pemasangan), bukan sekadar "dilewati".
//!
//! Modul ini hanya berisi deteksi dan utilitas; keputusan "tes mana butuh apa"
//! dibuat eksplisit di masing-masing tes.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use vir_sandbox::{Policy, RunResult, run};

/// Versi ABI Landlock efektif; `None` bila kernel tidak mendukung (< 5.13 atau
/// Landlock dimatikan saat boot).
pub fn landlock_abi() -> Option<u32> {
    vir_sandbox::landlock::detect_abi().ok()
}

/// `true` bila Landlock tersedia dengan ABI minimal yang diminta.
pub fn landlock_at_least(abi: u32) -> bool {
    landlock_abi().is_some_and(|v| v >= abi)
}

/// Dukungan seccomp-bpf (`prctl(PR_GET_SECCOMP)` tidak menolak).
pub fn seccomp_available() -> bool {
    vir_sandbox::seccomp::detect_support().is_ok()
}

/// Galat pemasangan konfinemen pada anak, bila ada.
///
/// "Kernel mendukung Landlock" tidak berarti "kernel ini mengizinkan
/// konfinemen": filter seccomp dapat ditolak (`unprivileged_bpf_disabled=2`)
/// atau aturan Landlock ditolak LSM lain. Uji dengan `true` di bawah kebijakan
/// bawaan agar tes isolasi tidak menuduh sandbox rusak padahal host yang menolak.
pub fn confinement_error() -> Option<String> {
    // `Policy::default()` meminta Landlock, yang mensyaratkan `workdir` (E-015);
    // galat pemasangan (spawn gagal) juga harus terlaporkan, bukan dianggap sukses.
    let dir = TempDir::new("konfinemen");
    let policy = policy_in(&dir);
    match run("true", &[], &policy) {
        Ok(out) if out.exit_code == Some(0) => None,
        Ok(out) => {
            let stderr: String = out.stderr.chars().take(240).collect();
            Some(format!("konfinemen anak gagal (exit {:?}): {stderr}", out.exit_code))
        }
        Err(err) => Some(format!("konfinemen anak gagal dipasang: {err}")),
    }
}

/// Ringkasan satu baris untuk pesan `SKIP`.
pub fn capabilities() -> String {
    format!(
        "landlock_abi={:?} seccomp={} pemasangan={:?}",
        landlock_abi(),
        seccomp_available(),
        confinement_error()
    )
}

/// Direktori kerja sementara unik per proses + pembersih saat gugur.
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path =
            std::env::temp_dir().join(format!("vir-iso-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&path).expect("buat direktori sementara");
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn file(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Kebijakan dengan workdir sementara dan timeout dinding yang wajar.
pub fn policy_in(dir: &TempDir) -> Policy {
    Policy::default()
        .with_workdir(dir.path())
        .with_timeout(Duration::from_secs(10))
}

/// Jalankan helper di dalam sandbox.
pub fn run_helper(args: &[&str], policy: &Policy) -> RunResult {
    run(helper(), args, policy).expect("helper vir-iso dapat dijalankan")
}

/// Ambil nilai bertanda `KUNCI=nilai` dari keluaran baris-per-baris.
pub fn field(stdout: &str, key: &str) -> Option<String> {
    stdout.lines().find_map(|line| {
        let (name, value) = line.split_once('=')?;
        (name == key).then(|| value.trim().to_string())
    })
}

/// Biner helper yang dibangun bersama tes (lihat `src/bin/vir-iso.rs`).
pub fn helper() -> &'static str {
    env!("CARGO_BIN_EXE_vir-iso")
}
