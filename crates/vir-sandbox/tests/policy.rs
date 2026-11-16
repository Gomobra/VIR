//! Tes CPU `vir-sandbox` yang berjalan tanpa kernel (F3.1).
//!
//! Job `nextest` bawaan tidak mengaktifkan fitur `sandbox`, jadi tes ini hanya
//! memakai logika murni (kebijakan, batas sumber daya) dan subprocess biasa
//! dengan rlimit + timeout. Konfinemen Landlock/seccomp diuji lewat build
//! `cargo build -p vir-sandbox --features sandbox` di job `vir-sandbox`.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use vir_sandbox::limits::ResourceLimits;
use vir_sandbox::{Policy, run, spawn};

#[test]
fn policy_bawaan_memuat_batas_wajar() {
    let policy = Policy::default();
    assert_eq!(policy.timeout, Duration::from_secs(5));
    assert_eq!(policy.max_output, 64 * 1024);
    assert!(
        policy.clear_env,
        "lingkungan harus dibersihkan secara bawaan"
    );
    assert!(
        policy.landlock && policy.seccomp,
        "konfinemen kernel wajib aktif secara bawaan"
    );
    assert!(policy.workdir.is_none());
    assert!(policy.allow_write.is_empty());
    assert_eq!(policy.limits, ResourceLimits::standard());
}

#[test]
fn policy_bawaan_berisi_env_terbatas() {
    let env = Policy::default().env;
    let keys: Vec<&str> = env.keys().map(String::as_str).collect();
    assert_eq!(keys, vec!["HOME", "LANG", "LC_ALL", "PATH", "TMPDIR"]);
    assert_eq!(env.get("TMPDIR").map(String::as_str), Some("/tmp"));
    assert_eq!(env.get("LANG").map(String::as_str), Some("C"));
}

#[test]
fn builder_mengubah_setiap_field() {
    let limits = ResourceLimits {
        cpu_secs: Some(1),
        mem_bytes: None,
        nofile: None,
        nproc: Some(2),
    };
    let policy = Policy::default()
        .with_timeout(Duration::from_millis(250))
        .with_max_output(16)
        .with_workdir("/srv/vir")
        .with_env("VIR_UJI", "1")
        .with_limits(limits)
        .with_landlock(false)
        .with_seccomp(false);
    assert_eq!(policy.timeout, Duration::from_millis(250));
    assert_eq!(policy.max_output, 16);
    assert_eq!(policy.workdir, Some(PathBuf::from("/srv/vir")));
    assert_eq!(policy.env.get("VIR_UJI").map(String::as_str), Some("1"));
    assert_eq!(policy.limits, limits);
    assert!(!policy.landlock && !policy.seccomp);
}

#[test]
fn allow_write_menambah_path_berurutan() {
    let policy = Policy::default()
        .with_allow_write("/a")
        .with_allow_write("/b");
    assert_eq!(
        policy.allow_write,
        vec![PathBuf::from("/a"), PathBuf::from("/b")]
    );
}

#[test]
fn resource_limits_bawaan_terpasang() {
    let limits = ResourceLimits::standard();
    assert_eq!(limits.cpu_secs, Some(10));
    assert_eq!(limits.mem_bytes, Some(512 * 1024 * 1024));
    assert_eq!(limits.nofile, Some(1024));
    assert_eq!(limits.nproc, Some(1024));
}

#[test]
fn resource_limits_kosong_tanpa_batas() {
    let kosong = ResourceLimits {
        cpu_secs: None,
        mem_bytes: None,
        nofile: None,
        nproc: None,
    };
    assert_eq!(ResourceLimits::default(), kosong);
}

/// Kebijakan untuk tes CPU: konfinemen kernel dimatikan agar yang diuji adalah
/// `run`/`spawn` (rlimit, timeout, penangkapan keluaran), bukan Landlock.
fn policy_cpu() -> Policy {
    Policy::default().with_landlock(false)
}

#[test]
fn run_menangkap_stdout_stderr_dan_status() {
    let policy = policy_cpu().with_max_output(4096);
    let out = run(
        "sh",
        &["-c", "printf halo; printf oops >&2; exit 3"],
        &policy,
    )
    .expect("perintah jalan");
    assert_eq!(out.stdout, "halo");
    assert_eq!(out.stderr, "oops");
    assert_eq!(out.exit_code, Some(3));
    assert!(
        !out.timed_out,
        "perintah cepat tidak boleh ditandai timeout"
    );
    assert!(!out.truncated, "keluaran kecil tidak boleh dipotong");
}

#[test]
fn run_meneruskan_env_kebijakan_ke_anak() {
    let policy = policy_cpu().with_env("VIR_UJI", "42");
    let out = run("sh", &["-c", "printf %s \"$VIR_UJI\""], &policy).expect("perintah jalan");
    assert_eq!(out.stdout, "42");
}

#[test]
fn run_memotong_keluaran_besar() {
    let policy = policy_cpu().with_max_output(1000);
    let out = run(
        "sh",
        &["-c", "head -c 5000 /dev/zero | tr '\\0' a"],
        &policy,
    )
    .expect("perintah jalan");
    assert!(out.truncated, "keluaran 5000 bita harus ditandai terpotong");
    assert_eq!(out.stdout.len(), 1000, "keluaran dipotong tepat pada batas");
}

#[test]
fn run_membunuh_proses_yang_melewati_timeout() {
    let policy = policy_cpu().with_timeout(Duration::from_millis(300));
    let start = Instant::now();
    let out = run("sh", &["-c", "sleep 30"], &policy).expect("perintah jalan");
    assert!(out.timed_out, "proses yang melewati timeout harus ditandai");
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "timeout harus membunuh proses, bukan menunggu selesai"
    );
}

#[test]
fn spawn_meneruskan_stdin_ke_stdout() {
    let mut child = spawn("cat", &[], &policy_cpu()).expect("spawn");
    {
        let mut stdin = child.stdin.take().expect("stdin anak");
        stdin.write_all(b"ping\n").expect("tulis ke anak");
    }
    let mut out = String::new();
    child
        .stdout
        .take()
        .expect("stdout anak")
        .read_to_string(&mut out)
        .expect("baca anak");
    let _ = child.wait();
    assert_eq!(out, "ping\n");
}

/// E-015: Landlock tanpa `workdir` tidak boleh diam-diam kehilangan konfinemen.
#[cfg(feature = "sandbox")]
#[test]
fn landlock_tanpa_workdir_ditolak() {
    let policy = Policy::default();
    assert!(policy.landlock && policy.workdir.is_none());
    let hasil = run("true", &[], &policy);
    let err = hasil.expect_err("kebijakan tanpa workdir harus ditolak");
    assert!(err.to_string().contains("workdir"), "pesan: {err}");
}

/// Tanpa Landlock, `workdir` kosong tetap sah (tes CPU di host apa pun).
#[test]
fn tanpa_landlock_tanpa_workdir_tetap_jalan() {
    let policy = policy_cpu();
    assert!(policy.workdir.is_none());
    let out = run("true", &[], &policy).expect("perintah jalan");
    assert_eq!(out.exit_code, Some(0));
}
