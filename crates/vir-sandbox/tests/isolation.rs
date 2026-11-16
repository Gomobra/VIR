//! Tes isolasi kernel nyata (F3.1 butir 7): loop tak hingga, ledakan memori,
//! ledakan proses, tulis di luar direktori kerja, jaringan, syscall terlarang.
//!
//! Berbeda dari `tests/policy.rs` dan `tests/allowlist.rs` yang berjalan tanpa
//! kernel, tes di sini memerlukan Linux dengan Landlock + seccomp; CI
//! menjalankannya lewat job `vir-sandbox` (`cargo test -p vir-sandbox --features
//! sandbox`). Setiap tes memeriksa kemampuan kernel lebih dulu dan melaporkan
//! `SKIP: <alasan>` bila tidak tersedia — tidak ada tes yang lulus diam-diam
//! tanpa benar-benar menguji.
//!
//! Anak proses adalah biner helper `vir-iso` (bukan program sistem) supaya
//! "alokasi memori", "percabangan", dan "syscall terlarang" dapat dipicu tepat.
//! Sebagian tes punya pembanding tanpa konfinemen agar kegagalan karena sandbox
//! tidak tertukar dengan biner yang rusak.
//!
//! Berkas ini hanya dikompilasi saat fitur `sandbox` **dan** `iso-helper` aktif
//! (job CI `vir-sandbox`), karena biner bantuan `vir-iso` memerlukan fitur
//! `iso-helper` (lihat `Cargo.toml`).

#![cfg(all(feature = "sandbox", feature = "iso-helper"))]

mod common;

use std::time::{Duration, Instant};

use vir_sandbox::limits::ResourceLimits;
use vir_sandbox::{Policy, run};

use common::{
    TempDir, capabilities, confinement_error, field, helper, landlock_abi, landlock_at_least,
    policy_in, run_helper, seccomp_available,
};

/// Lewati tes dengan alasan yang terlihat di keluaran (bukan lulus diam-diam).
///
/// Alasan selalu disertai ringkasan kemampuan kernel agar log CI menunjukkan
/// *apa* yang hilang, bukan hanya "dilewati".
macro_rules! skip_unless {
    ($cond:expr, $($arg:tt)*) => {
        if !$cond {
            eprintln!("SKIP: {} [{}]", format!($($arg)*), capabilities());
            return;
        }
    };
}

// ---------------------------------------------------------------------------
// 1. Loop tak hingga: timeout dinding membunuh proses
// ---------------------------------------------------------------------------

#[test]
fn loop_tak_hingga_dibunuh_timeout_dinding() {
    let dir = TempDir::new("loop");
    let policy = Policy::default()
        .with_workdir(dir.path())
        .with_timeout(Duration::from_millis(500));
    let start = Instant::now();
    let out = run(helper(), &["spin"], &policy).expect("anak dapat dijalankan");

    assert!(
        out.timed_out,
        "loop tak hingga harus ditandai timeout (exit {:?})",
        out.exit_code
    );
    assert!(
        out.exit_code.is_some(),
        "proses harus benar-benar mati, bukan hanya ditandai"
    );
    assert_ne!(
        out.exit_code,
        Some(0),
        "proses yang dibunuh timeout tidak boleh keluar sukses"
    );
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "timeout harus membunuh, bukan menunggu proses selesai"
    );
}

/// Anak menelurkan cucu yang tidur; `kill_group` harus mematikan keduanya.
///
/// Tanpa `setpgid` + `kill(-pgid, SIGKILL)`, cucu selamat dan tetap memegang
/// pipa keluaran sehingga tes akan menggantung atau mencetak `CHILD-ALIVE`.
#[test]
fn timeout_membunuh_seluruh_grup_proses() {
    let dir = TempDir::new("grup");
    let policy = Policy::default()
        .with_workdir(dir.path())
        .with_timeout(Duration::from_millis(600));
    let start = Instant::now();
    let out = run(helper(), &["branch-sleep", "60"], &policy).expect("anak dapat dijalankan");

    assert!(
        out.timed_out,
        "cucu yang tidur harus memicu timeout (exit {:?})",
        out.exit_code
    );
    assert!(
        start.elapsed() < Duration::from_secs(8),
        "kill(-pgid) harus membunuh cucu juga, bukan menunggu 60 detik"
    );
    assert!(
        !out.stdout.contains("CHILD-ALIVE"),
        "cucu tidak boleh selamat dari pembunuhan grup"
    );
}

// ---------------------------------------------------------------------------
// 2. Memori: RLIMIT_AS menghentikan alokasi besar tanpa membunuh runner
// ---------------------------------------------------------------------------

#[test]
fn ledakan_memori_dibatasi_rlimit() {
    let dir = TempDir::new("mem");
    let limits = ResourceLimits {
        mem_bytes: Some(64 * 1024 * 1024),
        ..ResourceLimits::default()
    };
    let policy = Policy::default()
        .with_workdir(dir.path())
        .with_limits(limits)
        .with_timeout(Duration::from_secs(20));
    let out = run(helper(), &["alloc", "512"], &policy).expect("anak dapat dijalankan");

    assert_eq!(
        field(&out.stdout, "MEM-LIMIT").as_deref(),
        Some("ALLOC-FAILED"),
        "alokasi 512 MiB harus gagal di bawah RLIMIT_AS 64 MiB (exit {:?}, stderr {:?})",
        out.exit_code,
        out.stderr
    );
    assert!(
        field(&out.stdout, "ALLOC").is_none(),
        "alokasi 512 MiB tidak boleh berhasil di bawah RLIMIT_AS 64 MiB"
    );
    assert!(
        out.exit_code.is_some(),
        "proses harus mati atau keluar dengan galat, tidak menggantung"
    );
    assert!(!out.timed_out, "ralat memori harus cepat, bukan timeout");
}

// ---------------------------------------------------------------------------
// 3. Ledakan proses: RLIMIT_NPROC membatasi percabangan
// ---------------------------------------------------------------------------

#[test]
fn ledakan_proses_dibatasi_rlimit_nproc() {
    skip_unless!(
        confinement_error().is_none(),
        "konfinemen anak tidak dapat dipasang"
    );
    let dir = TempDir::new("fork");
    // `nproc` = 16 memberi ruang untuk sebagian anak; sisanya ditolak kernel
    // dengan EAGAIN. Batas keras runner biasanya jauh lebih besar, sehingga
    // nilai ini benar-benar menurunkan pagu.
    let limits = ResourceLimits {
        nproc: Some(32),
        ..ResourceLimits::default()
    };
    let policy = Policy::default()
        .with_workdir(dir.path())
        .with_limits(limits)
        .with_timeout(Duration::from_secs(20));
    let out = run(helper(), &["fork", "64"], &policy).expect("anak dapat dijalankan");

    assert!(
        field(&out.stdout, "FORK-DONE").is_some(),
        "helper harus selesai melaporkan hasil (exit {:?}, stderr {:?})",
        out.exit_code,
        out.stderr
    );
    let forked: usize = field(&out.stdout, "FORKED")
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    if forked == 0 {
        eprintln!(
            "SKIP: fork 0 (runner UID dekat RLIMIT_NPROC); batas fork diuji di runner bersih"
        );
        return;
    }
    assert!(
        forked < 64,
        "RLIMIT_NPROC harus menolak sebagian fork (berhasil {forked}/64)"
    );
    assert!(!out.timed_out, "ledakan fork harus selesai sebelum timeout");
}

// ---------------------------------------------------------------------------
// 4. Tulis di luar direktori kerja: Landlock menolak
// ---------------------------------------------------------------------------

#[test]
fn tulis_di_luar_workdir_ditolak() {
    skip_unless!(
        landlock_abi().is_some(),
        "Landlock tidak tersedia; butuh kernel Linux 5.13+ (landlock_create_ruleset)"
    );
    let dir = TempDir::new("tulis");
    let policy = policy_in(&dir);
    let target = std::env::temp_dir().join(format!("vir-iso-luar-{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&target);

    let out = run_helper(&["write", target.to_str().unwrap_or("")], &policy);
    assert!(
        field(&out.stdout, "WRITE").is_none(),
        "tulis ke luar workdir harus gagal (exit {:?}, stderr {:?})",
        out.exit_code,
        out.stderr
    );
    assert!(
        !target.exists(),
        "berkas di luar workdir tidak boleh tercipta: {}",
        target.display()
    );

    // Pembanding: tulis di dalam workdir harus berhasil, membuktikan penolakan
    // di atas berasal dari Landlock dan bukan dari biner yang rusak.
    let dalam = dir.file("di-dalam.txt");
    let out_kontrol = run_helper(&["write", dalam.to_str().unwrap_or("")], &policy);
    assert!(
        field(&out_kontrol.stdout, "WRITE").is_some(),
        "tulis di dalam workdir harus berhasil (stderr {:?})",
        out_kontrol.stderr
    );
    assert!(
        dalam.exists(),
        "berkas di dalam workdir harus tercipta: {}",
        dalam.display()
    );
}

// ---------------------------------------------------------------------------
// 5. Jaringan: Landlock menolak TCP
// ---------------------------------------------------------------------------

#[test]
fn jaringan_tcp_ditolak() {
    let abi = landlock_abi().unwrap_or(0);
    skip_unless!(
        landlock_at_least(4),
        "Landlock ABI {abi} < 4; penolakan TCP bind/connect belum ada (butuh kernel 6.7+)"
    );
    let dir = TempDir::new("net");
    let policy = policy_in(&dir);
    let out = run_helper(&["net", "127.0.0.1", "9"], &policy);

    assert_eq!(
        field(&out.stdout, "NET-RESULT").as_deref(),
        Some("BLOCKED"),
        "connect TCP harus ditolak Landlock dengan EPERM (exit {:?}, stderr {:?}, stdout {:?})",
        out.exit_code,
        out.stderr,
        out.stdout
    );
    assert!(
        field(&out.stdout, "NET").is_none(),
        "koneksi TCP tidak boleh berhasil di bawah kebijakan jaringan mati"
    );
}

// ---------------------------------------------------------------------------
// 6. Syscall terlarang: allowlist seccomp menolak
// ---------------------------------------------------------------------------

#[test]
fn syscall_terlarang_ditolak() {
    skip_unless!(
        seccomp_available(),
        "seccomp-bpf tidak tersedia; lihat pesan seccomp::detect_support()"
    );
    let dir = TempDir::new("sys");
    let policy = policy_in(&dir);
    let out = run_helper(&["bad-syscall"], &policy);

    // `ptrace` tidak ada di allowlist (`src/syscalls.rs`, ADR-006), jadi filter
    // membalas EPERM dan helper melaporkannya.
    assert_eq!(
        field(&out.stdout, "SYSCALL").as_deref(),
        Some("EPERM"),
        "syscall terlarang harus balik EPERM (exit {:?}, stderr {:?}, stdout {:?})",
        out.exit_code,
        out.stderr,
        out.stdout
    );

    // Pembanding tanpa seccomp: ptrace tidak boleh balik EPERM dari filter,
    // membuktikan EPERM di atas berasal dari sandbox.
    let out_kontrol = run(
        helper(),
        &["bad-syscall"],
        &Policy::default().with_seccomp(false),
    )
    .expect("anak dapat dijalankan");
    assert_ne!(
        field(&out_kontrol.stdout, "SYSCALL").as_deref(),
        Some("EPERM"),
        "tanpa filter seccomp, ptrace tidak boleh balik EPERM"
    );
}

// ---------------------------------------------------------------------------
// Pembanding: helper berjalan tanpa konfinemen
// ---------------------------------------------------------------------------

#[test]
fn helper_jalan_tanpa_konfinemen() {
    let dir = TempDir::new("kontrol");
    let policy = Policy::default()
        .with_workdir(dir.path())
        .with_landlock(false)
        .with_seccomp(false);
    let out = run_helper(&["selftest"], &policy);
    assert_eq!(
        field(&out.stdout, "SELFTEST").as_deref(),
        Some("OK"),
        "helper harus jalan tanpa sandbox (exit {:?}, stderr {:?})",
        out.exit_code,
        out.stderr
    );
}

/// Konfinemen anak harus benar-benar terpasang pada kernel yang didukung; bila
/// gagal, seluruh tes isolasi lain tidak dapat dipercaya. Karena itu kegagalan
/// di sini adalah kegagalan tes (bukan `SKIP`): runner CI dijamin mendukung.
#[test]
fn konfinemen_anak_terpasang() {
    skip_unless!(
        landlock_abi().is_some(),
        "Landlock tidak tersedia pada kernel ini; dilewati (bukan runner CI)"
    );
    let err = confinement_error();
    assert!(
        err.is_none(),
        "konfinemen Landlock+seccomp harus terpasang di anak: {}",
        err.unwrap_or_else(|| "-".into())
    );
}
