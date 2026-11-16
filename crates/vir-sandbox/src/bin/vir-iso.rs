//! Pembantu untuk `tests/isolation.rs` (F3.1 butir 7): biner yang dipicu tes.
//!
//! Tes isolasi butuh proses dengan perilaku yang dapat diprediksi. Memakai biner
//! ini (bukan `python3`, `yes`, atau `sh`) membuat tes identik di runner CI, host
//! pengembang, dan container: tidak ada versi program yang berbeda, tidak ada
//! builtin shell yang menyamarkan galat, dan setiap kegagalan dapat dilaporkan
//! lewat penanda `NAMA=nilai` di stdout.
//!
//! Subperintah:
//!
//! | subperintah         | perilaku |
//! |---|---|
//! | `selftest`          | cetak `SELFTEST=OK` |
//! | `spin`              | putaran tak hingga tanpa alokasi |
//! | `branch-sleep <s>`  | lahirkan anak `sleep <s>`, lalu tidur `<s>` juga |
//! | `alloc <MiB>`       | alokasi + sentuh memori; `ALLOC=OK` atau `MEM-LIMIT=ALLOC-FAILED` |
//! | `fork <n>`          | `fork` berulang; `FORKED`, `FORK-FAILED`, `FORK-DONE` |
//! | `write <path>`      | tulis satu bita; `WRITE=OK` atau `WRITE-ERROR` |
//! | `net <host> <port>` | `connect` TCP; `NET=OK` atau `NET-RESULT=BLOCKED/ERROR` |
//! | `bad-syscall`       | panggil `ptrace(PTRACE_TRACEME)` yang tak ada di allowlist |
//!
//! Proses selalu keluar `0` bila berhasil melaporkan hasil (termasuk kegagalan
//! yang diharapkan tes), dan `2` bila biner salah dipakai. Dengan begitu tes
//! dapat membedakan "ditolak kernel" dari "biner rusak".
//!
//! Binari ini hanya dibangun saat fitur `iso-helper` aktif (lihat
//! `Cargo.toml` + `build.rs`), sehingga job `nextest` tanpa fitur tidak
//! menyalakan `CARGO_BIN_EXE_vir-iso`. Job `vir-sandbox` memakai
//! `--features sandbox,iso-helper` untuk menjalankan tes isolasi.

use std::io::Write;
use std::net::{TcpStream, ToSocketAddrs};
use std::os::unix::process::CommandExt;
use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("selftest") => selftest(),
        Some("spin") => spin(),
        Some("branch-sleep") => branch_sleep(arg_secs(&args)),
        Some("alloc") => alloc(arg_mib(&args)),
        Some("fork") => fork_bomb(arg_count(&args)),
        Some("write") => write_file(&args),
        Some("net") => net(&args),
        Some("bad-syscall") => bad_syscall(),
        other => {
            eprintln!("subperintah tak dikenal: {other:?}");
            2
        }
    };
    std::process::exit(i32::from(code));
}

/// Argumen pertama setelah subperintah, atau string kosong.
fn arg1(args: &[String]) -> String {
    args.get(1).cloned().unwrap_or_default()
}

fn arg_secs(args: &[String]) -> u64 {
    arg1(args).parse().unwrap_or(5)
}

fn arg_mib(args: &[String]) -> usize {
    arg1(args).parse().unwrap_or(64)
}

fn arg_count(args: &[String]) -> usize {
    arg1(args).parse().unwrap_or(16)
}

/// Cetak satu baris penanda dan segarkan agar tidak tertahan buffer.
fn out(line: &str) {
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
}

fn selftest() -> u8 {
    out("SELFTEST=OK");
    0
}

/// Putaran tak hingga tanpa alokasi: dijamin terbunuh oleh timeout dinding.
fn spin() -> u8 {
    let mut x: u64 = 0;
    loop {
        x = x.wrapping_add(1);
        std::hint::black_box(x);
    }
}

/// Lairkan anak yang tidur `<s>` detik, lalu induk tidur juga.
///
/// Dipakai untuk membuktikan `kill(-pgid, SIGKILL)` mematikan seluruh grup: bila
/// hanya induk dibunuh, anak tetap hidup dan sempat mencetak `CHILD-ALIVE`.
fn branch_sleep(secs: u64) -> u8 {
    // SAFETY: `fork` pada proses satu thread tanpa alokasi di antaranya; anak
    // langsung mengganti citra lewat `exec`, dan kegagalan apa pun mengakhiri
    // anak tanpa kembali ke kode Rust.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        eprintln!("fork gagal: {}", std::io::Error::last_os_error());
        return 2;
    }
    if pid == 0 {
        let err = std::process::Command::new("sleep")
            .arg(secs.to_string())
            .exec();
        eprintln!("exec gagal: {err}");
        std::process::exit(2);
    }
    out("PARENT-ALIVE");
    std::thread::sleep(Duration::from_secs(secs));
    out("CHILD-ALIVE");
    0
}

/// Alokasi `mib` MiB lalu sentuh setiap halaman.
///
/// Kegagalan dilaporkan sebagai `MEM-LIMIT=ALLOC-FAILED` agar tes tahu batas
/// rlimit yang bekerja, bukan biner yang salah. `try_reserve_exact` dipakai
/// supaya kegagalan menjadi `Err` (bukan abort) di bawah `RLIMIT_AS`.
fn alloc(mib: usize) -> u8 {
    let bytes = mib.saturating_mul(1024 * 1024);
    let mut buf = Vec::<u8>::new();
    if buf.try_reserve_exact(bytes).is_err() {
        out("MEM-LIMIT=ALLOC-FAILED");
        return 0;
    }
    buf.resize(bytes, 0x5a);
    // Sentuh tiap halaman agar memori benar-benar dialokasikan, bukan hanya
    // cadangan alamat virtual.
    for chunk in buf.chunks_mut(4096) {
        chunk[0] = 0x11;
    }
    out("MEM-LIMIT=NONE");
    out("ALLOC=OK");
    0
}

/// `fork` berulang sampai `n` berhasil atau kernel menolak.
///
/// Anak langsung `_exit` tanpa alokasi; induk memanen semuanya lalu melaporkan
/// jumlah percobaan yang berhasil.
fn fork_bomb(n: usize) -> u8 {
    let mut forked = 0usize;
    let mut failed = 0usize;
    for _ in 0..n {
        // SAFETY: `fork` pada proses satu thread; anak hanya memanggil `_exit`.
        let pid = unsafe { libc::fork() };
        if pid < 0 {
            failed += 1;
            continue;
        }
        if pid == 0 {
            unsafe { libc::_exit(0) };
        }
        forked += 1;
    }
    for _ in 0..forked {
        let mut status = 0;
        // SAFETY: memanen anak yang baru dibuat; `status` hanya ditulis kernel.
        unsafe {
            libc::waitpid(-1, &mut status, 0);
        }
    }
    out(&format!("FORKED={forked}"));
    out(&format!("FORK-FAILED={failed}"));
    out("FORK-DONE=OK");
    0
}

/// Tulis satu bita ke `path`.
fn write_file(args: &[String]) -> u8 {
    let path = arg1(args);
    match std::fs::File::create(&path).and_then(|mut f| f.write_all(b"x")) {
        Ok(()) => {
            out("WRITE=OK");
            0
        }
        Err(e) => {
            out(&format!("WRITE-ERROR={e}"));
            0
        }
    }
}

/// Koneksi TCP ke `host:port`.
///
/// Landlock ABI 4+ menolak `connect` TCP dengan `EPERM`, yang dilaporkan sebagai
/// `NET-RESULT=BLOCKED`. Galat lain (mis. `ENETUNREACH`) dilaporkan sebagai
/// `ERROR` agar tidak salah diartikan sebagai isolasi.
fn net(args: &[String]) -> u8 {
    let host = args.get(1).cloned().unwrap_or_else(|| "127.0.0.1".into());
    let port = args.get(2).cloned().unwrap_or_else(|| "9".into());
    let addrs = match format!("{host}:{port}").to_socket_addrs() {
        Ok(a) => a,
        Err(e) => {
            out("NET-RESULT=ERROR");
            out(&format!("NET-ERROR=resolve: {e}"));
            return 0;
        }
    };
    let Some(addr) = addrs.into_iter().next() else {
        out("NET-RESULT=ERROR");
        out("NET-ERROR=resolve: alamat kosong");
        return 0;
    };
    match TcpStream::connect_timeout(&addr, Duration::from_secs(2)) {
        Ok(_) => {
            out("NET=OK");
            0
        }
        Err(e) => {
            let blocked = e.raw_os_error() == Some(libc::EPERM);
            out(&format!(
                "NET-RESULT={}",
                if blocked { "BLOCKED" } else { "ERROR" }
            ));
            out(&format!("NET-ERROR={e}"));
            0
        }
    }
}

/// Panggil `ptrace(PTRACE_TRACEME)` yang sengaja tidak ada di allowlist.
///
/// Di bawah filter seccomp: `EPERM`. Tanpa filter: `ptrace` umumnya berhasil
/// (proses boleh menelusuri dirinya) atau balik galat lain — bukan `EPERM`.
fn bad_syscall() -> u8 {
    // SAFETY: hanya meminta `PTRACE_TRACEME` pada diri sendiri; argumen lain
    // diabaikan kernel untuk permintaan ini.
    let ret = unsafe { libc::ptrace(libc::PTRACE_TRACEME as _, 0, 0, 0) };
    if ret == 0 {
        out("SYSCALL=OK");
        return 0;
    }
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(-1);
    out(&format!("ERRNO={errno}"));
    let label = if errno == libc::EPERM {
        "EPERM"
    } else {
        "ERROR"
    };
    out(&format!("SYSCALL={label}"));
    0
}
