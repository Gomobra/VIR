//! seccomp-bpf: allowlist syscall per proses anak (F3.1).
//!
//! Filter dipasang setelah `PR_SET_NO_NEW_PRIVS` dan setelah aturan Landlock
//! pada `pre_exec`, tepat sebelum `exec` (lihat [`crate::limits`] untuk urutan
//! lengkap). Semua syscall yang tidak ada di allowlist ditolak dengan
//! `EPERM`, sehingga galat muncul di sisi program (`Operation not permitted`)
//! alih-alih proses dibunuh diam-diam; itu membuat kegagalan dapat dilacak.
//! Pilihan ini (allowlist + errno, bukan deny-list + SIGSYS seperti Sandlock)
//! dibahas di `docs/adr/ADR-006-sandbox.md`.
//!
//! Allowlist disusun agar menampung CPython (REPL F3.2) dan inti libc: proses,
//! memori, berkas, soket (bind/connect tetap diblokir Landlock), waktu, dan
//! sinyal. Syscall yang tidak dipakai sandbox sengaja tidak diizinkan:
//! `ptrace`, `process_vm_readv`/`writev`, `io_uring_*`, `mount*`,
//! `pivot_root`, `init_module`/`finit_module`, `bpf`, `kexec_load`,
//! `perf_event_open`, `userfaultfd`, `keyctl`/`add_key`, `mq_*`, `shm*`,
//! `sem*`, `fanotify_*`, dan `unshare`.
//!
//! Kernel tanpa dukungan seccomp-bpf ditolak dengan galat jelas lewat
//! [`detect_support`]; filter tidak pernah dilewati diam-diam.

use vir_common::VirError;

/// Nama syscall untuk pesan galat; `None` bila nomornya tak dikenal.
fn syscall_name(nr: i64) -> Option<&'static str> {
    const TABLE: &[(i64, &str)] = &[
        (libc::SYS_execve, "execve"),
        (libc::SYS_execveat, "execveat"),
        (libc::SYS_openat, "openat"),
        (libc::SYS_clone, "clone"),
        (libc::SYS_clone3, "clone3"),
        (libc::SYS_mmap, "mmap"),
        (libc::SYS_mprotect, "mprotect"),
        (libc::SYS_socket, "socket"),
        (libc::SYS_ptrace, "ptrace"),
        (libc::SYS_io_uring_setup, "io_uring_setup"),
        (libc::SYS_process_vm_readv, "process_vm_readv"),
    ];
    TABLE
        .iter()
        .find(|(nr_table, _)| *nr_table == nr)
        .map(|(_, name)| *name)
}

/// Terjemahkan nomor syscall yang ditolak menjadi nama bila dikenali.
///
/// Dipakai untuk memperkaya pesan galat yang dilihat pengguna.
pub fn describe(nr: i64) -> String {
    match syscall_name(nr) {
        Some(name) => format!("syscall {nr} ({name})"),
        None => format!("syscall {nr}"),
    }
}

/// Deteksi dukungan seccomp-bpf dengan menguji pasang filter lewat
/// `prctl(PR_GET_SECCOMP)`, yang melaporkan status seccomp pemanggil.
///
/// Mengembalikan galat jelas bila kernel tidak mendukung (kernel < 3.5,
/// `CONFIG_SECCOMP` mati, atau seccomp-bpf dimatikan lewat `seccomp=off`).
pub fn detect_support() -> Result<(), VirError> {
    #[cfg(feature = "sandbox")]
    {
        // SAFETY: `prctl` dengan `PR_GET_SECCOMP` hanya membaca status dan
        // tidak menulis ke memori pengguna.
        let ret = unsafe {
            libc::prctl(
                libc::PR_GET_SECCOMP,
                0 as libc::c_ulong,
                0 as libc::c_ulong,
                0 as libc::c_ulong,
                0 as libc::c_ulong,
            )
        };
        if ret == 0 {
            return Ok(());
        }
        let errno = std::io::Error::last_os_error();
        Err(VirError::Internal(format!(
            "seccomp-bpf tidak tersedia (errno {errno}); butuh kernel Linux 3.5+ dengan \
             CONFIG_SECCOMP dan CONFIG_SECCOMP_FILTER aktif"
        )))
    }
    #[cfg(not(feature = "sandbox"))]
    {
        Err(VirError::Internal(
            "fitur 'sandbox' tidak aktif; filter seccomp tidak tersedia".into(),
        ))
    }
}

/// Bangun allowlist lengkap untuk arsitektur proses ini.
///
/// Daftar syscall ada di [`crate::syscalls`] agar modul ini fokus pada
/// penyusunan dan pemasangan filter.
#[cfg(feature = "sandbox")]
fn allowlist() -> Vec<i64> {
    crate::syscalls::allowlist()
}

/// Susun program BPF: izinkan syscall di allowlist, tolak sisanya dengan `EPERM`.
#[cfg(feature = "sandbox")]
fn build_filter() -> Result<seccompiler::BpfProgram, VirError> {
    use std::collections::BTreeMap;
    use std::convert::TryInto;

    let arch = std::env::consts::ARCH
        .try_into()
        .map_err(|e| VirError::Internal(format!("seccomp: {e}")))?;
    // Aturan tanpa kondisi berarti "izinkan syscall ini apa pun argumennya".
    let rules: BTreeMap<i64, Vec<seccompiler::SeccompRule>> =
        allowlist().into_iter().map(|nr| (nr, Vec::new())).collect();
    let filter = seccompiler::SeccompFilter::new(
        rules,
        // Syscall yang tidak ada di allowlist ditolak dengan galat; proses tetap
        // hidup sehingga kegagalan terlihat di keluaran program.
        seccompiler::SeccompAction::Errno(libc::EPERM as u32),
        seccompiler::SeccompAction::Allow,
        arch,
    )
    .map_err(|e| VirError::Internal(format!("seccomp: {e}")))?;
    let program: seccompiler::BpfProgram = filter
        .try_into()
        .map_err(|e| VirError::Internal(format!("seccomp: {e}")))?;
    Ok(program)
}

/// Pasang filter seccomp-bpf pada proses saat ini (dipanggil di anak).
///
/// Wajib dipanggil setelah `PR_SET_NO_NEW_PRIVS` dipasang, dan harus menjadi
/// langkah terakhir sebelum `exec`. Bila kernel tidak mendukung, kembalikan
/// galat jelas alih-alih melanjutkan tanpa filter.
pub fn confine() -> Result<(), VirError> {
    #[cfg(feature = "sandbox")]
    {
        detect_support()?;
        let program = build_filter()?;
        seccompiler::apply_filter(&program)
            .map_err(|e| VirError::Internal(format!("seccomp: {e}")))?;
        Ok(())
    }
    #[cfg(not(feature = "sandbox"))]
    {
        Err(VirError::Internal(
            "fitur 'sandbox' tidak aktif; filter seccomp tidak tersedia".into(),
        ))
    }
}
