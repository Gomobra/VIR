//! Allowlist syscall untuk filter seccomp-bpf (F3.1).
//!
//! Daftar ini adalah inti kebijakan syscall VIR: hanya syscall yang tercantum
//! di sini yang boleh dijalankan proses ber-konfinemen; sisanya ditolak dengan
//! `EPERM` oleh [`crate::seccomp`]. Lihat ADR-006 untuk pilihan allowlist
//! (bukan deny-list) dan implikasinya.
//!
//! Penyusunan:
//!
//! - [`COMMON`] berlaku di semua arsitektur yang didukung `seccompiler`
//!   (x86_64 dan aarch64) dan semua libc (glibc, musl, Bionic);
//! - [`allowlist`] menambahkan syscall arsitektur-spesifik yang hanya ada pada
//!   sebagian libc, mis. `newfstatat` (tidak ada di Bionic) dan `open`
//!   (hanya x86_64).
//!
//! Yang sengaja **tidak** diizinkan: `ptrace`, `process_vm_readv`/`writev`,
//! `io_uring_*`, `mount*`, `pivot_root`, `init_module`/`finit_module`, `bpf`,
//! `kexec_load`, `perf_event_open`, `userfaultfd`, `keyctl`/`add_key`, `mq_*`,
//! `shm*`, `sem*`, `fanotify_*`, `unshare`, dan `seccomp` itu sendiri
//! (proses tidak boleh memasang filter baru).

/// Syscall yang diizinkan pada semua arsitektur yang didukung crate.
#[cfg(feature = "sandbox")]
const COMMON: &[i64] = &[
    libc::SYS_accept,
    libc::SYS_accept4,
    libc::SYS_bind,
    libc::SYS_brk,
    libc::SYS_clock_getres,
    libc::SYS_clock_gettime,
    libc::SYS_clock_nanosleep,
    libc::SYS_clone,
    libc::SYS_clone3,
    libc::SYS_close,
    libc::SYS_close_range,
    libc::SYS_connect,
    libc::SYS_copy_file_range,
    libc::SYS_dup,
    libc::SYS_dup3,
    libc::SYS_epoll_create1,
    libc::SYS_epoll_ctl,
    libc::SYS_epoll_pwait,
    libc::SYS_epoll_pwait2,
    libc::SYS_eventfd2,
    libc::SYS_execve,
    libc::SYS_execveat,
    libc::SYS_exit,
    libc::SYS_exit_group,
    libc::SYS_faccessat,
    libc::SYS_faccessat2,
    libc::SYS_fallocate,
    libc::SYS_fchdir,
    libc::SYS_fchmod,
    libc::SYS_fchmodat,
    libc::SYS_fchown,
    libc::SYS_fchownat,
    libc::SYS_fcntl,
    libc::SYS_fdatasync,
    libc::SYS_fgetxattr,
    libc::SYS_flistxattr,
    libc::SYS_flock,
    libc::SYS_fremovexattr,
    libc::SYS_fsetxattr,
    libc::SYS_fsync,
    libc::SYS_futex,
    libc::SYS_futex_waitv,
    libc::SYS_getcpu,
    libc::SYS_getcwd,
    libc::SYS_getdents64,
    libc::SYS_getegid,
    libc::SYS_geteuid,
    libc::SYS_getgid,
    libc::SYS_getgroups,
    libc::SYS_getpeername,
    libc::SYS_getpgid,
    libc::SYS_getpid,
    libc::SYS_getppid,
    libc::SYS_getpriority,
    libc::SYS_getrandom,
    libc::SYS_getresgid,
    libc::SYS_getresuid,
    libc::SYS_getrlimit,
    libc::SYS_getrusage,
    libc::SYS_getsid,
    libc::SYS_getsockname,
    libc::SYS_getsockopt,
    libc::SYS_gettid,
    libc::SYS_gettimeofday,
    libc::SYS_getuid,
    libc::SYS_getxattr,
    libc::SYS_lgetxattr,
    libc::SYS_linkat,
    libc::SYS_listen,
    libc::SYS_listxattr,
    libc::SYS_llistxattr,
    libc::SYS_lremovexattr,
    libc::SYS_lseek,
    libc::SYS_lsetxattr,
    libc::SYS_madvise,
    libc::SYS_membarrier,
    libc::SYS_memfd_create,
    libc::SYS_mincore,
    libc::SYS_mkdirat,
    libc::SYS_mknodat,
    libc::SYS_mlock,
    libc::SYS_mlock2,
    libc::SYS_mlockall,
    libc::SYS_mmap,
    libc::SYS_mprotect,
    libc::SYS_mremap,
    libc::SYS_msync,
    libc::SYS_munlock,
    libc::SYS_munlockall,
    libc::SYS_munmap,
    libc::SYS_nanosleep,
    libc::SYS_openat,
    libc::SYS_openat2,
    libc::SYS_pipe2,
    libc::SYS_ppoll,
    libc::SYS_prctl,
    libc::SYS_pread64,
    libc::SYS_preadv,
    libc::SYS_preadv2,
    libc::SYS_prlimit64,
    libc::SYS_pselect6,
    libc::SYS_pwrite64,
    libc::SYS_pwritev,
    libc::SYS_pwritev2,
    libc::SYS_read,
    libc::SYS_readahead,
    libc::SYS_readlinkat,
    libc::SYS_readv,
    libc::SYS_recvfrom,
    libc::SYS_recvmmsg,
    libc::SYS_recvmsg,
    libc::SYS_renameat2,
    libc::SYS_restart_syscall,
    libc::SYS_rt_sigaction,
    libc::SYS_rt_sigpending,
    libc::SYS_rt_sigprocmask,
    libc::SYS_rt_sigreturn,
    libc::SYS_rt_sigsuspend,
    libc::SYS_rt_sigtimedwait,
    libc::SYS_sched_get_priority_max,
    libc::SYS_sched_get_priority_min,
    libc::SYS_sched_getaffinity,
    libc::SYS_sched_getparam,
    libc::SYS_sched_getscheduler,
    libc::SYS_sched_rr_get_interval,
    libc::SYS_sched_setaffinity,
    libc::SYS_sched_setparam,
    libc::SYS_sched_setscheduler,
    libc::SYS_sched_yield,
    libc::SYS_sendmmsg,
    libc::SYS_sendmsg,
    libc::SYS_sendto,
    libc::SYS_set_robust_list,
    libc::SYS_set_tid_address,
    libc::SYS_setfsgid,
    libc::SYS_setfsuid,
    libc::SYS_setgid,
    libc::SYS_setgroups,
    libc::SYS_setitimer,
    libc::SYS_setpgid,
    libc::SYS_setpriority,
    libc::SYS_setregid,
    libc::SYS_setresgid,
    libc::SYS_setresuid,
    libc::SYS_setreuid,
    libc::SYS_setsid,
    libc::SYS_setsockopt,
    libc::SYS_setuid,
    libc::SYS_shutdown,
    libc::SYS_sigaltstack,
    libc::SYS_socket,
    libc::SYS_socketpair,
    libc::SYS_splice,
    libc::SYS_statx,
    libc::SYS_symlinkat,
    libc::SYS_sync,
    libc::SYS_sync_file_range,
    libc::SYS_syncfs,
    libc::SYS_sysinfo,
    libc::SYS_tgkill,
    libc::SYS_timer_create,
    libc::SYS_timer_delete,
    libc::SYS_timer_getoverrun,
    libc::SYS_timer_gettime,
    libc::SYS_timer_settime,
    libc::SYS_timerfd_create,
    libc::SYS_timerfd_gettime,
    libc::SYS_timerfd_settime,
    libc::SYS_times,
    libc::SYS_tkill,
    libc::SYS_umask,
    libc::SYS_uname,
    libc::SYS_unlinkat,
    libc::SYS_utimensat,
    libc::SYS_vmsplice,
    libc::SYS_wait4,
    libc::SYS_waitid,
    libc::SYS_write,
    libc::SYS_writev,
];

/// Syscall tambahan arsitektur-spesifik yang tidak ada di daftar umum.
///
/// Setiap kombinasi arsitektur/SCM memilih daftarnya sendiri karena nama
/// konstanta berbeda antar libc (mis. Bionic tidak punya `newfstatat`).
#[cfg(all(
    feature = "sandbox",
    target_arch = "x86_64",
    not(target_os = "android")
))]
const ARCH_EXTRA: &[i64] = &[
    libc::SYS_access,
    libc::SYS_chmod,
    libc::SYS_chown,
    libc::SYS_creat,
    libc::SYS_dup2,
    libc::SYS_fadvise64,
    libc::SYS_fork,
    libc::SYS_fstat,
    libc::SYS_link,
    libc::SYS_lstat,
    libc::SYS_mkdir,
    libc::SYS_mknod,
    libc::SYS_newfstatat,
    libc::SYS_open,
    libc::SYS_poll,
    libc::SYS_readlink,
    libc::SYS_rename,
    libc::SYS_rmdir,
    libc::SYS_rseq,
    libc::SYS_sendfile,
    libc::SYS_stat,
    libc::SYS_statfs,
    libc::SYS_symlink,
    libc::SYS_time,
    libc::SYS_truncate,
    libc::SYS_unlink,
    libc::SYS_utime,
    libc::SYS_utimes,
    libc::SYS_vfork,
];

/// Syscall lama pada Bionic x86_64 (nama 32-bit yang tetap ada).
#[cfg(all(feature = "sandbox", target_arch = "x86_64", target_os = "android"))]
const ARCH_EXTRA: &[i64] = &[
    libc::SYS_access,
    libc::SYS_chmod,
    libc::SYS_chown,
    libc::SYS_creat,
    libc::SYS_dup2,
    libc::SYS_fork,
    libc::SYS_link,
    libc::SYS_lstat,
    libc::SYS_mkdir,
    libc::SYS_mknod,
    libc::SYS_open,
    libc::SYS_poll,
    libc::SYS_readlink,
    libc::SYS_rename,
    libc::SYS_rmdir,
    libc::SYS_stat,
    libc::SYS_symlink,
    libc::SYS_time,
    libc::SYS_unlink,
    libc::SYS_utime,
    libc::SYS_utimes,
    libc::SYS_vfork,
];

/// Syscall aarch64 yang tidak disediakan Bionic.
#[cfg(all(
    feature = "sandbox",
    target_arch = "aarch64",
    not(target_os = "android")
))]
const ARCH_EXTRA: &[i64] = &[
    libc::SYS_fadvise64,
    libc::SYS_fstat,
    libc::SYS_fstatfs,
    libc::SYS_newfstatat,
    libc::SYS_rseq,
    libc::SYS_sendfile,
    libc::SYS_statfs,
    libc::SYS_truncate,
];

/// Bionic aarch64 tidak memerlukan syscall tambahan.
#[cfg(all(feature = "sandbox", target_arch = "aarch64", target_os = "android"))]
const ARCH_EXTRA: &[i64] = &[];

/// Gabungkan daftar umum dan tambahan arsitektur menjadi satu allowlist.
///
/// Diurutkan dan bebas duplikat agar hasilnya deterministik; filter seccomp
/// hanya butuh himpunan, tetapi urutan tetap memudahkan pengujian.
#[cfg(feature = "sandbox")]
pub fn allowlist() -> Vec<i64> {
    let mut list: Vec<i64> = COMMON.to_vec();
    list.extend_from_slice(ARCH_EXTRA);
    list.sort_unstable();
    list.dedup();
    list
}

#[cfg(all(test, feature = "sandbox"))]
mod tests {
    use super::*;

    #[test]
    fn allowlist_terurut_tanpa_duplikat() {
        let list = allowlist();
        assert!(!list.is_empty(), "allowlist tidak boleh kosong");
        let mut sorted = list.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(list, sorted, "allowlist harus terurut dan bebas duplikat");
    }

    #[test]
    fn syscall_inti_ada() {
        let list = allowlist();
        for required in [
            libc::SYS_execve,
            libc::SYS_openat,
            libc::SYS_read,
            libc::SYS_write,
        ] {
            assert!(
                list.contains(&required),
                "syscall {required} wajib diizinkan"
            );
        }
    }
}
