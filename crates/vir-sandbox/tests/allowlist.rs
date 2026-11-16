//! Tes kebijakan syscall (F3.1) yang berjalan tanpa kernel.
//!
//! [`vir_sandbox::syscalls::allowlist`] adalah data murni, jadi isi kebijakan
//! "syscall apa yang boleh jalan di dalam sandbox" dapat diperiksa di job CI
//! bawaan tanpa fitur `sandbox`. Yang diperiksa: allowlist terurut dan bebas
//! duplikat, syscall inti dan startup libc ada, syscall berbahaya tidak ada.

use vir_sandbox::syscalls::allowlist;

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
        libc::SYS_mmap,
        libc::SYS_mprotect,
        libc::SYS_brk,
        libc::SYS_exit_group,
    ] {
        assert!(
            list.contains(&required),
            "syscall {required} wajib diizinkan"
        );
    }
}

#[test]
fn syscall_startup_libc_diizinkan() {
    // Syscall yang dipakai pemuat dinamis dan libc saat proses mulai. Bila
    // salah satu hilang, program apa pun gagal dengan `EPERM` sebelum `main`.
    let list = allowlist();
    let mut wajib = vec![
        libc::SYS_ioctl,
        libc::SYS_mmap,
        libc::SYS_mprotect,
        libc::SYS_munmap,
        libc::SYS_mremap,
        libc::SYS_brk,
        libc::SYS_rt_sigaction,
        libc::SYS_rt_sigprocmask,
        libc::SYS_set_tid_address,
        libc::SYS_set_robust_list,
        libc::SYS_prlimit64,
        libc::SYS_getrandom,
        libc::SYS_fcntl,
        libc::SYS_getcwd,
        libc::SYS_readlinkat,
        libc::SYS_uname,
        libc::SYS_sched_getaffinity,
        libc::SYS_gettimeofday,
    ];
    #[cfg(not(target_os = "android"))]
    wajib.push(libc::SYS_newfstatat);
    #[cfg(target_arch = "x86_64")]
    wajib.extend([libc::SYS_arch_prctl, libc::SYS_readlink, libc::SYS_stat]);
    for nr in wajib {
        assert!(
            list.contains(&nr),
            "syscall {nr} wajib diizinkan agar proses libc dapat mulai"
        );
    }
}

#[test]
fn syscall_berbahaya_tidak_diizinkan() {
    // ADR-006: allowlist harus tetap menutup syscall yang bisa dipakai untuk
    // keluar dari konfinemen atau memata-matai proses lain.
    let list = allowlist();
    for nr in [
        libc::SYS_ptrace,
        libc::SYS_process_vm_readv,
        libc::SYS_io_uring_setup,
        libc::SYS_bpf,
        libc::SYS_mount,
        libc::SYS_pivot_root,
        libc::SYS_init_module,
        libc::SYS_perf_event_open,
        libc::SYS_userfaultfd,
        libc::SYS_add_key,
        libc::SYS_keyctl,
        libc::SYS_mq_open,
        libc::SYS_shmget,
        libc::SYS_semget,
        libc::SYS_unshare,
        libc::SYS_seccomp,
    ] {
        assert!(
            !list.contains(&nr),
            "syscall {nr} tidak boleh diizinkan (ADR-006)"
        );
    }
}
