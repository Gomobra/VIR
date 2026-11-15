//! Tes `seccomp::describe` (F3.1) yang berjalan tanpa kernel.
//!
//! Nomor syscall yang ditolak diubah menjadi nama agar galat dapat dilacak.
//! Itu logika murni, jadi benar/tidaknya tidak bergantung pada kernel.

use vir_sandbox::seccomp::describe;

#[test]
fn describe_mengenali_syscall_dikenal() {
    let execve = i64::from(libc::SYS_execve);
    assert_eq!(describe(execve), format!("syscall {execve} (execve)"));
    let ptrace = i64::from(libc::SYS_ptrace);
    assert_eq!(describe(ptrace), format!("syscall {ptrace} (ptrace)"));
}

#[test]
fn describe_kembali_ke_nomor_bila_tak_dikenal() {
    assert_eq!(describe(-1), "syscall -1");
    assert_eq!(describe(i64::MAX), format!("syscall {}", i64::MAX));
}
