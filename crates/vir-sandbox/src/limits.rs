//! Batas proses (rlimit) + process group (F3.1).

use std::os::unix::process::CommandExt;

/// Terapkan rlimit + setpgid di anak proses (setelah fork, sebelum exec).
pub fn apply_pre_exec(
    cpu_secs: u64,
    mem_bytes: u64,
    nofile: u64,
    nproc: u64,
) -> Box<dyn FnMut() -> Result<(), std::io::Error> + Send + Sync> {
    Box::new(move || {
        // process group sendiri
        unsafe {
            libc::setpgid(0, 0);
        }
        // rlimit
        set_rlimit(libc::RLIMIT_CPU, cpu_secs, cpu_secs)?;
        set_rlimit(libc::RLIMIT_AS, mem_bytes, mem_bytes)?;
        set_rlimit(libc::RLIMIT_NOFILE, nofile, nofile)?;
        set_rlimit(libc::RLIMIT_NPROC, nproc, nproc)?;
        Ok(())
    })
}

fn set_rlimit(resource: libc::c_int, soft: u64, hard: u64) -> Result<(), std::io::Error> {
    let lim = libc::rlimit {
        rlim_cur: soft as libc::rlim_t,
        rlim_max: hard as libc::rlim_t,
    };
    let rc = unsafe { libc::setrlimit(resource, &lim) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Terapkan kebijakan ke Command sebelum spawn.
pub fn confine_command(
    cmd: &mut std::process::Command,
    cpu: u64,
    mem: u64,
    nofile: u64,
    nproc: u64,
) {
    let f = apply_pre_exec(cpu, mem, nofile, nproc);
    unsafe {
        cmd.pre_exec(f);
    }
}
