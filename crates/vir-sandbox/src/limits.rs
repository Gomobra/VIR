//! Batas proses (rlimit) + process group (F3.1).

use nix::sys::resource::{Resource, setrlimit};
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
        setrlimit(Resource::RLIMIT_CPU, cpu_secs, cpu_secs).map_err(to_io)?;
        setrlimit(Resource::RLIMIT_AS, mem_bytes, mem_bytes).map_err(to_io)?;
        setrlimit(Resource::RLIMIT_NOFILE, nofile, nofile).map_err(to_io)?;
        let _ = nproc;
        Ok(())
    })
}

fn to_io(e: nix::Error) -> std::io::Error {
    std::io::Error::other(e.to_string())
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
