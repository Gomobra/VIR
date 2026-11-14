//! Batas proses (rlimit) dan process group untuk anak proses (F3.1).
//!
//! Urutan konfinemen mengikuti Sandlock: `setpgid` → `chdir`
//! (`Command::current_dir`) → `PR_SET_NO_NEW_PRIVS` + aturan Landlock
//! (`landlock::confine`) → `exec`. `setpgid` di sini membuat anak menjadi
//! pemimpin grup prosesnya sendiri sehingga timeout dapat membunuh seluruh
//! turunan dengan `kill(-pgid, SIGKILL)`.

use nix::sys::resource::{Resource, getrlimit, setrlimit};
use std::os::unix::process::CommandExt;

/// Sumber daya yang dibatasi lewat `setrlimit` pada anak proses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Batas waktu CPU (detik). `None` berarti tanpa batas.
    pub cpu_secs: Option<u64>,
    /// Batas ukuran ruang alamat virtual (bita). `None` berarti tanpa batas.
    pub mem_bytes: Option<u64>,
    /// Batas jumlah deskriptor berkas. `None` berarti tanpa batas.
    pub nofile: Option<u64>,
    /// Batas jumlah proses untuk UID nyata. `None` berarti tanpa batas.
    pub nproc: Option<u64>,
}

impl ResourceLimits {
    /// Batas aman untuk pemakaian umum (CPU 10 s, 512 MiB, 1024 fd, 1024 proses).
    pub fn standard() -> Self {
        Self {
            cpu_secs: Some(10),
            mem_bytes: Some(512 * 1024 * 1024),
            nofile: Some(1024),
            nproc: Some(1024),
        }
    }
}

/// Bangun closure `pre_exec` yang menetapkan `setpgid` dan rlimit.
pub fn apply_pre_exec(
    limits: ResourceLimits,
) -> Box<dyn FnMut() -> Result<(), std::io::Error> + Send + Sync> {
    Box::new(move || {
        // SAFETY: `setpgid(0, 0)` hanya mengubah pgid proses saat ini.
        unsafe {
            libc::setpgid(0, 0);
        }
        if let Some(secs) = limits.cpu_secs {
            set_limit(Resource::RLIMIT_CPU, secs)?;
        }
        if let Some(bytes) = limits.mem_bytes {
            set_limit(Resource::RLIMIT_AS, bytes)?;
        }
        if let Some(nofile) = limits.nofile {
            set_limit(Resource::RLIMIT_NOFILE, nofile)?;
        }
        if let Some(nproc) = limits.nproc {
            set_limit(Resource::RLIMIT_NPROC, nproc)?;
        }
        Ok(())
    })
}

/// Turunkan batas lunak dan keras ke `requested`, tetapi tidak pernah menaikkan
/// melebihi batas keras saat ini (menaikkan butuh hak istimewa).
fn set_limit(resource: Resource, requested: u64) -> Result<(), nix::Error> {
    let (_, hard) = getrlimit(resource)?;
    let value = requested.min(hard);
    setrlimit(resource, value, value)
}

/// Terapkan konfinemen sumber daya ke `Command` sebelum spawn.
pub fn confine_command(cmd: &mut std::process::Command, limits: ResourceLimits) {
    let f = apply_pre_exec(limits);
    // SAFETY: closure hanya memanggil `setpgid`/`setrlimit` yang aman di anak.
    unsafe {
        cmd.pre_exec(f);
    }
}

/// Kirim SIGKILL ke proses dan seluruh grupnya.
///
/// Anak dibuat sebagai pemimpin grup oleh `apply_pre_exec`, jadi `-pid`
/// adalah id grup proses anak. `pid` juga dibunuh langsung sebagai jaring
/// aman bila `setpgid` belum sempat berjalan.
pub fn kill_group(pid: u32) {
    // SAFETY: `pid` adalah anak langsung proses ini; `-pid` adalah pgid anak.
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
        libc::kill(-(pid as i32), libc::SIGKILL);
    }
}
