//! Sandbox eksekusi kode: subprocess terisolasi (F3.1).
//!
//! Lapisan isolasi yang diterapkan ke setiap proses anak:
//!
//! - rlimit (CPU, memori, berkas, proses) dan `setpgid` lewat
//!   [`limits::confine_command`];
//! - direktori kerja sementara dan lingkungan terbatas lewat [`Policy`];
//! - batas waktu dinding dengan pembunuhan seluruh grup proses;
//! - pemotongan keluaran besar;
//! - (fitur `sandbox`) pembatasan sistem berkas/jaringan dengan
//!   [`landlock::confine`] dan allowlist syscall dengan [`seccomp::confine`],
//!   keduanya dipasang di `pre_exec` mengikuti urutan Sandlock:
//!   `setpgid` → `chdir` → `PR_SET_NO_NEW_PRIVS` → Landlock → seccomp →
//!   `exec`.
//!
//! Tanpa fitur `sandbox`, pembatasan sistem berkas dan penyaringan syscall tidak
//! tersedia; hanya rlimit, timeout, dan batas keluaran yang berlaku.
//! Lihat `docs/adr/ADR-006-sandbox.md`.

pub mod landlock;
pub mod limits;
pub mod seccomp;
pub mod syscalls;

use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use vir_common::VirError;

use limits::ResourceLimits;

/// Kebijakan eksekusi satu perintah.
#[derive(Debug, Clone)]
pub struct Policy {
    /// Batas waktu dinding; seluruh grup proses dibunuh bila terlampaui.
    pub timeout: Duration,
    /// Batas ukuran keluaran stdout/stderr (bita); sisanya dipotong.
    pub max_output: usize,
    /// Direktori kerja anak; `None` berarti mewarisi cwd pemanggil.
    pub workdir: Option<PathBuf>,
    /// Bila `true`, lingkungan dibersihkan lalu hanya `env` yang dipasang.
    pub clear_env: bool,
    /// Lingkungan yang dipasang pada anak.
    pub env: BTreeMap<String, String>,
    /// Batas sumber daya (rlimit).
    pub limits: ResourceLimits,
    /// Jalur tambahan yang boleh ditulis (di samping `workdir`).
    pub allow_write: Vec<PathBuf>,
    /// Terapkan pembatasan Landlock bila fitur `sandbox` aktif.
    pub landlock: bool,
    /// Terapkan allowlist syscall seccomp-bpf bila fitur `sandbox` aktif.
    ///
    /// Wajib bekerja bersama [`Policy::landlock`]: keduanya dipasang pada anak
    /// sebelum `exec` (Landlock lebih dulu, seccomp terakhir).
    pub seccomp: bool,
}

impl Default for Policy {
    fn default() -> Self {
        let mut env = BTreeMap::new();
        // Warisi PATH host agar program ditemukan di lingkungan apa pun; variabel
        // lain dibersihkan sehingga lingkungan tetap terbatas.
        let path = std::env::var("PATH").unwrap_or_else(|_| {
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".to_string()
        });
        env.insert("PATH".to_string(), path);
        env.insert("HOME".to_string(), "/tmp".to_string());
        env.insert("TMPDIR".to_string(), "/tmp".to_string());
        env.insert("LANG".to_string(), "C".to_string());
        env.insert("LC_ALL".to_string(), "C".to_string());
        Self {
            timeout: Duration::from_secs(5),
            max_output: 64 * 1024,
            workdir: None,
            clear_env: true,
            env,
            limits: ResourceLimits::standard(),
            allow_write: Vec::new(),
            landlock: true,
            seccomp: true,
        }
    }
}

impl Policy {
    /// Batas waktu dinding.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Batas ukuran keluaran (bita).
    pub fn with_max_output(mut self, max_output: usize) -> Self {
        self.max_output = max_output;
        self
    }

    /// Direktori kerja anak.
    pub fn with_workdir(mut self, workdir: impl Into<PathBuf>) -> Self {
        self.workdir = Some(workdir.into());
        self
    }

    /// Tambah satu variabel lingkungan.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Ganti batas sumber daya.
    pub fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Izinkan menulis ke satu jalur tambahan.
    pub fn with_allow_write(mut self, path: impl Into<PathBuf>) -> Self {
        self.allow_write.push(path.into());
        self
    }

    /// Aktif/nonaktifkan Landlock.
    pub fn with_landlock(mut self, enabled: bool) -> Self {
        self.landlock = enabled;
        self
    }

    /// Aktif/nonaktifkan allowlist syscall seccomp-bpf.
    pub fn with_seccomp(mut self, enabled: bool) -> Self {
        self.seccomp = enabled;
        self
    }
}

/// Hasil eksekusi satu perintah.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    /// `true` bila proses melewati `Policy::timeout` dan dibunuh.
    pub timed_out: bool,
    /// `true` bila keluaran dipotong karena melebihi `Policy::max_output`.
    pub truncated: bool,
}

/// Bangun `Command` berkonfinemen tanpa spawn.
fn build_command(program: &str, args: &[&str], policy: &Policy, stdin: Stdio) -> Command {
    let mut cmd = Command::new(program);
    cmd.args(args);
    if let Some(dir) = &policy.workdir {
        cmd.current_dir(dir);
    }
    if policy.clear_env {
        cmd.env_clear();
    }
    for (key, value) in &policy.env {
        cmd.env(key, value);
    }
    cmd.stdin(stdin)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    limits::confine_command(&mut cmd, policy.limits);
    wire_landlock(&mut cmd, policy);
    wire_seccomp(&mut cmd, policy);
    cmd
}

/// Jalankan perintah sampai selesai, dengan timeout dan batas keluaran.
pub fn run(program: &str, args: &[&str], policy: &Policy) -> Result<RunResult, VirError> {
    let mut cmd = build_command(program, args, policy, Stdio::null());
    let child = cmd.spawn().map_err(VirError::Io)?;
    collect(child, policy)
}

/// Spawn proses anak berkonfinemen dengan pipa stdin/stdout/stderr.
///
/// Dipakai REPL berumur panjang; pemanggil bertangggung jawab menutup pipa dan
/// membunuh grup proses lewat [`limits::kill_group`].
pub fn spawn(program: &str, args: &[&str], policy: &Policy) -> Result<Child, VirError> {
    let mut cmd = build_command(program, args, policy, Stdio::piped());
    cmd.spawn().map_err(VirError::Io)
}

fn collect(mut child: Child, policy: &Policy) -> Result<RunResult, VirError> {
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| VirError::Internal("stdout anak tidak tersedia".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| VirError::Internal("stderr anak tidak tersedia".into()))?;

    let overflow = Arc::new(AtomicBool::new(false));
    let out_handle = spawn_reader(stdout, policy.max_output, overflow.clone());
    let err_handle = spawn_reader(stderr, policy.max_output, overflow.clone());

    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if overflow.load(Ordering::SeqCst) {
            limits::kill_group(child.id());
            break child.wait();
        }
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => {
                if start.elapsed() >= policy.timeout {
                    limits::kill_group(child.id());
                    timed_out = true;
                    break child.wait();
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => {
                limits::kill_group(child.id());
                let _ = child.wait();
                return Err(VirError::Io(e));
            }
        }
    };
    let status = status.map_err(VirError::Io)?;

    let stdout = join(out_handle);
    let stderr = join(err_handle);
    let (stdout, truncated_o) = cap(&stdout, policy.max_output);
    let (stderr, truncated_e) = cap(&stderr, policy.max_output);
    Ok(RunResult {
        stdout,
        stderr,
        exit_code: status.code(),
        timed_out,
        truncated: truncated_o || truncated_e || overflow.load(Ordering::SeqCst),
    })
}

fn spawn_reader<R: Read + Send + 'static>(
    mut reader: R,
    max: usize,
    overflow: Arc<AtomicBool>,
) -> JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    if buf.len() + n > max {
                        let room = max.saturating_sub(buf.len());
                        buf.extend_from_slice(&chunk[..room]);
                        overflow.store(true, Ordering::SeqCst);
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
        buf
    })
}

fn join(handle: JoinHandle<Vec<u8>>) -> Vec<u8> {
    handle.join().unwrap_or_default()
}

fn cap(bytes: &[u8], max: usize) -> (String, bool) {
    if bytes.len() <= max {
        (String::from_utf8_lossy(bytes).into_owned(), false)
    } else {
        (String::from_utf8_lossy(&bytes[..max]).into_owned(), true)
    }
}

#[cfg(feature = "sandbox")]
fn wire_landlock(cmd: &mut Command, policy: &Policy) {
    use std::os::unix::process::CommandExt;
    if !policy.landlock {
        return;
    }
    let Some(workdir) = policy.workdir.as_ref() else {
        return;
    };
    let workdir = workdir.clone();
    let allow_write: Vec<PathBuf> = policy.allow_write.clone();
    // SAFETY: closure hanya menerapkan Landlock pada proses anak sebelum exec.
    unsafe {
        cmd.pre_exec(move || {
            let wd = workdir.to_string_lossy().into_owned();
            let allow: Vec<String> = allow_write
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect();
            let allow_refs: Vec<&str> = allow.iter().map(|s| s.as_str()).collect();
            landlock::confine(&wd, &allow_refs).map_err(|e| std::io::Error::other(e.to_string()))
        });
    }
}

#[cfg(not(feature = "sandbox"))]
fn wire_landlock(_cmd: &mut Command, _policy: &Policy) {
    // Landlock tidak tersedia tanpa fitur `sandbox`; rlimit dan timeout tetap
    // berlaku.
}

#[cfg(feature = "sandbox")]
fn wire_seccomp(cmd: &mut Command, policy: &Policy) {
    use std::os::unix::process::CommandExt;
    if !policy.seccomp {
        return;
    }
    // SAFETY: closure hanya memasang `PR_SET_NO_NEW_PRIVS` dan filter seccomp
    // pada proses anak sebelum `exec`.
    unsafe {
        cmd.pre_exec(|| {
            // Landlock (yang dipasang lebih dulu) juga memasang flag ini, tetapi
            // pemasangan eksplisit membuat urutan pada modul ini berdiri sendiri
            // dan tetap benar bila Landlock nonaktif.
            if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            seccomp::confine().map_err(|e| std::io::Error::other(e.to_string()))
        });
    }
}

#[cfg(not(feature = "sandbox"))]
fn wire_seccomp(_cmd: &mut Command, _policy: &Policy) {
    // Filter syscall tidak tersedia tanpa fitur `sandbox`.
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::path::Path;
    use std::process::Stdio;

    fn command_for(policy: &Policy) -> Command {
        build_command("vir-uji", &["-c", "nilai"], policy, Stdio::null())
    }

    #[test]
    fn argv_dan_cwd_sesuai_kebijakan() {
        let policy = Policy::default().with_workdir("/srv/vir");
        let cmd = command_for(&policy);
        assert_eq!(cmd.get_program(), OsStr::new("vir-uji"));
        let args: Vec<&OsStr> = cmd.get_args().collect();
        assert_eq!(args, vec![OsStr::new("-c"), OsStr::new("nilai")]);
        assert_eq!(cmd.get_current_dir(), Some(Path::new("/srv/vir")));
    }

    #[test]
    fn cwd_tidak_dipasang_bila_kebijakan_tanpa_workdir() {
        let cmd = command_for(&Policy::default());
        assert!(cmd.get_current_dir().is_none());
    }

    #[test]
    fn env_kebijakan_dipasang_ke_anak() {
        let cmd = command_for(&Policy::default().with_env("VIR_UJI", "1"));
        let envs: Vec<(Option<&OsStr>, Option<&OsStr>)> = cmd.get_envs().collect();
        let vir = envs
            .iter()
            .find(|(key, _)| *key == Some(OsStr::new("VIR_UJI")))
            .map(|(_, value)| *value);
        assert_eq!(vir, Some(Some(OsStr::new("1"))));
        assert!(envs.iter().any(|(key, _)| *key == Some(OsStr::new("HOME"))));
    }
}
