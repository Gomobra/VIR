//! Landlock: pembatasan sistem berkas dan jaringan per proses (F3.1).
//!
//! Landlock tersedia pada Linux 5.13+ dan harus diaktifkan saat boot.
//! Modul ini mendeteksi ABI saat runtime dan menolak dengan galat jelas bila
//! kernel tidak mendukung; pembatasan tidak pernah dimatikan diam-diam.
//!
//! Aturan yang dipasang:
//!
//! - baca + `exec` di seluruh hierarki yang ada (agar program dan pustaka
//!   dinamis tetap dapat dimuat dan dijalankan);
//! - baca + tulis hanya pada direktori kerja dan jalur `allow_write`;
//! - jaringan TCP ditolak bila ABI >= 4 (kernel 6.7+); pada ABI lebih rendah
//!   pembatasan jaringan tidak tersedia dan itu dicatat sebagai galat dukungan.
//!
//! Landlock membatasi thread yang memanggilnya dan diwarisi oleh anak lewat
//! `exec`, sehingga pemasangannya di `pre_exec` cukup untuk proses anak.

use vir_common::VirError;

/// Jalur baca yang umum dipakai runtime. Ditambahkan sebagai baca + `exec`
/// bila ada; jalur yang tidak ada dilewati.
#[cfg(feature = "sandbox")]
const COMMON_READ: &[&str] = &[
    "/", "/usr", "/etc", "/lib", "/lib64", "/bin", "/sbin", "/proc", "/sys", "/dev", "/tmp", "/run",
];

/// ABI Landlock efektif pada kernel ini (kernel 5.13+).
///
/// Mengembalikan galat bila fitur `sandbox` tidak dikompilasi atau kernel
/// tidak mendukung Landlock.
pub fn detect_abi() -> Result<u32, VirError> {
    #[cfg(feature = "sandbox")]
    {
        // `landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION)`
        // mengembalikan versi ABI tanpa membuat aturan apa pun.
        // SAFETY: argumen valid; hanya mengembalikan versi atau errno negatif.
        let ret = unsafe {
            libc::syscall(
                libc::SYS_landlock_create_ruleset,
                std::ptr::null::<libc::c_void>(),
                0_usize,
                1_u32,
            )
        };
        if ret < 0 {
            return Err(VirError::Internal(format!(
                "landlock tidak tersedia (errno {}); butuh kernel Linux 5.13+ dengan Landlock aktif",
                -ret
            )));
        }
        Ok(ret as u32)
    }
    #[cfg(not(feature = "sandbox"))]
    {
        Err(VirError::Internal(
            "fitur 'sandbox' tidak aktif; pembatasan sistem berkas tidak tersedia".into(),
        ))
    }
}

/// Terapkan kebijakan Landlock ke proses saat ini (dipanggil di anak).
///
/// `workdir` menjadi satu-satunya direktori yang dapat ditulis bersama
/// `allow_write`. Bila kernel tidak mendukung, kembalikan galat jelas.
pub fn confine(workdir: &str, allow_write: &[&str]) -> Result<(), VirError> {
    #[cfg(feature = "sandbox")]
    {
        use landlock::RulesetCreatedAttr;
        use landlock::{Access, AccessFs, AccessNet, PathBeneath, PathFd, Ruleset, RulesetAttr};

        let abi = detect_abi()?;
        // ABI tertinggi yang dikenal crate; lapisan kompatibilitas menurunkan
        // hak yang tidak didukung kernel secara otomatis.
        let max = landlock::ABI::V9;
        let read_write = AccessFs::from_all(max);
        let read_exec = AccessFs::from_read(max);

        let mut builder = Ruleset::default()
            .handle_access(read_write)
            .map_err(internal)?;
        if abi >= 4 {
            // Tanpa aturan port, seluruh TCP bind/connect ditolak.
            builder = builder
                .handle_access(AccessNet::from_all(max))
                .map_err(internal)?;
        }
        let mut ruleset = builder.create().map_err(internal)?;

        // Baca + exec di seluruh hierarki; tanpa ini `exec` akan ditolak.
        // `/` selalu ada dan menutupi sisa hierarki; jalur mount lain yang ada
        // (mis. `/proc`, `/dev`, `/tmp`) ditambahkan bila tersedia.
        for &path in COMMON_READ {
            if let Ok(fd) = PathFd::new(path) {
                ruleset = ruleset
                    .add_rule(PathBeneath::new(fd, read_exec))
                    .map_err(internal)?;
            }
        }

        // Direktori kerja dan jalur tulis eksplisit menjadi read-write.
        let wd = PathFd::new(workdir).map_err(internal)?;
        ruleset = ruleset
            .add_rule(PathBeneath::new(wd, read_write))
            .map_err(internal)?;
        for &path in allow_write {
            let fd = PathFd::new(path).map_err(internal)?;
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, read_write))
                .map_err(internal)?;
        }

        // `restrict_self` juga memasang PR_SET_NO_NEW_PRIVS.
        ruleset.restrict_self().map_err(internal)?;
        Ok(())
    }
    #[cfg(not(feature = "sandbox"))]
    {
        let _ = (workdir, allow_write);
        Err(VirError::Internal(
            "fitur 'sandbox' tidak aktif; pembatasan sistem berkas tidak tersedia".into(),
        ))
    }
}

#[cfg(feature = "sandbox")]
fn internal<E: std::fmt::Display>(e: E) -> VirError {
    VirError::Internal(format!("landlock: {e}"))
}
