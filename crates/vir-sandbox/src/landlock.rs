//! Landlock + seccomp confinement (F3.1).

use vir_common::VirError;

/// Deteksi ABI Landlock (kernel >= 5.13).
pub fn detect_abi() -> Result<u32, VirError> {
    #[cfg(feature = "sandbox")]
    {
        let abi = landlock::ABI::new();
        Ok(abi as u32)
    }
    #[cfg(not(feature = "sandbox"))]
    {
        Err(VirError::Internal(
            "fitur 'sandbox' tidak aktif (build tanpa landlock)".into(),
        ))
    }
}

/// Terapkan kebijakan filesystem read-only pada direktori kerja.
/// `allow_write` = path yang boleh ditulis.
pub fn confine(workdir: &str, allow_write: &[&str]) -> Result<(), VirError> {
    #[cfg(feature = "sandbox")]
    {
        use landlock::{
            AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr,
            ABI,
        };
        let abi = ABI::new();
        let read_write = AccessFs::from_all(abi);
        let read_only = AccessFs::from_read(abi);

        let ruleset = Ruleset::new()
            .map_err(|e| VirError::Internal(e.to_string()))?
            .handle_access(read_write)
            .map_err(|e| VirError::Internal(e.to_string()))?
            .create()
            .map_err(|e| VirError::Internal(e.to_string()))?;

        // read-only: workdir
        let wd = PathFd::new(workdir).map_err(|e| VirError::Internal(e.to_string()))?;
        ruleset
            .add_rule(PathBeneath::new(wd, read_only))
            .map_err(|e| VirError::Internal(e.to_string()))?;

        // write: path yang diizinkan
        for p in allow_write {
            let fd = PathFd::new(p).map_err(|e| VirError::Internal(e.to_string()))?;
            ruleset
                .add_rule(PathBeneath::new(fd, read_write))
                .map_err(|e| VirError::Internal(e.to_string()))?;
        }

        ruleset
            .apply()
            .map_err(|e| VirError::Internal(e.to_string()))?;
        Ok(())
    }
    #[cfg(not(feature = "sandbox"))]
    {
        let _ = (workdir, allow_write);
        Err(VirError::Internal(
            "fitur 'sandbox' tidak aktif (build tanpa landlock)".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_detected_or_disabled() {
        // tanpa fitur sandbox, harus error jelas (bukan panic)
        let r = detect_abi();
        if cfg!(feature = "sandbox") {
            assert!(r.is_ok());
        } else {
            assert!(r.is_err());
        }
    }
}