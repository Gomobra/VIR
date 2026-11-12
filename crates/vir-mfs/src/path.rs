//! Normalisasi path ruang nama memori.

use crate::{Result, VirError};

const MAX_PATH_LEN: usize = 1024;

/// Normalkan path: mulai `/`, tanpa segmen kosong atau `.`; `..` memangkas.
pub fn normalize_path(path: &str) -> Result<String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(VirError::Config("path kosong".into()));
    }
    if trimmed.len() > MAX_PATH_LEN {
        return Err(VirError::Config(format!(
            "path lebih dari {MAX_PATH_LEN} karakter"
        )));
    }
    if trimmed.contains('\0') {
        return Err(VirError::Config("path mengandung NUL".into()));
    }
    let mut parts: Vec<&str> = Vec::new();
    for seg in trimmed.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    if parts.is_empty() {
        return Ok("/".into());
    }
    Ok(format!("/{}", parts.join("/")))
}

/// Ruang nama = segmen pertama path.
pub fn namespace_of(path: &str) -> String {
    let path = normalize_path(path).unwrap_or_else(|_| "/".into());
    let trimmed = path.trim_matches('/');
    match trimmed.split_once('/') {
        Some((first, _)) => first.to_string(),
        None if trimmed.is_empty() => "akar".to_string(),
        None => trimmed.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_dinormalkan() {
        assert_eq!(normalize_path("a/./b").unwrap(), "/a/b");
        assert_eq!(normalize_path("/a/b/../c").unwrap(), "/a/c");
        assert_eq!(normalize_path("/").unwrap(), "/");
        assert!(normalize_path("   ").is_err());
        assert_eq!(namespace_of("/episodik/e1"), "episodik");
        assert_eq!(namespace_of("/"), "akar");
    }
}
