//! Transformasi FIM (PSM/SPM) + invers (PRD F1.3 langkah 4).
//!
//! PSM: `[pre] prefix [suf] suffix [mid] middle` (urutan stabil untuk decoder).
//! SPM: `[suf] suffix [pre] prefix [mid] middle` (gaya "suffix-then-prefix").
//! Kedua mode memakai tiga penanda token khusus dari `vir-text`, sehingga
//! transformasi dapat dibalik tepat: `parse(transform(x)) == x`.

use vir_text::SpecialIds;

/// Tiga penanda FIM (id token).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FimMarkers {
    pub prefix: u32,
    pub suffix: u32,
    pub middle: u32,
}

impl FimMarkers {
    /// Penanda eksplisit.
    pub const fn new(prefix: u32, suffix: u32, middle: u32) -> Self {
        Self {
            prefix,
            suffix,
            middle,
        }
    }

    /// Penanda bawaan dari tata letak `vir-text`.
    pub const fn defaults() -> Self {
        let ids = SpecialIds::new();
        Self::new(ids.fim_prefix, ids.fim_suffix, ids.fim_middle)
    }

    /// `true` bila id adalah salah satu penanda FIM.
    pub const fn is_marker(&self, id: u32) -> bool {
        id == self.prefix || id == self.suffix || id == self.middle
    }
}

impl Default for FimMarkers {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Mode penataan FIM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FimMode {
    /// Prefix, suffix, middle.
    Psm,
    /// Suffix, prefix, middle.
    Spm,
}

impl FimMode {
    /// Nama mode untuk log.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Psm => "psm",
            Self::Spm => "spm",
        }
    }
}

/// Bagian hasil `parse`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FimParts<'a> {
    pub prefix: &'a [u32],
    pub suffix: &'a [u32],
    pub middle: &'a [u32],
}

/// Bangun urutan FIM sesuai mode.
pub fn transform(
    mode: FimMode,
    prefix: &[u32],
    suffix: &[u32],
    middle: &[u32],
    markers: &FimMarkers,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(prefix.len() + suffix.len() + middle.len() + 3);
    match mode {
        FimMode::Psm => {
            out.push(markers.prefix);
            out.extend_from_slice(prefix);
            out.push(markers.suffix);
            out.extend_from_slice(suffix);
        }
        FimMode::Spm => {
            out.push(markers.suffix);
            out.extend_from_slice(suffix);
            out.push(markers.prefix);
            out.extend_from_slice(prefix);
        }
    }
    out.push(markers.middle);
    out.extend_from_slice(middle);
    out
}

/// Balikkan urutan FIM; `None` bila penanda tidak lengkap, berulang, atau liar.
pub fn parse<'a>(tokens: &'a [u32], markers: &FimMarkers) -> Option<FimParts<'a>> {
    let mut sections: Vec<(u32, &[u32])> = Vec::with_capacity(3);
    let mut pos = 0usize;
    while pos < tokens.len() {
        let id = tokens[pos];
        if !markers.is_marker(id) {
            return None;
        }
        let start = pos + 1;
        let mut end = start;
        while end < tokens.len() && !markers.is_marker(tokens[end]) {
            end += 1;
        }
        sections.push((id, &tokens[start..end]));
        pos = end;
    }
    if sections.len() != 3 {
        return None;
    }
    let (a_id, a_slice) = sections[0];
    let (b_id, b_slice) = sections[1];
    let (c_id, c_slice) = sections[2];
    if c_id != markers.middle {
        return None;
    }
    if a_id == markers.prefix && b_id == markers.suffix {
        return Some(FimParts {
            prefix: a_slice,
            suffix: b_slice,
            middle: c_slice,
        });
    }
    if a_id == markers.suffix && b_id == markers.prefix {
        return Some(FimParts {
            suffix: a_slice,
            prefix: b_slice,
            middle: c_slice,
        });
    }
    None
}

/// PSM dengan penanda bawaan (API lama).
pub fn fim_psm(prefix: &[u32], suffix: &[u32], middle: &[u32]) -> Vec<u32> {
    transform(
        FimMode::Psm,
        prefix,
        suffix,
        middle,
        &FimMarkers::defaults(),
    )
}

/// SPM dengan penanda bawaan (API lama).
pub fn fim_spm(prefix: &[u32], suffix: &[u32], middle: &[u32]) -> Vec<u32> {
    transform(
        FimMode::Spm,
        prefix,
        suffix,
        middle,
        &FimMarkers::defaults(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty<'a>() -> FimParts<'a> {
        FimParts {
            prefix: &[],
            suffix: &[],
            middle: &[],
        }
    }

    #[test]
    fn psm_dan_spm_dapat_dibalik() {
        let markers = FimMarkers::defaults();
        let prefix = [1u32, 2, 3];
        let suffix = [4u32, 5];
        let middle = [6u32, 7, 8, 9];
        for mode in [FimMode::Psm, FimMode::Spm] {
            let tokens = transform(mode, &prefix, &suffix, &middle, &markers);
            let parts = parse(&tokens, &markers).unwrap_or_else(empty);
            assert_eq!(parts.prefix, prefix.as_slice(), "mode {mode:?}");
            assert_eq!(parts.suffix, suffix.as_slice(), "mode {mode:?}");
            assert_eq!(parts.middle, middle.as_slice(), "mode {mode:?}");
        }
    }

    #[test]
    fn api_lama_konsisten_dengan_mode() {
        let markers = FimMarkers::defaults();
        assert_eq!(
            fim_psm(&[1], &[2], &[3]),
            transform(FimMode::Psm, &[1], &[2], &[3], &markers)
        );
        assert_eq!(
            fim_spm(&[1], &[2], &[3]),
            transform(FimMode::Spm, &[1], &[2], &[3], &markers)
        );
    }

    #[test]
    fn penanda_khusus_dapat_dibuat() {
        let markers = FimMarkers::new(900, 901, 902);
        assert!(markers.is_marker(900));
        assert!(markers.is_marker(902));
        assert!(!markers.is_marker(903));
        let tokens = transform(FimMode::Psm, &[7], &[8], &[9], &markers);
        assert_eq!(tokens, vec![900, 7, 901, 8, 902, 9]);
        let parts = parse(&tokens, &markers).unwrap_or_else(empty);
        assert_eq!(parts.prefix, [7].as_slice());
        assert_eq!(parts.suffix, [8].as_slice());
        assert_eq!(parts.middle, [9].as_slice());
    }

    #[test]
    fn urutan_salah_dan_berulang_ditolak() {
        let markers = FimMarkers::defaults();
        let tokens = transform(FimMode::Spm, &[1], &[2], &[3], &markers);
        assert_eq!(tokens[0], markers.suffix);
        let parts = parse(&tokens, &markers).unwrap_or_else(empty);
        assert_eq!(parts.suffix, [2].as_slice());
        assert_eq!(parts.prefix, [1].as_slice());
        assert!(parse(&[1, 2, 3], &markers).is_none());
        assert!(parse(&[markers.prefix, 1, markers.middle], &markers).is_none());
        assert!(parse(&[markers.middle, 1, markers.middle], &markers).is_none());
    }
}
