//! Diff baris sederhana berbasis LCS (tanpa dependensi luar).

/// Jenis baris diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffKind {
    Same,
    Add,
    Remove,
}

/// Satu baris hasil diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffLine {
    pub kind: DiffKind,
    pub text: String,
}

/// Hasil diff dua versi.
#[derive(Debug, Clone)]
pub struct Diff {
    pub a: i64,
    pub b: i64,
    pub lines: Vec<DiffLine>,
    pub added: usize,
    pub removed: usize,
}

impl Diff {
    pub fn changed(&self) -> bool {
        self.added > 0 || self.removed > 0
    }

    pub fn unchanged(&self) -> bool {
        !self.changed()
    }

    /// Ringkasan satu baris.
    pub fn summary(&self) -> String {
        format!("+{} -{}", self.added, self.removed)
    }

    /// Diff bergaya unified tanpa header.
    pub fn unified(&self) -> String {
        self.lines
            .iter()
            .map(|l| match l.kind {
                DiffKind::Same => format!("  {}", l.text),
                DiffKind::Add => format!("+ {}", l.text),
                DiffKind::Remove => format!("- {}", l.text),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Hitung diff baris antara dua teks.
pub fn diff_lines(a: &str, b: &str) -> Vec<DiffLine> {
    let left: Vec<&str> = a.split('\n').collect();
    let right: Vec<&str> = b.split('\n').collect();
    let (prefix, left_mid, right_mid, suffix) = trim(&left, &right);
    let mut out: Vec<DiffLine> = prefix
        .iter()
        .map(|s| DiffLine {
            kind: DiffKind::Same,
            text: (*s).to_string(),
        })
        .collect();
    out.extend(middle(&left_mid, &right_mid));
    out.extend(suffix.iter().map(|s| DiffLine {
        kind: DiffKind::Same,
        text: (*s).to_string(),
    }));
    out
}

fn trim<'a>(
    left: &'a [&'a str],
    right: &'a [&'a str],
) -> (Vec<&'a str>, Vec<&'a str>, Vec<&'a str>, Vec<&'a str>) {
    let mut p = 0;
    while p < left.len() && p < right.len() && left[p] == right[p] {
        p += 1;
    }
    let mut s = 0;
    while s < left.len() - p
        && s < right.len() - p
        && left[left.len() - 1 - s] == right[right.len() - 1 - s]
    {
        s += 1;
    }
    (
        left[..p].to_vec(),
        left[p..left.len() - s].to_vec(),
        right[p..right.len() - s].to_vec(),
        left[left.len() - s..].to_vec(),
    )
}

/// LCS dua potongan tengah sebagai baris diff berurutan.
fn middle(left: &[&str], right: &[&str]) -> Vec<DiffLine> {
    if left.is_empty() {
        return map_kind(right, DiffKind::Add);
    }
    if right.is_empty() {
        return map_kind(left, DiffKind::Remove);
    }
    if left.len() * right.len() > 1_200_000 {
        let mut out = map_kind(left, DiffKind::Remove);
        out.extend(map_kind(right, DiffKind::Add));
        return out;
    }
    let width = right.len() + 1;
    let mut table = vec![0_u32; (left.len() + 1) * width];
    for i in (0..left.len()).rev() {
        for j in (0..right.len()).rev() {
            table[i * width + j] = if left[i] == right[j] {
                table[(i + 1) * width + j + 1] + 1
            } else {
                table[(i + 1) * width + j].max(table[i * width + j + 1])
            };
        }
    }
    walk(&table, width, left, right)
}

fn map_kind(lines: &[&str], kind: DiffKind) -> Vec<DiffLine> {
    lines
        .iter()
        .map(|s| DiffLine {
            kind,
            text: (*s).to_string(),
        })
        .collect()
}

fn walk(table: &[u32], width: usize, left: &[&str], right: &[&str]) -> Vec<DiffLine> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        if left[i] == right[j] {
            out.push(DiffLine {
                kind: DiffKind::Same,
                text: left[i].to_string(),
            });
            i += 1;
            j += 1;
        } else if table[(i + 1) * width + j] >= table[i * width + j + 1] {
            out.push(DiffLine {
                kind: DiffKind::Remove,
                text: left[i].to_string(),
            });
            i += 1;
        } else {
            out.push(DiffLine {
                kind: DiffKind::Add,
                text: right[j].to_string(),
            });
            j += 1;
        }
    }
    out.extend(map_kind(&left[i..], DiffKind::Remove));
    out.extend(map_kind(&right[j..], DiffKind::Add));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identik() {
        let d = diff_lines("a\nb\n", "a\nb\n");
        assert!(d.iter().all(|l| l.kind == DiffKind::Same));
    }

    #[test]
    fn tambah_hapus() {
        let d = diff_lines("a\nb\nc", "a\nx\nc\nd");
        assert!(d.iter().any(|l| l.kind == DiffKind::Add && l.text == "d"));
        assert!(
            d.iter()
                .any(|l| l.kind == DiffKind::Remove && l.text == "b")
        );
        assert!(d.iter().any(|l| l.kind == DiffKind::Add && l.text == "x"));
    }

    #[test]
    fn diff_struktur() {
        let d = Diff {
            a: 1,
            b: 2,
            lines: vec![
                DiffLine {
                    kind: DiffKind::Remove,
                    text: "x".into(),
                },
                DiffLine {
                    kind: DiffKind::Add,
                    text: "y".into(),
                },
            ],
            added: 1,
            removed: 1,
        };
        assert!(d.changed());
        assert_eq!(d.summary(), "+1 -1");
        assert!(d.unified().contains("+ y"));
    }
}
