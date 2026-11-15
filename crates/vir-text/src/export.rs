//! Ekspor dan pemuat format tiktoken (PRD F1.2 langkah 4).
//!
//! Varian VIR: satu baris per token `<base64> <id>`, dengan `id` = peringkat,
//! sehingga id hasil impor identik dengan id asal. Token khusus dan slot cadangan
//! modalitas ikut ditulis (slot cadangan sebagai baris kosong) agar tata letak id
//! blok dasar tetap presisi. Merge tidak disimpan; ia dipulihkan dengan menjalankan
//! encoder pada byte tiap token (lihat [`crate::bpe::MergeTable::from_vocab`]).

use std::fs;
use std::io::Write;
use std::path::Path;

use vir_common::VirError;

use crate::bpe::MergeTable;
use crate::special::{SPECIAL_TOKENS, SpecialIds};

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Ekspor vocab ke format tiktoken (satu baris per token).
pub fn export_tiktoken(table: &MergeTable) -> String {
    let mut out = String::new();
    for (id, bytes) in table.vocab().iter().enumerate() {
        out.push_str(&base64_encode(bytes));
        out.push(' ');
        out.push_str(&id.to_string());
        out.push('\n');
    }
    out
}

/// Impor vocab dari format tiktoken lalu pulihkan merge.
pub fn load_tiktoken(text: &str) -> Result<MergeTable, VirError> {
    let mut vocab: Vec<Vec<u8>> = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let line_no = index + 1;
        let Some((token, rank)) = line.rsplit_once(' ') else {
            return Err(VirError::Internal(format!(
                "tiktoken baris {line_no}: '{line}' tanpa peringkat"
            )));
        };
        let rank: usize = rank.parse().map_err(|_| {
            VirError::Internal(format!(
                "tiktoken baris {line_no}: peringkat '{rank}' bukan angka"
            ))
        })?;
        if rank != vocab.len() {
            return Err(VirError::Internal(format!(
                "tiktoken baris {line_no}: peringkat {rank} tidak berurutan"
            )));
        }
        vocab.push(base64_decode(token)?);
    }
    MergeTable::from_vocab(vocab)
}

/// Peta token khusus dalam JSON untuk interoperabilitas dengan tiktoken.
pub fn export_specials(table: &MergeTable) -> String {
    let ids = SpecialIds::new();
    let mut out = String::from("{");
    let mut first = true;
    for token in SPECIAL_TOKENS {
        let Some(id) = ids.id_of(token) else { continue };
        if !table.token(id).is_some() {
            continue;
        }
        if !first {
            out.push(',');
        }
        first = false;
        push_json_string(&mut out, token);
        out.push(':');
        out.push_str(&id.to_string());
    }
    out.push('}');
    out
}

/// Simpan vocab ke berkas.
pub fn save(path: impl AsRef<Path>, table: &MergeTable) -> Result<(), VirError> {
    let mut file = fs::File::create(path)?;
    file.write_all(export_tiktoken(table).as_bytes())?;
    file.flush()?;
    Ok(())
}

/// Muat vocab dari berkas.
pub fn load(path: impl AsRef<Path>) -> Result<MergeTable, VirError> {
    let text = fs::read_to_string(path)?;
    load_tiktoken(&text)
}

/// Base64 (alfabet standar, dengan padding).
pub(crate) fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = u32::from(chunk[0]) << 16;
        let b1 = u32::from(chunk.get(1).copied().unwrap_or(0)) << 8;
        let b2 = u32::from(chunk.get(2).copied().unwrap_or(0));
        let triple = b0 | b1 | b2;
        out.push(symbol((triple >> 18) & 63));
        out.push(symbol((triple >> 12) & 63));
        out.push(if chunk.len() > 1 {
            symbol((triple >> 6) & 63)
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            symbol(triple & 63)
        } else {
            '='
        });
    }
    out
}

/// Base64 (alfabet standar, padding wajib).
pub(crate) fn base64_decode(text: &str) -> Result<Vec<u8>, VirError> {
    let bytes = text.as_bytes();
    if bytes.len() % 4 != 0 {
        return Err(VirError::Internal(format!(
            "base64 '{text}' panjangnya bukan kelipatan 4"
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for quad in bytes.chunks(4) {
        let mut acc = 0u32;
        let mut pad = 0u32;
        for (k, c) in quad.iter().enumerate() {
            let value = if *c == b'=' {
                if k < 2 {
                    return Err(VirError::Internal(format!(
                        "base64 '{text}' padding tidak sah"
                    )));
                }
                pad += 1;
                0
            } else {
                u32::from(decode_symbol(*c).ok_or_else(|| {
                    VirError::Internal(format!("base64 '{text}' simbol tidak dikenal"))
                })?)
            };
            acc = (acc << 6) | value;
        }
        out.push((acc >> 16) as u8);
        if pad < 2 {
            out.push((acc >> 8) as u8);
        }
        if pad < 1 {
            out.push(acc as u8);
        }
    }
    Ok(out)
}

fn symbol(value: u32) -> char {
    ALPHABET[(value & 63) as usize] as char
}

fn decode_symbol(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::TrainConfig;

    #[test]
    fn base64_voktor_diketahui() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64_decode("aGVsbG8=").ok(), Some(b"hello".to_vec()));
        assert_eq!(base64_decode("Zm9v").ok(), Some(b"foo".to_vec()));
        assert!(base64_decode("Zm9v=").is_err());
        assert!(base64_decode("Z").is_err());
    }

    #[test]
    fn ekspor_impor_identik() {
        let encoder = crate::encoder::BpeEncoder::trained(
            ["banana bandana", "abab", "Привет мир", "日本語"],
            &TrainConfig::default(),
        );
        let table = encoder.table();
        let text = export_tiktoken(table);
        let back = load_tiktoken(&text).ok();
        assert_eq!(back.as_ref(), Some(table));
        let specials = export_specials(table);
        assert!(specials.contains("\"<pad>\":256"));
        assert!(specials.contains("\"<eos>\":258"));
    }
}
