//! Indeks semantik: embedder di belakang trait, default n-gram berbasis hash,
//! dan pencarian kosinus brute-force (cukup sampai ~100 ribu vektor).
//!
//! Titik ganti brute-force: pada ~100 ribu vektor, satu kueri memindai seluruh
//! blob (O(N·d)). Jalur skala yang disiapkan PRD adalah LanceDB (HNSW) dengan
//! Tantivy untuk teks; skema `vectors` di sini sudah cukup untuk memuat indeks
//! itu tanpa mengubah tabel versi.

use vir_common::VirError;

use crate::Result;

/// Embedder yang dapat diganti. Harus deterministik untuk isi yang sama.
pub trait Embedder: Send + Sync {
    /// Dimensi vektor keluaran.
    fn dim(&self) -> usize;
    /// Kodekan teks menjadi vektor (belum tentu ternormalisasi).
    fn embed(&self, text: &str) -> Vec<f32>;
    /// Nama pendek untuk telemetri.
    fn name(&self) -> &'static str {
        "custom"
    }
}

/// Embedder default: hash FNV-1a atas kata dan n-gram karakter.
///
/// Sifat: deterministik, tanpa dependensi model, kosinus tinggi untuk teks yang
/// beririsan leksikal. Cukup untuk Vbeta dan menjadi jalur yang bisa diganti
/// model nyata tanpa mengubah skema.
#[derive(Debug, Clone, Copy)]
pub struct HashNgramEmbedder {
    dim: usize,
    n: usize,
}

impl Default for HashNgramEmbedder {
    fn default() -> Self {
        Self { dim: 256, n: 3 }
    }
}

impl HashNgramEmbedder {
    /// Buat embedder; nilai tidak valid (0) dikembalikan ke bawaan.
    pub fn new(dim: usize, n: usize) -> Self {
        Self {
            dim: if dim == 0 { 256 } else { dim },
            n: if n == 0 { 3 } else { n },
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn n(&self) -> usize {
        self.n
    }
}

impl Embedder for HashNgramEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &'static str {
        "hash-ngram"
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0_f32; self.dim];
        let normalized: Vec<char> = text
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        if normalized.is_empty() {
            return v;
        }
        for word in text.split(|c: char| !c.is_alphanumeric()) {
            let word = word.trim();
            if word.chars().count() >= 2 {
                let hash = fnv1a(word.to_lowercase().as_bytes());
                add_feature(&mut v, hash, 1.0);
            }
        }
        if normalized.len() >= self.n {
            for window in normalized.windows(self.n) {
                let hash = fnv1a(&char_bytes(window));
                add_feature(&mut v, hash, 1.0);
            }
        } else {
            let hash = fnv1a(&char_bytes(&normalized));
            add_feature(&mut v, hash, 1.0);
        }
        l2_normalize(&mut v);
        v
    }
}

fn char_bytes(chars: &[char]) -> Vec<u8> {
    chars
        .iter()
        .flat_map(|c| (*c as u32).to_le_bytes())
        .collect()
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn add_feature(v: &mut [f32], hash: u64, weight: f32) {
    let idx = (hash % v.len() as u64) as usize;
    let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
    v[idx] += sign * weight;
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Kosinus dua vektor; 0 bila salah satunya kosong atau nol.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Peringkat kosinus brute-force: `(id, skor)` menurun.
pub fn brute_force<'a, I>(query: &[f32], records: I, k: usize) -> Vec<(i64, f64)>
where
    I: IntoIterator<Item = (i64, &'a [f32])>,
{
    let mut scored: Vec<(i64, f64)> = records
        .into_iter()
        .map(|(id, vec)| (id, f64::from(cosine(query, vec))))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.truncate(k);
    scored
}

/// Vektor menjadi blob little-endian untuk disimpan di SQLite.
pub fn encode_vector(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Blob kembali menjadi vektor; galat bila panjangnya bukan kelipatan 4.
pub fn decode_vector(blob: &[u8]) -> Result<Vec<f32>> {
    if blob.len() % 4 != 0 {
        return Err(VirError::Internal(format!(
            "blob vektor rusak: {} bita",
            blob.len()
        )));
    }
    let mut out = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.chunks_exact(4) {
        let bytes: [u8; 4] = chunk
            .try_into()
            .map_err(|_| VirError::Internal("blob vektor tidak dapat dibaca sebagai f32".into()))?;
        out.push(f32::from_le_bytes(bytes));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministik() {
        let e = HashNgramEmbedder::default();
        assert_eq!(e.embed("memori episodik"), e.embed("memori episodik"));
        assert_eq!(e.embed("memori episodik"), e.embed("Memori  episodik"));
    }

    #[test]
    fn kosinus_dasar() {
        let e = HashNgramEmbedder::default();
        let a = e.embed("rust memori");
        let b = e.embed("rust memori");
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-5);
        assert!(cosine(&a, &e.embed("zzz qqq")) < 0.2);
        assert_eq!(cosine(&[], &[]), 0.0);
    }

    #[test]
    fn brute_force_urut() {
        let e = HashNgramEmbedder::default();
        let q = e.embed("alpha beta");
        let recs = [
            (1_i64, e.embed("alpha beta gamma")),
            (2_i64, e.embed("zzz yyy")),
            (3_i64, e.embed("alpha")),
        ];
        let borrowed: Vec<(i64, &[f32])> = recs.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let hits = brute_force(&q, borrowed, 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 1);
        assert!(hits[0].1 >= hits[1].1);
    }

    #[test]
    fn blob_roundtrip() {
        let v = vec![1.0_f32, -2.5, 0.0];
        let blob = encode_vector(&v);
        assert_eq!(decode_vector(&blob).unwrap(), v);
        assert!(decode_vector(&[1, 2, 3]).is_err());
    }
}
