//! Tokenizer BPE tingkat byte (implementasi acuan).

use std::collections::HashMap;

/// Token khusus (indeks awal setelah 256 byte).
pub const SPECIAL_TOKENS: &[&str] = &[
    "<pad>",
    "<bos>",
    "<eos>",
    "<mask>",
    "<fim_pre>",
    "<fim_suf>",
    "<fim_mid>",
];

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// token id -> bytes
    pub vocab: Vec<Vec<u8>>,
    /// bytes -> token id
    index: HashMap<Vec<u8>, u32>,
    /// pasangan -> token id baru (merge)
    merges: HashMap<(u32, u32), u32>,
    /// urutan merge (rank rendah = prioritas tinggi)
    merge_rank: HashMap<(u32, u32), usize>,
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BpeTokenizer {
    pub fn new() -> Self {
        let mut vocab: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let mut index = HashMap::new();
        for (i, v) in vocab.iter().enumerate() {
            index.insert(v.clone(), i as u32);
        }
        // token khusus
        for sp in SPECIAL_TOKENS {
            let bytes = sp.as_bytes().to_vec();
            if let std::collections::hash_map::Entry::Vacant(e) = index.entry(bytes.clone()) {
                let id = vocab.len() as u32;
                vocab.push(bytes.clone());
                e.insert(id);
            }
        }
        Self {
            vocab,
            index,
            merges: HashMap::new(),
            merge_rank: HashMap::new(),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Latih dari korpus teks sampai ukuran vocab target.
    pub fn train(&mut self, texts: &[String], target_vocab: usize) {
        // representasi tiap teks sebagai Vec<u32> byte
        let mut corpus: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| t.bytes().map(|b| b as u32).collect())
            .collect();

        while self.vocab.len() < target_vocab {
            // hitung frekuensi pasangan
            let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
            for seq in &corpus {
                for w in seq.windows(2) {
                    *counts.entry((w[0], w[1])).or_insert(0) += 1;
                }
            }
            if counts.is_empty() {
                break;
            }
            // pasangan paling sering
            let best = counts.into_iter().max_by_key(|&(_, c)| c);
            let Some(((a, b), _)) = best else { break };
            if self.merges.contains_key(&(a, b)) {
                break;
            }
            // buat token baru
            let mut new_bytes = self.vocab[a as usize].clone();
            new_bytes.extend_from_slice(&self.vocab[b as usize]);
            let new_id = self.vocab.len() as u32;
            self.vocab.push(new_bytes.clone());
            self.index.insert(new_bytes, new_id);
            self.merges.insert((a, b), new_id);
            self.merge_rank.insert((a, b), self.merge_rank.len());

            // terapkan merge ke korpus
            for seq in &mut corpus {
                let mut out = Vec::with_capacity(seq.len());
                let mut i = 0;
                while i < seq.len() {
                    if i + 1 < seq.len() && seq[i] == a && seq[i + 1] == b {
                        out.push(new_id);
                        i += 2;
                    } else {
                        out.push(seq[i]);
                        i += 1;
                    }
                }
                *seq = out;
            }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        if tokens.len() < 2 {
            return tokens;
        }
        // terapkan merge berdasarkan rank (prioritas tinggi dulu)
        let mut changed = true;
        while changed {
            changed = false;
            // cari pasangan dengan rank terendah
            let mut best_pos: Option<usize> = None;
            let mut best_rank: Option<usize> = None;
            for i in 0..tokens.len().saturating_sub(1) {
                if let Some(&r) = self.merge_rank.get(&(tokens[i], tokens[i + 1])) {
                    if best_rank.is_none_or(|br| r < br) {
                        best_rank = Some(r);
                        best_pos = Some(i);
                    }
                }
            }
            if let Some(i) = best_pos {
                let a = tokens[i];
                let b = tokens[i + 1];
                let new_id = self.merges[&(a, b)];
                tokens[i] = new_id;
                tokens.remove(i + 1);
                changed = true;
            }
        }
        tokens
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &t in tokens {
            if let Some(v) = self.vocab.get(t as usize) {
                bytes.extend_from_slice(v);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Byte-level fallback tokenizer (roundtrip selalu).
#[derive(Debug, Clone, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn new() -> Self {
        Self
    }
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }
    pub fn decode(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

pub fn roundtrip_ok(text: &str) -> bool {
    let t = ByteTokenizer::new();
    t.decode(&t.encode(text)) == text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_utf8() {
        for s in [
            "Indonesia",
            "English",
            "Русский",
            "日本語",
            "😀🎉",
            "fn main() {}",
        ] {
            assert!(roundtrip_ok(s), "gagal roundtrip: {s}");
        }
    }

    #[test]
    fn bpe_train_and_roundtrip() {
        let texts = vec!["hello hello world".to_string(); 4];
        let mut tok = BpeTokenizer::new();
        tok.train(&texts, 280);
        assert!(tok.vocab_size() >= 256);
        let enc = tok.encode("hello world");
        let dec = tok.decode(&enc);
        assert_eq!(dec, "hello world");
    }

    #[test]
    fn bpe_encode_decode_consistency() {
        let texts = vec!["abc abc abc".to_string(); 2];
        let mut tok = BpeTokenizer::new();
        tok.train(&texts, 260);
        let s = "abc";
        let enc = tok.encode(s);
        assert_eq!(tok.decode(&enc), s);
    }
}
