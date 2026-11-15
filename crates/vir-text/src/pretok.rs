//! Pra-tokenisasi tingkat byte (PRD F1.2 langkah 1).
//!
//! Tujuannya mencegah merge BPE melintasi batas yang merusak skrip: Jepang,
//! Thai, dan Khmer tidak memakai spasi, jadi tiap aksara menjadi potongan
//! sendiri; Cyrillic dan aksara Latin dikelompokkan per kata agar lemma ("привет")
//! utuh. Pemecah memakai kelas karakter, bukan regex, supaya cepat dan tanpa
//! dependensi. Setiap potongan dibatasi `max_word_bytes` agar biaya merge pada
//! masukan adversarial (satu kata berisi ratusan ribu byte) tetap terbatas.

/// Batas panjang potongan bawaan (byte).
pub const DEFAULT_MAX_WORD_BYTES: usize = 64;

/// Kelas karakter yang dipakai pemecah.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharClass {
    /// Spasi atau karakter lebar putih.
    Space,
    /// Angka (Nd/Nl/No).
    Digit,
    /// Aksara berspasi: Latin, Cyrillic, Yunani, Arab, Ibrani, Devanagari, ...
    Word,
    /// Aksara tanpa spasi: Han, Hiragana, Katakana, Hangul, Thai, Khmer, ...
    Unspaced,
    /// Tanda diakritik gabungan; menempel pada potongan sebelumnya.
    Mark,
    /// Emoji dan simbol bergambar.
    Symbol,
    /// Tanda baca dan sisanya.
    Punct,
}

/// Pemecah potongan byte dengan batas panjang.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreTokenizer {
    /// Panjang maksimum satu potongan (byte).
    pub max_word_bytes: usize,
}

impl Default for PreTokenizer {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_WORD_BYTES)
    }
}

impl PreTokenizer {
    /// Pemecah dengan batas panjang `max_word_bytes`.
    pub const fn new(max_word_bytes: usize) -> Self {
        Self { max_word_bytes }
    }

    /// Potong teks menjadi potongan byte yang siap di-merge.
    ///
    /// Spasi menempel pada kata yang mengikutinya (gaya GPT-2); potongan tidak
    /// pernah kosong dan hasil sambungannya selalu sama dengan `text`.
    pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut out: Vec<&'a str> = Vec::new();
        let mut start = 0usize;
        let mut prev: Option<CharClass> = None;
        for (i, c) in text.char_indices() {
            let class = classify(c);
            if class == CharClass::Mark && prev.is_some() {
                continue;
            }
            if prev.is_some_and(|p| self.breaks(p, class, i - start)) {
                out.push(&text[start..i]);
                start = i;
            }
            prev = Some(class);
        }
        if start < text.len() {
            out.push(&text[start..]);
        }
        out
    }

    /// `true` bila kelas `cur` harus memulai potongan baru setelah `prev`.
    fn breaks(&self, prev: CharClass, cur: CharClass, len: usize) -> bool {
        if len >= self.max_word_bytes {
            return true;
        }
        match (prev, cur) {
            (CharClass::Mark, _) | (_, CharClass::Mark) => false,
            (CharClass::Space, CharClass::Space) => true,
            (CharClass::Space, _) => false,
            (_, CharClass::Space) => true,
            (CharClass::Unspaced, _) | (_, CharClass::Unspaced) => true,
            (CharClass::Symbol, CharClass::Symbol) => false,
            (CharClass::Symbol, _) | (_, CharClass::Symbol) => true,
            _ => prev != cur,
        }
    }
}

/// Potong teks dengan pemecah bawaan.
pub fn pretokenize(text: &str) -> Vec<&str> {
    PreTokenizer::default().split(text)
}

/// Kelas sebuah karakter.
pub fn classify(c: char) -> CharClass {
    if c.is_whitespace() {
        return CharClass::Space;
    }
    if c.is_numeric() {
        return CharClass::Digit;
    }
    if is_mark(c) {
        return CharClass::Mark;
    }
    if c.is_alphabetic() {
        return if is_unspaced(c) {
            CharClass::Unspaced
        } else {
            CharClass::Word
        };
    }
    if is_symbol(c) {
        return CharClass::Symbol;
    }
    CharClass::Punct
}

/// Aksara tanpa spasi: satu aksara = satu potongan.
fn is_unspaced(c: char) -> bool {
    matches!(
        c as u32,
        0x0E00..=0x0E7F
            | 0x0E80..=0x0EFF
            | 0x1000..=0x109F
            | 0x1100..=0x11FF
            | 0x1780..=0x17FF
            | 0x3040..=0x30FF
            | 0x3130..=0x318F
            | 0x31F0..=0x31FF
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xA960..=0xA97F
            | 0xAC00..=0xD7AF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2FA1F
    )
}

/// Tanda diakritik gabungan (perkiraan rentang yang lazim).
fn is_mark(c: char) -> bool {
    matches!(
        c as u32,
        0x0300..=0x036F
            | 0x0483..=0x0489
            | 0x0591..=0x05BD
            | 0x0610..=0x061A
            | 0x064B..=0x065F
            | 0x0670
            | 0x06D6..=0x06DC
            | 0x0900..=0x0903
            | 0x093A..=0x094F
            | 0x0951..=0x0957
            | 0x0E31
            | 0x0E34..=0x0E3A
            | 0x0E47..=0x0E4E
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x20D0..=0x20F0
            | 0xFE00..=0xFE0F
            | 0xFE20..=0xFE2F
    )
}

/// Emoji dan simbol bergambar (dipakai sebagai satu kelas agar urutan ZWJ utuh).
fn is_symbol(c: char) -> bool {
    matches!(
        c as u32,
        0x00A9
            | 0x00AE
            | 0x203C
            | 0x2049
            | 0x2122
            | 0x2139
            | 0x2194..=0x2199
            | 0x21A9..=0x21AA
            | 0x231A..=0x231B
            | 0x2328
            | 0x23CF
            | 0x23E9..=0x23F3
            | 0x23F8..=0x23FA
            | 0x24C2
            | 0x25AA..=0x25AB
            | 0x25B6
            | 0x25C0
            | 0x25FB..=0x25FE
            | 0x2600..=0x27BF
            | 0x2934..=0x2935
            | 0x2B00..=0x2BFF
            | 0x1F000..=0x1FAFF
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jepang_per_aksara_dan_rusia_per_kata() {
        let ja = pretokenize("猫が座る");
        assert_eq!(ja, vec!["猫", "が", "座", "る"]);
        // Spasi menempel pada kata yang mengikutinya.
        let ru = pretokenize("мама мыла раму");
        assert_eq!(ru, vec!["мама", " мыла", " раму"]);
    }

    #[test]
    fn sambungan_potongan_sama_dengan_teks() {
        for text in [
            "",
            "a",
            "  ",
            " halo dunia ",
            "Привет, мир!",
            "日本語テスト",
            "😀😀 x",
        ] {
            let joined: String = pretokenize(text).concat();
            assert_eq!(joined, text);
        }
    }

    #[test]
    fn batas_panjang_dipatuhi() {
        let long = "a".repeat(200);
        let pretok = PreTokenizer::new(16);
        for chunk in pretok.split(&long) {
            assert!(chunk.len() <= 16);
        }
    }
}
