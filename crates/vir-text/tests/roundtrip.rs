//! Round-trip encode/decode byte-level dan invariant F1.2 (PRD F1.2).

use vir_text::{
    BYTE_VOCAB, BpeEncoder, CharClass, Encoder, PreTokenizer, SPECIAL_TOKENS, SpecialIds,
    TrainConfig, classify, export_tiktoken, load_tiktoken, pretokenize, resolve_backend, train_bpe,
};

/// RNG kecil deterministik; cukup untuk data uji tanpa dependensi tambahan.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Kumpulan teks satu kalimat per bahasa/domain.
const KORPUS: &[&str] = &[
    "Halo, apa kabar? Saya sedang belajar membaca teks berbahasa Indonesia.",
    "The quick brown fox jumps over the lazy dog while the cat sleeps quietly.",
    "Привет, мир! Мама мыла раму, а папа читал газету на кухне.",
    "日本語のテキストを正しく分割できることを確認します。カタカナとひらがなも含みます。",
    "한국어 문장도 정상적으로 처리되는지 확인합니다.",
    "Emoji 😀😃😄 dan simbol <>&\"' serta karakter kontrol \\t\\n aman.",
    "def add(a: int, b: int) -> int:\n    return a + b\n\nprint(add(2, 3))",
    "Angka campuran 1234567890 dan 3,14159 serta 0x1F600 harus utuh.",
];

fn corpus_texts() -> Vec<String> {
    KORPUS.iter().map(|t| (*t).to_string()).collect()
}

#[test]
fn roundtrip_multibahasa_byte_persis() {
    let encoder = BpeEncoder::trained(KORPUS.iter().copied(), &TrainConfig::default());
    for text in KORPUS {
        let ids = encoder.encode(text);
        assert_eq!(encoder.decode_bytes(&ids), text.as_bytes(), "gagal: {text}");
        assert!(ids.iter().all(|id| (*id as usize) < encoder.vocab_size()));
    }
}

#[test]
fn roundtrip_byte_utf8_acak() {
    let mut rng = SplitMix(0x5EED);
    let encoder = BpeEncoder::trained(KORPUS.iter().copied(), &TrainConfig::default());
    let mut checked = 0;
    for _ in 0..300 {
        let len = rng.below(48);
        let mut bytes = Vec::with_capacity(len);
        for _ in 0..len {
            bytes.push(rng.below(256) as u8);
        }
        let Ok(text) = String::from_utf8(bytes) else {
            continue;
        };
        checked += 1;
        let ids = encoder.encode(&text);
        assert_eq!(encoder.decode_bytes(&ids), text.as_bytes());
    }
    assert!(checked > 100, "terlalu sedikit sampel UTF-8 sah: {checked}");
}

#[test]
fn tokenizer_tanpa_merge_adalah_byte_level() {
    let encoder = BpeEncoder::new();
    for text in KORPUS {
        let ids = encoder.encode(text);
        assert_eq!(ids, text.bytes().map(u32::from).collect::<Vec<u32>>());
    }
}

#[test]
fn token_khusus_tidak_pernah_dihasilkan_encode() {
    let ids = SpecialIds::new();
    let encoder = BpeEncoder::trained(KORPUS.iter().copied(), &TrainConfig::default());
    for text in KORPUS {
        for id in encoder.encode(text) {
            assert!(!ids.is_special(id), "encode menghasilkan token khusus {id}");
            assert!(!ids.is_reserved(id));
        }
    }
    for token in SPECIAL_TOKENS {
        let id = ids.id_of(token).unwrap_or(u32::MAX);
        assert_eq!(ids.name_of(id), Some(*token));
    }
    assert_eq!(
        ids.base_len() as usize,
        BYTE_VOCAB as usize + SPECIAL_TOKENS.len() + 8
    );
}

#[test]
fn ekspor_impor_tiktoken_setara() {
    let table = train_bpe(KORPUS.iter().copied(), &TrainConfig::default());
    let text = export_tiktoken(&table);
    let back = load_tiktoken(&text).ok();
    assert_eq!(back.as_ref(), Some(&table));
    assert_eq!(text.lines().count(), table.vocab_size());
    assert!(text.lines().next().unwrap_or("").ends_with(" 0"));
}

#[test]
fn latih_deterministik_dan_tidak_bergantung_urutan() {
    let a = train_bpe(KORPUS.iter().copied(), &TrainConfig::default());
    let b = train_bpe(KORPUS.iter().copied(), &TrainConfig::default());
    assert_eq!(a, b);
    let mut reversed: Vec<&str> = KORPUS.to_vec();
    reversed.reverse();
    let c = train_bpe(reversed, &TrainConfig::default());
    assert_eq!(a, c);
}

#[test]
fn masukan_adversarial_selesai_dan_lossless() {
    let encoder = BpeEncoder::trained(["aaaaaaaaaaaaaaaa"], &TrainConfig::default());
    let adversarial = "a".repeat(200_000);
    let ids = encoder.encode(&adversarial);
    assert!(
        ids.len() < adversarial.len(),
        "merge tidak mengurangi token"
    );
    assert_eq!(encoder.decode_bytes(&ids), adversarial.as_bytes());
}

#[test]
fn pra_tokenisasi_jepang_rusia_dan_emoji() {
    assert_eq!(pretokenize("猫が座る"), vec!["猫", "が", "座", "る"]);
    assert_eq!(pretokenize("мама мыла"), vec!["мама", " мыла"]);
    assert_eq!(pretokenize("😀😀x"), vec!["😀😀", "x"]);
    for text in KORPUS {
        let joined: String = pretokenize(text).concat();
        assert_eq!(joined, *text);
    }
}

#[test]
fn kelas_karakter_dasar() {
    assert_eq!(classify(' '), CharClass::Space);
    assert_eq!(classify('7'), CharClass::Digit);
    assert_eq!(classify('Ж'), CharClass::Word);
    assert_eq!(classify('猫'), CharClass::Unspaced);
    assert_eq!(classify('😀'), CharClass::Symbol);
    assert_eq!(classify('!'), CharClass::Punct);
    assert_eq!(classify('\u{0301}'), CharClass::Mark);
    let pretok = PreTokenizer::new(4);
    assert!(pretok.split("abcdefgh").iter().all(|c| c.len() <= 4));
}

#[test]
fn backend_khusus_reference_dipakai() {
    let table = train_bpe(KORPUS.iter().copied(), &TrainConfig::default());
    let names = resolve_backend("reference", &table)
        .map(|encoder| encoder.name())
        .ok();
    assert_eq!(names, Some("bpe-reference"));
    let fallback = resolve_backend("reference", &table).ok();
    let encoded = fallback
        .map(|encoder| encoder.encode("halo dunia"))
        .unwrap_or_default();
    assert!(!encoded.is_empty());
    assert!(resolve_backend("snaptokens", &table).is_err());
    assert!(resolve_backend("gigatoken", &table).is_err());
    assert!(resolve_backend("entah", &table).is_err());
}

#[test]
fn vocab_susut_saat_korpus_kecil() {
    let table = train_bpe(["abab abab"], &TrainConfig::default());
    assert!(table.vocab_size() > table.base_len() as usize);
    assert!(table.merge_count() > 0);
    let mut encoder = BpeEncoder::new();
    encoder.train(&texts, 300);
    assert!(encoder.table().merge_count() >= 10);
    assert!(encoder.vocab_size() <= 300);
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(64))]

    #[test]
    fn proptest_roundtrip_byte_persis(text in ".*") {
        let encoder = BpeEncoder::new();
        let ids = encoder.encode(&text);
        prop_assert_eq!(encoder.decode_bytes(&ids), text.as_bytes());
    }

    #[test]
    fn proptest_pra_tokenisasi_reversibel(text in ".{0,64}") {
        let joined: String = pretokenize(&text).concat();
        prop_assert_eq!(&joined, &text);
    }
}
