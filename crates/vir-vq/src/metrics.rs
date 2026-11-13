//! Evaluasi rekonstruksi (PRD F1.4.e).
//!
//! Akurasi token saja tidak cukup untuk reasoning dan kode, jadi metrik di sini
//! terdiri dari:
//!
//! 1. **Akurasi eksak per level** — [`level_accuracy`], dibandingkan pada token
//!    argmax decoder untuk `ẑ^(k)` setiap level `k = 1..K` (NCM Lampiran G).
//! 2. **Kategori kesalahan** — [`ErrorCategory`] dan [`ErrorReport`], mengikuti
//!    NCM Tabel 11 (entitas, diakritik, angka, nama pengguna, markup, batas
//!    kata) ditambah kategori "kode gagal parse".
//! 3. **Utilisasi dan perplexity per level** — [`LevelReport`], diserialkan ke
//!    JSON untuk TUI ([`reports_json`]).
//!
//! Klasifikasi kategori bekerja pada **teks token**, bukan id, sehingga tidak
//! bergantung pada tokenizer tertentu.

use crate::quantizer::QuantizerStats;
use burn::prelude::*;

/// Akurasi token eksak `benar / total` (0 bila kosong).
pub fn token_accuracy(predicted: &[i64], expected: &[i64]) -> f64 {
    let n = predicted.len().min(expected.len());
    if n == 0 {
        return 0.0;
    }
    let hit = predicted
        .iter()
        .zip(expected.iter())
        .filter(|(p, e)| p == e)
        .count();
    hit as f64 / n as f64
}

/// Akurasi dari dua tensor indeks berbentuk sama (`[N]` atau `[B, L]`).
pub fn tensor_accuracy(predicted: &Tensor<2, Int>, expected: &Tensor<2, Int>) -> f64 {
    let p: Vec<i64> = predicted
        .clone()
        .into_data()
        .try_to_vec::<i64>()
        .unwrap_or_default();
    let e: Vec<i64> = expected
        .clone()
        .into_data()
        .try_to_vec::<i64>()
        .unwrap_or_default();
    token_accuracy(&p, &e)
}

/// Akurasi level `k` dari logits decoder (`[B, L, V]`) terhadap target.
pub fn level_accuracy(logits: &Tensor<3>, target: &Tensor<2, Int>) -> f64 {
    let [b, l, _] = logits.dims();
    let predicted = logits.clone().argmax(2).reshape([b, l]);
    tensor_accuracy(&predicted, target)
}

/// Kategori kesalahan (NCM Tabel 11 + "kode gagal parse").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Token benar (bukan kesalahan; dipakai sebagai kategori pembanding).
    Exact,
    /// Entitas bernama (inisial kapital di tengah, nama, organisasi).
    Entity,
    /// Diakritik (mis. `é`, `й`, かな di luar BMP ASCII).
    Diacritic,
    /// Angka (digit, pemisah ribuan, desimal, persen).
    Number,
    /// Nama pengguna / tagar / URL.
    Username,
    /// Markup (`<`, `>`, `{`, `}`, `]`, `"`).
    Markup,
    /// Batas kata (spasi, tanda baca, token kosong).
    WordBoundary,
    /// Kode yang gagal parse setelah rekonstruksi.
    CodeParse,
    /// Selain itu.
    Other,
}

impl ErrorCategory {
    /// Nama ringkas untuk JSON/TUI.
    pub fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Entity => "entity",
            Self::Diacritic => "diacritic",
            Self::Number => "number",
            Self::Username => "username",
            Self::Markup => "markup",
            Self::WordBoundary => "word_boundary",
            Self::CodeParse => "code_parse",
            Self::Other => "other",
        }
    }

    /// Semua kategori non-`Exact`, urut stabil.
    pub fn all() -> [Self; 8] {
        [
            Self::Entity,
            Self::Diacritic,
            Self::Number,
            Self::Username,
            Self::Markup,
            Self::WordBoundary,
            Self::CodeParse,
            Self::Other,
        ]
    }
}

/// True bila token mengandung huruf di luar ASCII (diakritik, Cyrillic, CJK).
pub fn has_diacritic(token: &str) -> bool {
    token.chars().any(|c| c.is_alphabetic() && !c.is_ascii())
}

/// True bila token hanya berisi angka dan pemisah angka.
pub fn is_number(token: &str) -> bool {
    let mut seen_digit = false;
    for c in token.chars() {
        if c.is_ascii_digit() {
            seen_digit = true;
        } else if !matches!(c, '.' | ',' | '-' | '+' | '%' | '_') {
            return false;
        }
    }
    seen_digit
}

/// True bila token membawa markup atau tanda baca struktural.
pub fn is_markup(token: &str) -> bool {
    token
        .chars()
        .any(|c| matches!(c, '<' | '>' | '{' | '}' | '[' | ']' | '"' | '\'' | '|' | '&'))
}

/// True bila token adalah nama pengguna, tagar, atau URL.
pub fn is_username(token: &str) -> bool {
    let t = token.trim();
    t.starts_with('@')
        || t.starts_with('#')
        || t.starts_with("http://")
        || t.starts_with("https://")
}

/// True bila token adalah batas kata: spasi/marka kata atau tanda baca.
pub fn is_word_boundary(token: &str) -> bool {
    if token.is_empty() {
        return true;
    }
    token
        .chars()
        .all(|c| c.is_whitespace() || c.is_ascii_punctuation())
}

/// Klasifikasi satu token berdasarkan bentuknya (NCM Tabel 11).
pub fn classify_token(token: &str) -> ErrorCategory {
    if is_username(token) {
        return ErrorCategory::Username;
    }
    if is_number(token) {
        return ErrorCategory::Number;
    }
    if has_diacritic(token) {
        return ErrorCategory::Diacritic;
    }
    if is_markup(token) {
        return ErrorCategory::Markup;
    }
    if is_word_boundary(token) {
        return ErrorCategory::WordBoundary;
    }
    ErrorCategory::Entity
}

/// Hitungan kesalahan per kategori.
#[derive(Debug, Clone)]
pub struct ErrorReport {
    counts: Vec<u64>,
}

impl Default for ErrorReport {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorReport {
    /// Laporan kosong.
    pub fn new() -> Self {
        Self {
            counts: vec![0; ErrorCategory::all().len()],
        }
    }

    fn slot(category: ErrorCategory) -> usize {
        ErrorCategory::all()
            .iter()
            .position(|c| *c == category)
            .unwrap_or(0)
    }

    /// Catat satu kesalahan. `expected` adalah token acuan; `code_parse`
    /// menandai kegagalan parse pada kode hasil rekonstruksi.
    pub fn record(&mut self, expected: &str, code_parse: bool) {
        let category = if code_parse {
            ErrorCategory::CodeParse
        } else {
            classify_token(expected)
        };
        let slot = Self::slot(category);
        self.counts[slot] += 1;
    }

    /// Catat satu kesalahan dengan kategori eksplisit.
    pub fn record_category(&mut self, category: ErrorCategory) {
        let slot = Self::slot(category);
        self.counts[slot] += 1;
    }

    /// Jumlah kesalahan pada satu kategori.
    pub fn count(&self, category: ErrorCategory) -> u64 {
        self.counts[Self::slot(category)]
    }

    /// Total kesalahan tercatat.
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Pasangan kategori → jumlah, urut stabil.
    pub fn by_category(&self) -> Vec<(ErrorCategory, u64)> {
        ErrorCategory::all()
            .iter()
            .map(|c| (*c, self.count(*c)))
            .collect()
    }

    /// JSON satu baris (TUI/artefak).
    pub fn to_json(&self) -> String {
        let body: Vec<String> = self
            .by_category()
            .iter()
            .map(|(c, n)| format!("\"{}\":{}", c.name(), n))
            .collect();
        format!("{{{}}}", body.join(","))
    }
}

/// Metrik satu level (akurasi + utilisasi + kategori kesalahan).
#[derive(Debug, Clone)]
pub struct LevelReport {
    /// Indeks level 1-based.
    pub level: usize,
    /// Panjang level `ℓ_k`.
    pub length: usize,
    /// Akurasi token eksak pada level ini.
    pub accuracy: f64,
    /// Statistik codebook level ini.
    pub stats: QuantizerStats,
    /// Kesalahan per kategori.
    pub errors: ErrorReport,
}

impl LevelReport {
    pub fn new(level: usize, length: usize, accuracy: f64, stats: QuantizerStats) -> Self {
        Self {
            level,
            length,
            accuracy,
            stats,
            errors: ErrorReport::new(),
        }
    }

    /// JSON satu baris.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"level\":{},\"length\":{},\"accuracy\":{:.5},\"stats\":{},\"errors\":{}}}",
            self.level,
            self.length,
            self.accuracy,
            self.stats.to_json(),
            self.errors.to_json()
        )
    }
}

/// JSON array satu baris untuk seluruh level (F1.4.e langkah 3).
pub fn reports_json(reports: &[LevelReport]) -> String {
    let body: Vec<String> = reports.iter().map(LevelReport::to_json).collect();
    format!("[{}]", body.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accuracy_counts_exact_matches() {
        assert!((token_accuracy(&[1, 2, 3], &[1, 5, 3]) - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(token_accuracy(&[], &[]), 0.0);
        // Panjang berbeda: hanya sepanjang yang lebih pendek.
        assert_eq!(token_accuracy(&[1, 2, 3], &[1, 2]), 1.0);
    }

    #[test]
    fn category_rules_are_ordered() {
        assert_eq!(classify_token("@user"), ErrorCategory::Username);
        assert_eq!(classify_token("#topik"), ErrorCategory::Username);
        assert_eq!(classify_token("3.14"), ErrorCategory::Number);
        assert_eq!(classify_token("1,5%"), ErrorCategory::Number);
        assert_eq!(classify_token("café"), ErrorCategory::Diacritic);
        assert_eq!(classify_token("Москва"), ErrorCategory::Diacritic);
        assert_eq!(classify_token("<b>"), ErrorCategory::Markup);
        assert_eq!(classify_token(" "), ErrorCategory::WordBoundary);
        assert_eq!(classify_token(","), ErrorCategory::WordBoundary);
        assert_eq!(classify_token("Jakarta"), ErrorCategory::Entity);
    }

    #[test]
    fn report_groups_by_category() {
        let mut report = ErrorReport::new();
        report.record("1990", false);
        report.record("café", false);
        report.record("x", true);
        report.record("Jakarta", false);
        assert_eq!(report.total(), 4);
        assert_eq!(report.count(ErrorCategory::Number), 1);
        assert_eq!(report.count(ErrorCategory::Diacritic), 1);
        assert_eq!(report.count(ErrorCategory::CodeParse), 1);
        assert_eq!(report.count(ErrorCategory::Entity), 1);
        assert_eq!(report.count(ErrorCategory::Markup), 0);
    }

    #[test]
    fn report_json_lists_every_category() {
        let report = ErrorReport::new();
        let json = report.to_json();
        for category in ErrorCategory::all() {
            assert!(json.contains(&format!("\"{}\":0", category.name())));
        }
    }

    #[test]
    fn level_report_json_has_level_fields() {
        let report = LevelReport::new(3, 24, 0.9995, QuantizerStats::from_counts(&[2, 1]));
        let json = report.to_json();
        assert!(json.contains("\"level\":3"));
        assert!(json.contains("\"length\":24"));
        assert!(json.contains("\"accuracy\":0.99950"));
        assert!(reports_json(&[report]).starts_with('['));
    }

    #[test]
    fn tensor_accuracy_matches_slice() {
        let device = Device::default();
        let pred = Tensor::<2, Int>::from_ints([[1i64, 2], [3, 4]], &device);
        let exp = Tensor::<2, Int>::from_ints([[1i64, 9], [3, 4]], &device);
        assert!((tensor_accuracy(&pred, &exp) - 0.75).abs() < 1e-9);
    }
}