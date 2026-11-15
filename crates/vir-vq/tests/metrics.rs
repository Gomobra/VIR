//! F1.4.e — metrik akurasi, kategori galat, JSON (PRD F1.4.e).

use vir_vq::metrics::{
    ErrorCategory, ErrorReport, classify_token, has_diacritic, is_markup, is_number, is_username,
    token_accuracy,
};

#[test]
fn akurasi_token_numerik() {
    assert_eq!(token_accuracy(&[1, 2, 3, 4], &[1, 2, 9, 4]), 0.75);
    assert_eq!(token_accuracy(&[1, 2, 3], &[1, 9, 3]), 2.0 / 3.0);
    assert_eq!(token_accuracy(&[], &[]), 0.0);
    assert_eq!(token_accuracy(&[1], &[1]), 1.0);
}

#[test]
fn klasifikasi_bidang_galat() {
    assert!(is_number("123"));
    assert!(!is_number("abc"));
    assert!(is_markup("<b>"));
    assert!(is_username("@budi"));
    assert!(!has_diacritic("plain"));
    assert!(has_diacritic("café"));
}

#[test]
fn kategori_dihitung_dan_dijumlahkan() {
    let mut report = ErrorReport::new();
    report.record("angka 42", false);
    report.record_category(ErrorCategory::Number);
    report.record_category(ErrorCategory::Entity);
    assert_eq!(report.total(), 3);
    assert_eq!(report.count(ErrorCategory::Number), 1);
    let json = report.to_json();
    assert!(json.contains("number"), "json memuat kategori: {json}");
}

#[test]
fn classify_token_empat_kategori() {
    assert_eq!(classify_token("12345"), ErrorCategory::Number);
    assert_eq!(classify_token("<div>"), ErrorCategory::Markup);
    assert_eq!(classify_token("@pengguna"), ErrorCategory::Username);
}

#[test]
fn semua_kategori_punya_nama() {
    for c in ErrorCategory::all() {
        assert!(!c.name().is_empty());
    }
}
