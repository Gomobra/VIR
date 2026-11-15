//! F3.8.g — tema, palet, dan `NO_COLOR`.

use ratatui::style::Color;
use vir_tui::theme::{ColorMode, Palette, ThemeName, detect_mode, mode_for, no_color};

fn warna_penting(palette: &Palette) -> [Color; 4] {
    [palette.fg, palette.accent, palette.ok, palette.error]
}

#[test]
fn parse_dan_label_setiap_tema() {
    for theme in ThemeName::ALL {
        let parsed = ThemeName::parse(theme.label()).expect("label harus bisa diparse");
        assert_eq!(parsed, theme, "label {}", theme.label());
    }
    assert_eq!(ThemeName::parse("HC"), Some(ThemeName::HighContrast));
    assert_eq!(ThemeName::parse("colorblind"), Some(ThemeName::Colorblind));
    assert_eq!(ThemeName::parse("neon"), None);
    assert_eq!(ThemeName::parse(""), None);
}

#[test]
fn siklus_tema_berputar_penuh() {
    let urutan = [
        ThemeName::Light,
        ThemeName::HighContrast,
        ThemeName::Colorblind,
        ThemeName::Dark,
    ];
    let mut theme = ThemeName::Dark;
    for harapan in urutan {
        theme = theme.next();
        assert_eq!(theme, harapan);
    }
    theme = ThemeName::Dark;
    for _ in urutan {
        theme = theme.prev();
    }
    assert_eq!(theme, ThemeName::Dark, "prev() harus periodik 4");
}

#[test]
fn palet_tiap_tema_berbeda_dan_tiap_warna_punya_kontras() {
    let dark = Palette::new(ThemeName::Dark, ColorMode::Truecolor, false);
    let light = Palette::new(ThemeName::Light, ColorMode::Truecolor, false);
    let kontras = Palette::new(ThemeName::HighContrast, ColorMode::Truecolor, false);
    let buta_warna = Palette::new(ThemeName::Colorblind, ColorMode::Truecolor, false);

    assert_ne!(dark.bg, light.bg, "gelap dan terang harus berbeda");
    assert_eq!(kontras.bg, Color::Black);
    assert_ne!(buta_warna.loss, dark.loss, "ramah buta warna harus beda");

    for palette in [dark, light, kontras, buta_warna] {
        for color in warna_penting(&palette) {
            assert_ne!(color, palette.bg, "warna depan harus kontras dengan latar");
        }
        assert_ne!(palette.selection, palette.bg);
        assert_ne!(palette.muted, palette.bg);
    }
}

#[test]
fn mode_256_dan_none_disesuaikan_seragam() {
    for theme in ThemeName::ALL {
        let indexed = Palette::new(theme, ColorMode::Ansi256, false);
        assert!(matches!(indexed.bg, Color::Indexed(_)), "{theme:?}");
        assert!(matches!(indexed.error, Color::Indexed(_)), "{theme:?}");

        let none = Palette::new(theme, ColorMode::None, false);
        for color in warna_penting(&none) {
            assert_eq!(color, Color::Reset, "{theme:?} tanpa warna");
        }
        assert_eq!(none.bg, Color::Reset);
        assert_eq!(none.selection, Color::Reset);
    }
}

#[test]
fn colorblind_hanya_menimpa_warna_seri_gelap() {
    let campur = Palette::new(ThemeName::Colorblind, ColorMode::Truecolor, true);
    let murni = Palette::new(ThemeName::Colorblind, ColorMode::Truecolor, false);
    assert_eq!(campur, murni, "tema colorblind tidak boleh berubah");

    let dark_cb = Palette::new(ThemeName::Dark, ColorMode::Truecolor, true);
    let dark = Palette::new(ThemeName::Dark, ColorMode::Truecolor, false);
    assert_ne!(dark_cb.loss, dark.loss);
    assert_eq!(dark_cb.loss, murni.loss);
    assert_eq!(dark_cb.lr, murni.lr);
}

#[test]
fn no_color_selalu_menang_atas_setelan() {
    for configured in [ColorMode::Truecolor, ColorMode::Ansi256, ColorMode::None] {
        assert_eq!(mode_for(true, configured), ColorMode::None);
    }
    assert_eq!(mode_for(false, ColorMode::Truecolor), ColorMode::Truecolor);
    assert_eq!(mode_for(false, ColorMode::Ansi256), ColorMode::Ansi256);
    assert_eq!(detect_mode(true), ColorMode::None);
    assert_eq!(detect_mode(false), ColorMode::Truecolor);
}

#[test]
fn no_color_mengikuti_semantik_no_color_org() {
    let terpasang = std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty());
    assert_eq!(no_color(), terpasang, "NO_COLOR kosong bukan tanpa warna");
}

#[test]
fn mode_warna_berantai() {
    let mode = mode_for(true, ColorMode::Truecolor);
    let palette = Palette::new(ThemeName::Dark, mode, false);
    assert_eq!(palette.fg, Color::Reset);
    assert_eq!(palette.accent, Color::Reset);
    assert_eq!(ColorMode::Truecolor.next(), ColorMode::Ansi256);
    assert_eq!(ColorMode::Ansi256.next(), ColorMode::None);
    assert_eq!(ColorMode::None.next(), ColorMode::Truecolor);
    assert_eq!(ColorMode::Truecolor.prev(), ColorMode::None);
}
