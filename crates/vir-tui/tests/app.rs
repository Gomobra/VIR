//! F3.8.a/F3.8.g — `update` murni: tab, palet, toast, filter, setelan.

mod common;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::path::Path;
use vir_tui::Screen;
use vir_tui::app::{App, Msg};
use vir_tui::settings::{Row, TuiConfig};
use vir_tui::theme::ThemeName;

use common::Fixture;

fn key(code: KeyCode) -> Msg {
    Msg::Key(KeyEvent::new(code, KeyModifiers::NONE))
}

fn ctrl(character: char) -> Msg {
    Msg::Key(KeyEvent::new(
        KeyCode::Char(character),
        KeyModifiers::CONTROL,
    ))
}

fn press(app: &mut App, code: KeyCode) {
    app.update(key(code));
}

fn ketik(app: &mut App, teks: &str) {
    for character in teks.chars() {
        press(app, KeyCode::Char(character));
    }
}

fn toast_text(app: &App) -> Option<&str> {
    app.toast.as_ref().map(|toast| toast.text.as_str())
}

#[test]
fn digit_pindah_layar_dan_di_luar_terabaikan() {
    let mut fixture = Fixture::empty();
    for index in 1..=8u8 {
        let digit = char::from(b'0' + index);
        press(&mut fixture.app, KeyCode::Char(digit));
        assert_eq!(
            fixture.app.screen,
            Screen::from_index(usize::from(index - 1)),
            "digit {digit}"
        );
    }
    press(&mut fixture.app, KeyCode::Char('9'));
    assert_eq!(
        fixture.app.screen,
        Screen::Settings,
        "digit di luar 1..8 diabaikan"
    );
}

#[test]
fn panah_membungkus_layar() {
    let mut fixture = Fixture::empty();
    press(&mut fixture.app, KeyCode::Left);
    assert_eq!(fixture.app.screen, Screen::Settings, "kiri dari dasbor");
    press(&mut fixture.app, KeyCode::Right);
    assert_eq!(fixture.app.screen, Screen::Dashboard, "kanan dari setelan");
}

#[test]
fn q_keluar_dan_tanda_tanya_bantuan() {
    let mut fixture = Fixture::empty();
    press(&mut fixture.app, KeyCode::Char('?'));
    assert!(fixture.app.help);
    press(&mut fixture.app, KeyCode::Esc);
    assert!(!fixture.app.help, "tombol apa pun menutup bantuan");
    press(&mut fixture.app, KeyCode::Char('q'));
    assert!(fixture.app.quit, "q harus keluar");
}

#[test]
fn palet_perintah_dibuka_dan_dijalankan() {
    let mut fixture = Fixture::empty();
    press(&mut fixture.app, KeyCode::Char(':'));
    assert!(fixture.app.palette.open);
    ketik(&mut fixture.app, "theme");
    assert_eq!(
        fixture.app.palette.current().as_deref(),
        Some("theme:next"),
        "pencarian fuzzy harus menemukan tema"
    );
    press(&mut fixture.app, KeyCode::Enter);
    assert!(
        !fixture.app.palette.open,
        "palet tertutup setelah dijalankan"
    );
    assert_eq!(fixture.app.settings.config.theme, ThemeName::Light);
    assert!(fixture.app.settings.dirty);
    assert_eq!(toast_text(&fixture.app), Some("tema light"));
}

#[test]
fn ctrl_p_tidak_mengetik_p_ke_query() {
    let mut fixture = Fixture::empty();
    fixture.app.update(ctrl('p'));
    assert!(fixture.app.palette.open);
    assert!(fixture.app.palette.query.is_empty(), "query harus kosong");
    fixture.app.update(ctrl('p'));
    assert!(!fixture.app.palette.open, "ctrl+p menutup palet");
}

#[test]
fn perintah_go_dan_quit() {
    let mut fixture = Fixture::empty();
    fixture.app.update(Msg::Command("go:logs".into()));
    assert_eq!(fixture.app.screen, Screen::Logs);
    fixture.app.update(Msg::Command("go:entah".into()));
    assert_eq!(fixture.app.screen, Screen::Logs, "id tak dikenal diabaikan");
    fixture.app.update(Msg::Command(String::new()));
    assert_eq!(fixture.app.screen, Screen::Logs, "id kosong diabaikan");
    fixture.app.update(Msg::Command("quit".into()));
    assert!(fixture.app.quit);
}

#[test]
fn toast_menipis_sampai_hilang() {
    let mut fixture = Fixture::empty();
    fixture.app.notify("uji");
    assert_eq!(
        fixture.app.toast.as_ref().map(|toast| toast.ticks),
        Some(30)
    );
    for _ in 0..30 {
        fixture.app.update(Msg::Tick);
    }
    assert!(fixture.app.toast.is_none(), "toast harus kedaluwarsa");
    assert_eq!(fixture.app.ticks, 30, "tick harus tepat satu per Msg::Tick");
}

#[test]
fn jeda_checkpoint_dan_resize() {
    let mut fixture = Fixture::empty();
    fixture.app.screen = Screen::Train;
    press(&mut fixture.app, KeyCode::Char('p'));
    assert!(fixture.app.paused);
    assert_eq!(fixture.app.status, "latih dijeda");
    press(&mut fixture.app, KeyCode::Char('p'));
    assert!(!fixture.app.paused);
    press(&mut fixture.app, KeyCode::Char('c'));
    assert_eq!(fixture.app.status, "checkpoint manual diminta");
    fixture.app.update(Msg::Resize(10, 10));
    assert_eq!(fixture.app.status, "ukuran terminal berubah");
}

#[test]
fn filter_log_berputar_dan_saring_modul() {
    let mut fixture = Fixture::with_data();
    fixture.app.screen = Screen::Logs;
    assert_eq!(fixture.app.log_filter.level.label(), "ALL");
    press(&mut fixture.app, KeyCode::Char('l'));
    assert_eq!(fixture.app.log_filter.level.label(), "TRACE");
    press(&mut fixture.app, KeyCode::Char('c'));
    assert_eq!(
        fixture.app.log_filter.level.label(),
        "ALL",
        "c membersihkan filter"
    );

    press(&mut fixture.app, KeyCode::Char('/'));
    assert!(fixture.app.filter_mode);
    ketik(&mut fixture.app, "vir_train");
    press(&mut fixture.app, KeyCode::Enter);
    assert!(!fixture.app.filter_mode);
    assert_eq!(fixture.app.log_filter.module, "vir_train");
    let filtered = fixture.app.logs.filtered(&fixture.app.log_filter);
    assert_eq!(filtered.len(), 1, "hanya log vir_train yang lolos");
    assert!(filtered[0].message.contains("step 30"));
}

#[test]
fn setelan_enter_mengubah_dan_menyimpan() {
    let mut fixture = Fixture::empty();
    fixture.app.screen = Screen::Settings;
    assert_eq!(fixture.app.settings.row(), Row::Theme);
    press(&mut fixture.app, KeyCode::Down);
    assert_eq!(fixture.app.settings.row(), Row::ColorMode);
    press(&mut fixture.app, KeyCode::Up);
    press(&mut fixture.app, KeyCode::Enter);
    assert_eq!(fixture.app.settings.config.theme, ThemeName::Light);
    assert!(fixture.app.settings.dirty);

    press(&mut fixture.app, KeyCode::Char('s'));
    assert_eq!(toast_text(&fixture.app), Some("setelan disimpan"));
    assert!(!fixture.app.settings.dirty);
    assert!(fixture.temp.exists("tui.toml"), "setelan harus ditulis");
    let saved = TuiConfig::load(Path::new(&fixture.temp.file("tui.toml")));
    assert_eq!(saved.theme, ThemeName::Light);
}

#[test]
fn memori_pilih_segarkan_dan_cari() {
    let mut fixture = Fixture::with_data();
    fixture.app.screen = Screen::Memory;
    fixture.app.memory.rows = vec![
        (0, "/".into(), "dir".into()),
        (1, "/semantik".into(), "dir".into()),
        (1, "/semantik/rust".into(), "file".into()),
    ];
    press(&mut fixture.app, KeyCode::Down);
    assert_eq!(fixture.app.memory.selected, 1);
    assert_eq!(fixture.app.memory.selected_path, "/semantik");
    press(&mut fixture.app, KeyCode::Up);
    assert_eq!(fixture.app.memory.selected, 0);

    press(&mut fixture.app, KeyCode::Char('r'));
    assert!(
        fixture.app.memory.status.contains("entri"),
        "r menyegarkan pohon: {}",
        fixture.app.memory.status
    );

    press(&mut fixture.app, KeyCode::Char('/'));
    assert!(fixture.app.filter_mode);
    ketik(&mut fixture.app, "rust");
    press(&mut fixture.app, KeyCode::Enter);
    assert!(!fixture.app.filter_mode);
    assert_eq!(fixture.app.memory.search, "rust");
}

#[test]
fn rlm_langkah_dan_batal() {
    let mut fixture = Fixture::empty();
    fixture.app.screen = Screen::Rlm;
    press(&mut fixture.app, KeyCode::Char('n'));
    assert_eq!(fixture.app.status, "langkah RLM diminta");
    press(&mut fixture.app, KeyCode::Char('x'));
    assert_eq!(fixture.app.status, "RLM dibatalkan");
}
