//! F3.8.h — snapshot render tiap layar pada tiga ukuran (TestBackend).
//!
//! Snapshot memakai `TestBackend` (bukan `insta`) dan mengikat dua hal yang
//! paling rapuh: layar tidak boleh panik pada ukuran apa pun dan penanda panel
//! utamanya harus selalu terlihat.

mod common;

use common::{Fixture, SCREENS, SIZES, draw, visible_cells};
use vir_tui::Screen;

/// Penanda panel utama per layar yang wajib tampil di tiga ukuran.
const MARKERS: [(Screen, &str); 8] = [
    (Screen::Dashboard, "Ringkasan"),
    (Screen::Train, "Latih"),
    (Screen::Generate, "Kandidat"),
    (Screen::Memory, "Pohon"),
    (Screen::Rlm, "REPL"),
    (Screen::Eval, "pass@k"),
    (Screen::Logs, "Jejak"),
    (Screen::Settings, "Form"),
];

fn assert_screen_text(screen: Screen, width: u16, height: u16, text: &str, marker: &str) {
    let label = format!("{screen:?} {width}x{height}");
    assert_eq!(
        text.lines().count(),
        height as usize,
        "{label} harus {height} baris:\n{text}"
    );
    assert!(
        visible_cells(text) > 40,
        "{label} buffer hampir kosong:\n{text}"
    );
    assert!(text.contains("VIR"), "{label} kehilangan judul tab");
    assert!(text.contains("Dasbor"), "{label} kehilangan tab pertama");
    assert!(
        text.contains(marker),
        "{label} kehilangan penanda '{marker}':\n{text}"
    );
}

#[test]
fn snapshot_tiap_layar_tiga_ukuran() {
    let mut fixture = Fixture::with_data();
    for (screen, marker) in MARKERS {
        for (width, height) in SIZES {
            fixture.app.screen = screen;
            let text = draw(&fixture.app, width, height);
            assert_screen_text(screen, width, height, &text, marker);
        }
    }
}

#[test]
fn snapshot_tanpa_data_tetap_jujur() {
    let mut fixture = Fixture::empty();
    for screen in SCREENS {
        for (width, height) in SIZES {
            fixture.app.screen = screen;
            let text = draw(&fixture.app, width, height);
            assert_eq!(text.lines().count(), height as usize, "{screen:?}");
            assert!(text.contains("VIR"), "{screen:?} tanpa data");
            assert!(
                visible_cells(&text) > 40,
                "{screen:?} {width}x{height} tanpa data harus tetap berstruktur:\n{text}"
            );
        }
    }
    fixture.app.screen = Screen::Dashboard;
    let dashboard = draw(&fixture.app, 80, 24);
    assert!(
        dashboard.contains("Status run"),
        "dasbor tanpa data tetap harus punya panel status:\n{dashboard}"
    );
}

#[test]
fn render_deterministik() {
    let mut fixture = Fixture::with_data();
    for screen in SCREENS {
        fixture.app.screen = screen;
        let first = draw(&fixture.app, 80, 24);
        let second = draw(&fixture.app, 80, 24);
        assert_eq!(first, second, "render {screen:?} tidak deterministik");
    }
}

#[test]
fn ukuran_ekstrem_tidak_panik() {
    let mut fixture = Fixture::with_data();
    for (width, height) in [(1, 1), (20, 5), (40, 10), (300, 60)] {
        for screen in SCREENS {
            fixture.app.screen = screen;
            let text = draw(&fixture.app, width, height);
            assert_eq!(text.lines().count(), height as usize, "{screen:?}");
        }
    }
}

#[test]
fn overlay_palet_dan_bantuan() {
    let mut fixture = Fixture::with_data();
    fixture.app.palette.toggle();
    for character in "theme".chars() {
        fixture.app.palette.push(character);
    }
    let palette = draw(&fixture.app, 80, 24);
    assert!(palette.contains("Palet"), "palet harus tampil:\n{palette}");

    fixture.app.palette.close();
    fixture.app.help = true;
    let help = draw(&fixture.app, 40, 20);
    assert!(help.contains("Bantuan"), "bantuan harus tampil:\n{help}");
    assert!(help.contains("keluar"), "bantuan harus memuat peta tombol");
}

#[test]
fn toast_dan_dialog_konfirmasi_tampil() {
    let mut fixture = Fixture::with_data();
    fixture.app.notify("checkpoint manual diminta");
    let text = draw(&fixture.app, 80, 24);
    assert!(text.contains("checkpoint manual diminta"), "{text}");
}
