//! F3.8.g — responsif: tiga tata letak berdasarkan lebar terminal.

mod common;

use common::{Fixture, draw, row_of};
use ratatui::layout::{Constraint, Rect};
use vir_tui::layout::{LayoutMode, body};

#[test]
fn mode_dari_lebar_ada_tiga_ambang() {
    assert_eq!(LayoutMode::from_width(0), LayoutMode::Compact);
    assert_eq!(LayoutMode::from_width(40), LayoutMode::Compact);
    assert_eq!(LayoutMode::from_width(59), LayoutMode::Compact);
    assert_eq!(LayoutMode::from_width(60), LayoutMode::Medium);
    assert_eq!(LayoutMode::from_width(100), LayoutMode::Medium);
    assert_eq!(LayoutMode::from_width(101), LayoutMode::Wide);
    assert_eq!(LayoutMode::from_width(200), LayoutMode::Wide);
}

#[test]
fn jumlah_kolom_sesuai_mode() {
    assert_eq!(LayoutMode::Compact.columns(), 1);
    assert_eq!(LayoutMode::Medium.columns(), 2);
    assert_eq!(LayoutMode::Wide.columns(), 3);
}

#[test]
fn compact_menumpuk_panel_secara_vertikal() {
    let area = Rect::new(0, 0, 40, 20);
    let chunks = body(
        area,
        LayoutMode::Compact,
        Constraint::Percentage(50),
        Constraint::Percentage(50),
    );
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].x, chunks[1].x, "compact: x harus sama");
    assert_eq!(chunks[0].width, chunks[1].width, "compact: lebar penuh");
    assert!(chunks[0].y < chunks[1].y, "compact: panel bertumpuk");
}

#[test]
fn medium_dan_wide_berdampingan() {
    for mode in [LayoutMode::Medium, LayoutMode::Wide] {
        let area = Rect::new(0, 0, 80, 24);
        let chunks = body(
            area,
            mode,
            Constraint::Percentage(40),
            Constraint::Percentage(60),
        );
        assert_eq!(chunks.len(), 2, "{mode:?}");
        assert_eq!(chunks[0].y, chunks[1].y, "{mode:?}: y harus sama");
        assert!(chunks[0].x < chunks[1].x, "{mode:?}: panel berdampingan");
        assert!(
            chunks[0].right() <= chunks[1].left(),
            "{mode:?}: panel tidak boleh tumpang tindih"
        );
    }
}

#[test]
fn tata_layar_selalu_di_dalam_area() {
    for mode in [LayoutMode::Compact, LayoutMode::Medium, LayoutMode::Wide] {
        let area = Rect::new(3, 2, 44, 18);
        for constraint in [Constraint::Length(10), Constraint::Min(5)] {
            for chunk in body(area, mode, constraint, Constraint::Min(5)).iter() {
                assert!(chunk.x >= area.x && chunk.y >= area.y, "{mode:?}");
                assert!(chunk.right() <= area.right(), "{mode:?}");
                assert!(chunk.bottom() <= area.bottom(), "{mode:?}");
            }
        }
    }
}

#[test]
fn dasbor_berubah_tumpuk_ke_damping_menurut_lebar() {
    let fixture = Fixture::with_data();
    let compact = draw(&fixture.app, 50, 24);
    let wide = draw(&fixture.app, 120, 40);

    let compact_summary = row_of(&compact, "Ringkasan").expect("Ringkasan di compact");
    let compact_activity = row_of(&compact, "Aktivitas").expect("Aktivitas di compact");
    assert!(
        compact_summary < compact_activity,
        "compact harus menumpuk:\n{compact}"
    );

    let wide_summary = row_of(&wide, "Ringkasan").expect("Ringkasan di wide");
    let wide_activity = row_of(&wide, "Aktivitas").expect("Aktivitas di wide");
    assert_eq!(
        wide_summary, wide_activity,
        "wide harus berdampingan:\n{wide}"
    );
}
