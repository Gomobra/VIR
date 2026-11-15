//! F3.8.h — anggaran latensi render per frame (F3.8.a: 30 fps ≈ 33 ms).
//!
//! Anggaran ADR adalah 33 ms/frame; tes ini memakai pagar 150 ms (≈4×) supaya
//! regresi berat (mis. render yang tumbuh kuadratik) tertangkap tanpa flaky.

mod common;

use std::time::Instant;
use vir_tui::Screen;

use common::{Fixture, SCREENS, SIZES, draw};

const FRAME_BUDGET_MS: u128 = 33;
const CEILING_MS: u128 = 150;

#[test]
fn render_di_bawah_anggaran_frame() {
    let mut fixture = Fixture::with_data();
    let mut slowest = (0u128, Screen::Dashboard, 0u16, 0u16);

    for screen in SCREENS {
        fixture.app.screen = screen;
        for (width, height) in SIZES {
            let _ = draw(&fixture.app, width, height);
            let start = Instant::now();
            for _ in 0..5 {
                let _ = draw(&fixture.app, width, height);
            }
            let per_frame = start.elapsed().as_millis() / 5;
            if per_frame > slowest.0 {
                slowest = (per_frame, screen, width, height);
            }
        }
    }

    assert!(
        slowest.0 <= CEILING_MS,
        "render {}x{} pada {:?} butuh {} ms (anggaran {FRAME_BUDGET_MS} ms, pagar {CEILING_MS} ms)",
        slowest.2,
        slowest.3,
        slowest.1,
        slowest.0
    );
}
