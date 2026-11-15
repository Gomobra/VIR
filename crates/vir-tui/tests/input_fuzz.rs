//! F3.8.h — input acak (tombol + resize) tidak boleh membuat panik (R9).
//!
//! Invarian yang dijaga: tick terhitung tepat, layar dan pilihan selalu di
//! dalam rentang, dan render tetap aman pada state apa pun.

mod common;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use proptest::prelude::*;
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::layout::Rect;
use vir_tui::app::Msg;
use vir_tui::screens;
use vir_tui::settings::Row;

use common::{Fixture, SCREENS};

const CODES: [KeyCode; 12] = [
    KeyCode::Char('q'),
    KeyCode::Char('?'),
    KeyCode::Char(':'),
    KeyCode::Char('p'),
    KeyCode::Char('1'),
    KeyCode::Char('8'),
    KeyCode::Char('9'),
    KeyCode::Left,
    KeyCode::Right,
    KeyCode::Up,
    KeyCode::Down,
    KeyCode::Esc,
];

fn arb_key() -> impl Strategy<Value = KeyEvent> {
    (0u8..CODES.len() as u8, 0u8..8).prop_map(|(index, bits)| {
        let code = CODES[usize::from(index) % CODES.len()];
        KeyEvent::new(code, KeyModifiers::from_bits_truncate(bits))
    })
}

fn arb_msg() -> BoxedStrategy<Msg> {
    prop_oneof![
        arb_key().prop_map(Msg::Key),
        (0u16..200, 0u16..60).prop_map(|(width, height)| Msg::Resize(width, height)),
        Just(Msg::Tick),
    ]
    .boxed()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn input_acak_tidak_panik_dan_invariant_terjaga(
        messages in prop::collection::vec(arb_msg(), 1usize..40),
        width in 10u16..200,
        height in 5u16..60,
    ) {
        let mut fixture = Fixture::empty();
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal");
        let mut ticks = 0u64;

        for msg in &messages {
            fixture.app.update(msg.clone());
            if matches!(msg, Msg::Tick) {
                ticks = ticks.wrapping_add(1);
            }
            terminal
                .draw(|frame| screens::render(frame, &fixture.app))
                .expect("render setelah input acak");
        }

        prop_assert_eq!(fixture.app.ticks, ticks, "tick harus satu per Msg::Tick");
        prop_assert!(fixture.app.screen.index() < 8);
        prop_assert!(fixture.app.settings.selected < Row::ALL.len());
        prop_assert!(fixture.app.palette.selected < fixture.app.palette.filtered().len().max(1));
        if let Some(toast) = fixture.app.toast.as_ref() {
            prop_assert!(toast.ticks <= 30, "umur toast harus terbatas");
        }
        prop_assert!(fixture.app.memory.selected < fixture.app.memory.rows.len().max(1));
        let nodes = fixture.app.rlm.snapshot().nodes.len();
        prop_assert!(nodes == 0 || fixture.app.rlm.snapshot().selected < nodes);
    }

    #[test]
    fn resize_acak_tidak_panik(
        sizes in prop::collection::vec((10u16..200, 5u16..60), 1usize..20),
    ) {
        let mut fixture = Fixture::with_data();
        for screen in SCREENS {
            fixture.app.screen = screen;
            let mut terminal =
                Terminal::new(TestBackend::new(sizes[0].0, sizes[0].1)).expect("terminal");
            for (width, height) in &sizes {
                terminal
                    .resize(Rect::new(0, 0, *width, *height))
                    .expect("resize terminal");
                terminal
                    .draw(|frame| screens::render(frame, &fixture.app))
                    .expect("render setelah resize");
            }
        }
    }
}
