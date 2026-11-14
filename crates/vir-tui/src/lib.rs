//! TUI VIR: arsitektur Elm dengan layar latihan, generate, memori, RLM,
//! evaluasi, log, dan pengaturan.

pub mod app;
pub mod eval;
pub mod filter;
pub mod generate;
pub mod json;
pub mod keymap;
pub mod layout;
pub mod logs;
pub mod memory_view;
pub mod metrics;
pub mod palette;
pub mod paths;
pub mod rlm;
pub mod screens;
pub mod settings;
pub mod theme;
pub mod widgets;

use app::{App, AppPaths};
use crossterm::ExecutableCommand;
use crossterm::event::{self, Event};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use std::io::{Stdout, stdout};
use std::time::Duration;

pub use app::Msg;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Dashboard,
    Train,
    Generate,
    Memory,
    Rlm,
    Eval,
    Logs,
    Settings,
}

impl Screen {
    pub fn titles() -> &'static [&'static str] {
        &[
            "Dasbor", "Latih", "Generate", "Memori", "RLM", "Eval", "Log", "Setelan",
        ]
    }

    pub fn index(self) -> usize {
        match self {
            Screen::Dashboard => 0,
            Screen::Train => 1,
            Screen::Generate => 2,
            Screen::Memory => 3,
            Screen::Rlm => 4,
            Screen::Eval => 5,
            Screen::Logs => 6,
            Screen::Settings => 7,
        }
    }

    pub fn from_index(index: usize) -> Self {
        match index {
            0 => Screen::Dashboard,
            1 => Screen::Train,
            2 => Screen::Generate,
            3 => Screen::Memory,
            4 => Screen::Rlm,
            5 => Screen::Eval,
            6 => Screen::Logs,
            _ => Screen::Settings,
        }
    }
}

/// Jalankan TUI pada path bawaan (bisa ditimpa env `VIR_*`).
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    run_with_paths(AppPaths::default())
}

pub fn run_with_paths(paths: AppPaths) -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    install_panic_hook();
    let app = App::with_paths(paths);
    let result = run_loop(&mut terminal, app);
    restore_terminal();
    result
}

fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        previous(info);
    }));
}

fn restore_terminal() {
    let _ = disable_raw_mode();
    let _ = stdout().execute(LeaveAlternateScreen);
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    mut app: App,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        let fps = if app.settings.config.low_bandwidth {
            app.settings.config.low_bandwidth_fps
        } else {
            app.settings.config.fps
        };
        let interval = Duration::from_millis(1000 / u64::from(fps.max(1)));
        terminal.draw(|frame| screens::render(frame, &app))?;
        if app.quit {
            break;
        }
        if event::poll(interval)? {
            match event::read()? {
                Event::Key(key) => app.update(Msg::Key(key)),
                Event::Resize(_, _) => app.update(Msg::Resize(0, 0)),
                _ => {}
            }
        } else {
            app.update(Msg::Tick);
        }
    }
    Ok(())
}
