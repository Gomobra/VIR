//! TUI VIR (dasbor latihan versi pertama).

use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};
use std::io::stdout;
use std::time::Duration;
use vir_common::build_info;

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    loop {
        terminal.draw(|f| {
            let area = f.area();
            let para = Paragraph::new(format!(
                "VIR Vbeta TUI\n{}\n\nTekan 'q' untuk keluar.",
                build_info()
            ))
            .block(Block::default().title("Dasbor").borders(Borders::ALL));
            f.render_widget(para, area);
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press && k.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
