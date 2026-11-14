//! TUI VIR (dasbor + layar).

use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Tabs};
use std::io::stdout;
use std::time::Duration;
use vir_common::build_info;

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
    pub fn from_index(i: usize) -> Self {
        match i {
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

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    let mut screen = Screen::Dashboard;

    loop {
        terminal.draw(|f| {
            let area = f.area();
            let titles: Vec<Line> = Screen::titles().iter().map(|t| Line::from(*t)).collect();
            let tabs = Tabs::new(titles)
                .block(Block::default().borders(Borders::ALL).title("VIR"))
                .select(screen.index())
                .highlight_style(Style::default().fg(Color::Yellow));
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(1)])
                .split(area);
            f.render_widget(tabs, chunks[0]);
            let body = match screen {
                Screen::Dashboard => {
                    format!("Dasbor\n{}\n\nTab: 1-8 / ←→\nq: keluar", build_info())
                }
                Screen::Train => "Latih: kurva loss, throughput, status gerbang".to_string(),
                Screen::Generate => "Generate: prompt, level kasar→halus, kandidat MBR".to_string(),
                Screen::Memory => "Memori: pohon, cari, versi, diff".to_string(),
                Screen::Rlm => "RLM: pohon rekursi, REPL, anggaran".to_string(),
                Screen::Eval => "Eval: pass@k, kurva loop".to_string(),
                Screen::Logs => "Log: tracing".to_string(),
                Screen::Settings => "Setelan: tema, keymap".to_string(),
            };
            f.render_widget(
                Paragraph::new(body).block(Block::default().borders(Borders::ALL)),
                chunks[1],
            );
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                match k.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char(c) if c.is_ascii_digit() => {
                        let n = c.to_digit(10).unwrap_or(0) as usize;
                        if (1..=8).contains(&n) {
                            screen = Screen::from_index(n - 1);
                        }
                    }
                    KeyCode::Right => screen = Screen::from_index((screen.index() + 1) % 8),
                    KeyCode::Left => screen = Screen::from_index((screen.index() + 7) % 8),
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
