//! TUI VIR (dasbor nyata: JSONL tail + chart braille + status).

use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Sparkline, Tabs};
use std::fs::File;
use std::io::stdout;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
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

/// Pembaca metrik JSONL: tail + parse loss.
pub struct MetricsReader {
    path: String,
    pos: u64,
    losses: Vec<f64>,
}

impl MetricsReader {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            pos: 0,
            losses: Vec::new(),
        }
    }

    pub fn poll(&mut self) {
        let Ok(f) = File::open(&self.path) else {
            return;
        };
        let mut reader = BufReader::new(f);
        if reader.seek(SeekFrom::Start(self.pos)).is_err() {
            return;
        }
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    if let Some(loss) = parse_loss(&line) {
                        self.losses.push(loss);
                        if self.losses.len() > 240 {
                            self.losses.remove(0);
                        }
                    }
                }
                Err(_) => break,
            }
        }
        self.pos = reader.stream_position().unwrap_or(self.pos);
    }

    pub fn sparkline(&self) -> Sparkline<'_> {
        let data: Vec<u64> = self.losses.iter().map(|&l| (l * 100.0) as u64).collect();
        Sparkline::default().data(&data)
    }

    pub fn latest_loss(&self) -> Option<f64> {
        self.losses.last().copied()
    }

    pub fn count(&self) -> usize {
        self.losses.len()
    }
}

fn parse_loss(line: &str) -> Option<f64> {
    // cari "loss": <num>
    let key = "\"loss\":";
    let i = line.find(key)?;
    let rest = &line[i + key.len()..];
    let end = rest.find([',', '}', ' ']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    let mut screen = Screen::Dashboard;
    let mut reader = MetricsReader::new("metrics.jsonl");

    loop {
        reader.poll();
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

            match screen {
                Screen::Dashboard => {
                    let chunks2 = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Length(3), Constraint::Min(1)])
                        .split(chunks[1]);
                    let status = format!(
                        "{} | loss={} | steps={} | backend={}",
                        build_info(),
                        reader
                            .latest_loss()
                            .map(|l| format!("{l:.4}"))
                            .unwrap_or("-".into()),
                        reader.count(),
                        "cpu"
                    );
                    f.render_widget(
                        Paragraph::new(status)
                            .block(Block::default().borders(Borders::ALL).title("Status")),
                        chunks2[0],
                    );
                    f.render_widget(
                        reader
                            .sparkline()
                            .block(Block::default().borders(Borders::ALL).title("Loss")),
                        chunks2[1],
                    );
                }
                Screen::Train => {
                    f.render_widget(
                        Paragraph::new("Latih: JSONL tail + chart (F3.8.b)")
                            .block(Block::default().borders(Borders::ALL)),
                        chunks[1],
                    );
                }
                Screen::Generate => {
                    f.render_widget(
                        Paragraph::new("Generate: prompt, level kasar→halus (F3.8.c)")
                            .block(Block::default().borders(Borders::ALL)),
                        chunks[1],
                    );
                }
                Screen::Memory => {
                    f.render_widget(
                        Paragraph::new("Memori: pohon, cari, versi (F3.8.d)")
                            .block(Block::default().borders(Borders::ALL)),
                        chunks[1],
                    );
                }
                Screen::Rlm => {
                    f.render_widget(
                        Paragraph::new("RLM: pohon rekursi, REPL (F3.8.e)")
                            .block(Block::default().borders(Borders::ALL)),
                        chunks[1],
                    );
                }
                Screen::Eval => {
                    f.render_widget(
                        Paragraph::new("Eval: pass@k, kurva loop (F3.8.f)")
                            .block(Block::default().borders(Borders::ALL)),
                        chunks[1],
                    );
                }
                Screen::Logs => {
                    f.render_widget(
                        Paragraph::new("Log: tracing (F3.8.f)")
                            .block(Block::default().borders(Borders::ALL)),
                        chunks[1],
                    );
                }
                Screen::Settings => {
                    f.render_widget(
                        Paragraph::new("Setelan: tema, keymap (F3.8.f)")
                            .block(Block::default().borders(Borders::ALL)),
                        chunks[1],
                    );
                }
            }
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                match k.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char(c) if (1..=8).contains(&c.to_digit(10).unwrap_or(0)) => {
                        let n = c.to_digit(10).unwrap_or(0) as usize;
                        screen = Screen::from_index(n - 1);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_loss_extracts_value() {
        let line = r#"{"step":1,"loss":0.1234,"lr":0.001}"#;
        assert_eq!(parse_loss(line), Some(0.1234));
    }

    #[test]
    fn parse_loss_missing() {
        assert_eq!(parse_loss("{\"step\":1}"), None);
    }
}
