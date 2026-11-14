//! Layar TUI.

pub mod dashboard;
pub mod eval;
pub mod generate;
pub mod logs;
pub mod memory;
pub mod overlay;
pub mod rlm;
pub mod settings;
pub mod train;

use crate::Screen;
use crate::app::App;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Paragraph, Tabs};

pub fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(1)])
        .split(area);
    render_tabs(frame, chunks[0], app);
    match app.screen {
        Screen::Dashboard => dashboard::render(frame, chunks[1], app),
        Screen::Train => train::render(frame, chunks[1], app),
        Screen::Generate => generate::render(frame, chunks[1], app),
        Screen::Memory => memory::render(frame, chunks[1], app),
        Screen::Rlm => rlm::render(frame, chunks[1], app),
        Screen::Eval => eval::render(frame, chunks[1], app),
        Screen::Logs => logs::render(frame, chunks[1], app),
        Screen::Settings => settings::render(frame, chunks[1], app),
    }
    if let Some(toast) = app.toast.as_ref() {
        let rect = Rect {
            x: chunks[1].x,
            y: chunks[1].bottom().saturating_sub(1),
            width: chunks[1].width,
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(toast.text.clone())
                .style(Style::default().fg(app.palette().bg).bg(app.palette().ok)),
            rect,
        );
    }
    if app.palette.open {
        overlay::render_palette(frame, area, app);
    }
    if app.help {
        overlay::render_help(frame, area, app);
    }
}

fn render_tabs(frame: &mut Frame, area: Rect, app: &App) {
    let titles: Vec<Line> = Screen::titles()
        .iter()
        .map(|title| Line::from(*title))
        .collect();
    let tabs = Tabs::new(titles)
        .block(
            Block::bordered()
                .title("VIR")
                .border_style(Style::default().fg(app.palette().accent)),
        )
        .select(app.screen.index())
        .highlight_style(
            Style::default()
                .fg(app.palette().bg)
                .bg(app.palette().accent)
                .add_modifier(Modifier::BOLD),
        );
    frame.render_widget(tabs, area);
}
