//! Layar Memori: pohon ruang nama, pratinjau, dan hasil pencarian hibrida.

use crate::app::App;
use crate::layout::{LayoutMode, vertical};
use crate::widgets::{panel, truncate};
use ratatui::prelude::*;
use ratatui::widgets::{List, ListItem, ListState, Paragraph};

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let mode = LayoutMode::from_width(area.width);
    let chunks = vertical(area, &[Constraint::Length(3), Constraint::Min(5)]);
    let search = format!(
        "Cari: {} · / untuk saring · Enter buka · r segarkan · {}",
        app.memory.search, app.memory.status
    );
    frame.render_widget(
        Paragraph::new(search)
            .block(panel("Memori"))
            .style(Style::default().fg(app.palette().fg)),
        chunks[0],
    );
    let body = if mode == LayoutMode::Compact {
        vertical(
            chunks[1],
            &[Constraint::Percentage(52), Constraint::Percentage(48)],
        )
    } else {
        crate::layout::horizontal(
            chunks[1],
            &[Constraint::Percentage(42), Constraint::Percentage(58)],
        )
    };
    render_tree(frame, body[0], app);
    render_preview(frame, body[1], app);
}

fn render_tree(frame: &mut Frame, area: Rect, app: &App) {
    let items: Vec<ListItem> = app
        .memory
        .rows
        .iter()
        .map(|(depth, path, kind)| {
            let indent = "  ".repeat(*depth);
            let icon = if kind == "dir" { "▾" } else { "·" };
            ListItem::new(format!("{indent}{icon} {}", truncate(path, 44)))
        })
        .collect();
    let mut state = ListState::default().with_selected(Some(app.memory.selected));
    frame.render_stateful_widget(
        List::new(items)
            .block(panel("Pohon MFS"))
            .highlight_style(
                Style::default()
                    .fg(app.palette().accent)
                    .bg(app.palette().selection),
            )
            .highlight_symbol("› "),
        area,
        &mut state,
    );
}

fn render_preview(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = if app.memory.hits.is_empty() {
        vertical(area, &[Constraint::Percentage(100)])
    } else {
        vertical(
            area,
            &[Constraint::Percentage(68), Constraint::Percentage(32)],
        )
    };
    render_preview_text(frame, chunks[0], app);
    if !app.memory.hits.is_empty() {
        render_hits(frame, chunks[1], app);
    }
}

fn render_preview_text(frame: &mut Frame, area: Rect, app: &App) {
    let title = format!("Pratinjau · {}", truncate(&app.memory.selected_path, 36));
    frame.render_widget(
        Paragraph::new(app.memory.preview.clone())
            .block(panel(title))
            .wrap(ratatui::widgets::Wrap { trim: true })
            .style(Style::default().fg(app.palette().fg)),
        area,
    );
}

fn render_hits(frame: &mut Frame, area: Rect, app: &App) {
    let lines: Vec<Line> = app
        .memory
        .hits
        .iter()
        .map(|hit| {
            Line::from(vec![
                Span::styled(
                    format!("{:.3}", hit.score),
                    Style::default().fg(app.palette().ok),
                ),
                Span::raw(format!("  {}", truncate(&hit.path, 40))),
            ])
        })
        .collect();
    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("Hasil find"))
            .style(Style::default().fg(app.palette().fg)),
        area,
    );
}
