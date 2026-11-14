//! Layar RLM: pohon rekursi, meter anggaran, sel REPL, sorotan `mem`.

use crate::app::App;
use crate::layout::{LayoutMode, vertical};
use crate::widgets::{key_value, panel, truncate};
use ratatui::prelude::*;
use ratatui::widgets::{LineGauge, List, ListItem, ListState, Paragraph};

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let mode = LayoutMode::from_width(area.width);
    let chunks = vertical(area, &[Constraint::Length(4), Constraint::Min(5)]);
    render_budget(frame, chunks[0], app);
    let body = if mode == LayoutMode::Compact {
        vertical(
            chunks[1],
            &[Constraint::Percentage(48), Constraint::Percentage(52)],
        )
    } else {
        crate::layout::horizontal(
            chunks[1],
            &[Constraint::Percentage(45), Constraint::Percentage(55)],
        )
    };
    render_tree(frame, body[0], app);
    render_repl(frame, body[1], app);
}

fn render_budget(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = crate::layout::horizontal(
        area,
        &[
            Constraint::Length(28),
            Constraint::Percentage(30),
            Constraint::Percentage(30),
            Constraint::Min(20),
        ],
    );
    render_budget_status(frame, chunks[0], app);
    render_budget_meters(frame, &chunks[1..3], app);
    render_budget_footer(frame, chunks[3], app);
}

fn render_budget_status(frame: &mut Frame, area: Rect, app: &App) {
    let rlm = app.rlm.snapshot();
    frame.render_widget(
        Paragraph::new(format!(
            "RLM {}\nnode {} / {} panggilan",
            rlm.status, rlm.calls, rlm.max_calls
        ))
        .block(panel("Anggaran"))
        .style(Style::default().fg(app.palette().fg)),
        area,
    );
}

fn render_budget_meters(frame: &mut Frame, areas: &[Rect], app: &App) {
    let rlm = app.rlm.snapshot();
    frame.render_widget(
        LineGauge::default()
            .block(panel("Kedalaman"))
            .ratio(rlm.depth_ratio())
            .label(format!("{} / {}", rlm.depth, rlm.max_depth))
            .style(Style::default().fg(app.palette().lr)),
        areas[0],
    );
    frame.render_widget(
        LineGauge::default()
            .block(panel("Panggilan"))
            .ratio(rlm.calls_ratio())
            .label(format!("{} / {}", rlm.calls, rlm.max_calls))
            .style(Style::default().fg(app.palette().accent)),
        areas[1],
    );
}

fn render_budget_footer(frame: &mut Frame, area: Rect, app: &App) {
    let rlm = app.rlm.snapshot();
    let mem = rlm
        .mem_calls()
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let mem = if mem.is_empty() { "-".to_string() } else { mem };
    frame.render_widget(
        Paragraph::new(vec![
            key_value("mem", &mem),
            key_value("final", rlm.final_answer.as_deref().unwrap_or("-")),
        ])
        .block(panel("Status"))
        .style(Style::default().fg(app.palette().fg)),
        area,
    );
}

fn render_tree(frame: &mut Frame, area: Rect, app: &App) {
    let rlm = app.rlm.snapshot();
    let items: Vec<ListItem> = rlm
        .visible_rows()
        .into_iter()
        .map(|(depth, node)| {
            let indent = "  ".repeat(depth);
            let mark = if node.mem { "@" } else { "·" };
            let done = if node.done { "✓" } else { "…" };
            let text = format!("{indent}{mark} {done} {}", truncate(&node.label, 38));
            if node.id == rlm.selected {
                ListItem::new(Line::from(Span::styled(
                    text,
                    Style::default().fg(app.palette().accent),
                )))
            } else {
                ListItem::new(text)
            }
        })
        .collect();
    let mut state = ListState::default().with_selected(Some(rlm.selected));
    frame.render_stateful_widget(
        List::new(items)
            .block(panel("Pohon rekursi"))
            .highlight_style(Style::default().bg(app.palette().selection))
            .highlight_symbol("› "),
        area,
        &mut state,
    );
}

fn render_repl(frame: &mut Frame, area: Rect, app: &App) {
    let rlm = app.rlm.snapshot();
    let mut lines: Vec<Line> = rlm
        .repl_lines()
        .into_iter()
        .map(|line| {
            if line.starts_with("$ ") {
                Line::from(Span::styled(line, Style::default().fg(app.palette().ok)))
            } else {
                Line::from(Span::styled(line, Style::default().fg(app.palette().fg)))
            }
        })
        .collect();
    if rlm.final_answer.is_some() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("FINAL {}", rlm.final_answer.as_deref().unwrap_or("")),
            Style::default().fg(app.palette().warn),
        )));
    }
    if rlm.error.is_some() {
        lines.push(Line::from(Span::styled(
            rlm.error.as_deref().unwrap_or(""),
            Style::default().fg(app.palette().error),
        )));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("REPL & keluaran"))
            .wrap(ratatui::widgets::Wrap { trim: true })
            .style(Style::default().fg(app.palette().fg)),
        area,
    );
}
