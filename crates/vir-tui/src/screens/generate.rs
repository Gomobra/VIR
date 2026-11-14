//! Layar Generate: prompt, parameter, keluaran per level, kandidat MBR.

use crate::app::App;
use crate::layout::{LayoutMode, vertical};
use crate::widgets::{key_value, panel, truncate};
use ratatui::prelude::*;
use ratatui::widgets::{List, ListItem, Paragraph};

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let mode = LayoutMode::from_width(area.width);
    let chunks = vertical(
        area,
        &[
            Constraint::Length(5),
            Constraint::Min(6),
            Constraint::Length(5),
        ],
    );
    render_prompt(frame, chunks[0], app);
    let body = generate_body(chunks[1], mode);
    render_output(frame, body[0], app);
    render_candidates(frame, body[1], app);
    render_footer(frame, chunks[2], app);
}

fn render_prompt(frame: &mut Frame, area: Rect, app: &App) {
    let view = app.generate.view();
    let text = format!(
        "Prompt:\n{}\n\nn_loops={} · CFG={:.1} · kandidat={} · level {} · {}",
        view.prompt, view.n_loops, view.cfg_scale, view.n_candidates, view.level, view.halting
    );
    frame.render_widget(
        Paragraph::new(text)
            .block(panel("Generate"))
            .wrap(ratatui::widgets::Wrap { trim: true })
            .style(Style::default().fg(app.palette().fg)),
        area,
    );
}

fn generate_body(area: Rect, mode: LayoutMode) -> Vec<Rect> {
    if mode == LayoutMode::Compact {
        vertical(
            area,
            &[Constraint::Percentage(65), Constraint::Percentage(35)],
        )
        .to_vec()
    } else {
        crate::layout::horizontal(
            area,
            &[Constraint::Percentage(65), Constraint::Percentage(35)],
        )
        .to_vec()
    }
}

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let view = app.generate.view();
    let text = format!(
        "{} · keyakinan {:?} · {}",
        view.status, view.confidence, app.status
    );
    frame.render_widget(
        Paragraph::new(text)
            .block(panel("Status"))
            .style(Style::default().fg(app.palette().fg)),
        area,
    );
}

fn render_output(frame: &mut Frame, area: Rect, app: &App) {
    let view = app.generate.view();
    let mut lines = vec![
        key_value("halting", &view.halting),
        key_value("level", &view.level.to_string()),
        Line::from(""),
    ];
    for line in view.output.lines().take(16) {
        lines.push(Line::from(Span::styled(
            truncate(line, 120),
            Style::default().fg(app.palette().fg),
        )));
    }
    if !view.confidence.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(
            view.confidence
                .iter()
                .take(32)
                .map(|v| {
                    Span::styled(
                        "▁",
                        confidence_style(*v, app.palette().ok, app.palette().error),
                    )
                })
                .collect::<Vec<_>>(),
        ));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("Keluaran per level"))
            .wrap(ratatui::widgets::Wrap { trim: true })
            .style(Style::default().fg(app.palette().fg)),
        area,
    );
}

fn render_candidates(frame: &mut Frame, area: Rect, app: &App) {
    let items: Vec<ListItem> = if app.generate.view().candidates.is_empty() {
        vec![ListItem::new("belum ada kandidat MBR")]
    } else {
        app.generate
            .view()
            .candidates
            .iter()
            .map(|candidate| {
                ListItem::new(format!(
                    "{:.3}  {}",
                    candidate.score,
                    truncate(&candidate.text, 36)
                ))
            })
            .collect()
    };
    frame.render_widget(List::new(items).block(panel("Kandidat MBR")), area);
}

fn confidence_style(value: f64, ok: Color, error: Color) -> Style {
    let t = value.clamp(0.0, 1.0);
    Style::default().fg(if t > 0.6 {
        ok
    } else if t < 0.3 {
        error
    } else {
        Color::Yellow
    })
}
