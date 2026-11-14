//! Layar Eval: tabel pass@k dan kurva `n_loops` / panjang konteks.

use crate::app::App;
use crate::layout::{LayoutMode, vertical};
use crate::widgets::{chart, panel};
use ratatui::prelude::*;
use ratatui::widgets::{Cell, Paragraph, Row, Table};

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let mode = LayoutMode::from_width(area.width);
    let chunks = vertical(area, &[Constraint::Length(3), Constraint::Min(5)]);
    let snapshot = app.eval.snapshot();
    let summary = format!(
        "{} baris · pass@1 {} · pass@5 {} · pass@10 {} · sumber {}",
        snapshot.rows.len(),
        pct(snapshot.average_pass(0)),
        pct(snapshot.average_pass(1)),
        pct(snapshot.average_pass(2)),
        app.paths.eval
    );
    frame.render_widget(
        Paragraph::new(summary)
            .block(panel("Eval"))
            .style(Style::default().fg(app.palette().fg)),
        chunks[0],
    );
    let body = if mode == LayoutMode::Compact {
        vertical(
            chunks[1],
            &[Constraint::Percentage(48), Constraint::Percentage(52)],
        )
    } else {
        crate::layout::horizontal(
            chunks[1],
            &[Constraint::Percentage(48), Constraint::Percentage(52)],
        )
    };
    render_table(frame, body[0], app);
    render_curves(frame, body[1], app);
}

fn render_table(frame: &mut Frame, area: Rect, app: &App) {
    let rows: Vec<Row> = app
        .eval
        .snapshot()
        .rows
        .iter()
        .map(|row| {
            Row::new(vec![
                Cell::from(row.task.clone()),
                Cell::from(row.n.to_string()),
                Cell::from(row.c.to_string()),
                Cell::from(pct(Some(row.pass_at(0)))),
                Cell::from(pct(Some(row.pass_at(1)))),
                Cell::from(pct(Some(row.pass_at(2)))),
            ])
        })
        .collect();
    let table = Table::new(
        rows,
        [
            Constraint::Percentage(36),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(9),
            Constraint::Length(9),
            Constraint::Length(9),
        ],
    )
    .header(
        Row::new(vec!["task", "n", "c", "pass@1", "pass@5", "pass@10"]).style(
            Style::default()
                .fg(app.palette().accent)
                .add_modifier(Modifier::BOLD),
        ),
    )
    .block(panel("pass@k"))
    .style(Style::default().fg(app.palette().fg));
    frame.render_widget(table, area);
}

fn render_curves(frame: &mut Frame, area: Rect, app: &App) {
    let snapshot = app.eval.snapshot();
    let chunks = vertical(
        area,
        &[Constraint::Percentage(50), Constraint::Percentage(50)],
    );
    render_curve(
        frame,
        chunks[0],
        &snapshot.loop_curve(),
        "Akurasi vs n_loops",
        app.palette().ok,
        "n_loops",
        "kurva n_loops belum ada",
    );
    render_curve(
        frame,
        chunks[1],
        &snapshot.context_curve(),
        "Akurasi vs konteks",
        app.palette().lr,
        "context",
        "kurva konteks belum ada",
    );
}

fn render_curve(
    frame: &mut Frame,
    area: Rect,
    points: &[(f64, f64)],
    title: &str,
    color: Color,
    x_label: &str,
    empty: &str,
) {
    if points.is_empty() {
        frame.render_widget(Paragraph::new(empty).block(panel(title)), area);
        return;
    }
    frame.render_widget(chart(points, title, color, x_label, "accuracy"), area);
}

fn pct(value: Option<f64>) -> String {
    value
        .map(|v| format!("{:.1}%", v * 100.0))
        .unwrap_or_else(|| "-".into())
}
