//! Dasbor: status run, backend, gerbang, metrik terbaru.

use crate::app::App;
use crate::layout::{LayoutMode, vertical};
use crate::widgets::{key_value, panel, sparkline};
use ratatui::prelude::*;
use ratatui::widgets::Paragraph;
use vir_common::build_info;

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let mode = LayoutMode::from_width(area.width);
    let chunks = vertical(
        area,
        &[
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(4),
            Constraint::Min(4),
        ],
    );
    let metrics = app.metrics.snapshot();
    let status = format!(
        "{} · {} · {}",
        build_info(),
        if app.paused {
            "JEDA"
        } else if metrics.running {
            "BERJALAN"
        } else {
            "HENTI"
        },
        metrics.summary()
    );
    frame.render_widget(Paragraph::new(status).block(panel("Status run")), chunks[0]);
    frame.render_widget(
        sparkline(&metrics.losses, app.palette().loss).block(panel("Loss")),
        chunks[1],
    );
    frame.render_widget(
        sparkline(&metrics.lrs, app.palette().lr).block(panel("LR")),
        chunks[2],
    );
    render_details(frame, chunks[3], app, mode);
}

fn render_details(frame: &mut Frame, area: Rect, app: &App, mode: LayoutMode) {
    let chunks = if mode == LayoutMode::Compact {
        vertical(
            area,
            &[Constraint::Percentage(50), Constraint::Percentage(50)],
        )
    } else {
        crate::layout::horizontal(
            area,
            &[Constraint::Percentage(50), Constraint::Percentage(50)],
        )
    };
    frame.render_widget(
        Paragraph::new(summary_lines(app))
            .block(panel("Ringkasan"))
            .style(Style::default().fg(app.palette().fg)),
        chunks[0],
    );
    frame.render_widget(
        Paragraph::new(activity_text(app))
            .block(panel("Aktivitas"))
            .style(Style::default().fg(app.palette().fg)),
        chunks[1],
    );
}

fn summary_lines(app: &App) -> Vec<Line<'static>> {
    let metrics = app.metrics.snapshot();
    let gate = metrics
        .gates
        .first()
        .map(|g| format!("{} {} · {}", g.name, g.state, g.detail))
        .unwrap_or_else(|| "gerbang: -".into());
    vec![
        key_value("gerbang", &gate),
        key_value("checkpoint", &format!("{}", metrics.checkpoints.len())),
        key_value("memori", &app.memory.status),
        key_value("eval", &format!("{} baris", app.eval.snapshot().rows.len())),
    ]
}

fn activity_text(app: &App) -> String {
    format!(
        "log {} baris · filter {}\nRLM {} · langkah {}\nGenerate level {}\n{}",
        app.logs.len(),
        app.log_filter.summary(),
        app.rlm.snapshot().status,
        app.rlm.snapshot().nodes.len(),
        app.generate.view().level,
        app.status
    )
}
