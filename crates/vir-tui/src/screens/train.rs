//! Layar Latih: loss/LR, level, codebook, throughput, gerbang, checkpoint.

use crate::app::App;
use crate::layout::{LayoutMode, vertical};
use crate::widgets::{bar, chart, heat_style, key_value, panel, sparkline, truncate};
use ratatui::prelude::*;
use ratatui::widgets::{LineGauge, List, ListItem, Paragraph};

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let mode = LayoutMode::from_width(area.width);
    let chunks = train_chunks(area, mode);
    render_status(frame, chunks[0], app);
    if mode == LayoutMode::Compact {
        render_loss(frame, chunks[1], app);
        render_lr(frame, chunks[2], app);
        render_levels(frame, chunks[3], app);
        render_checkpoints(frame, chunks[4], app);
    } else {
        render_wide(frame, &chunks, app);
    }
}

fn train_chunks(area: Rect, mode: LayoutMode) -> Vec<Rect> {
    let constraints: Vec<Constraint> = if mode == LayoutMode::Compact {
        vec![
            Constraint::Length(3),
            Constraint::Length(5),
            Constraint::Length(3),
            Constraint::Length(5),
            Constraint::Min(4),
        ]
    } else {
        vec![
            Constraint::Length(3),
            Constraint::Length(6),
            Constraint::Min(10),
        ]
    };
    vertical(area, &constraints).to_vec()
}

fn render_wide(frame: &mut Frame, chunks: &[Rect], app: &App) {
    let charts = crate::layout::horizontal(
        chunks[1],
        &[Constraint::Percentage(65), Constraint::Percentage(35)],
    );
    render_loss(frame, charts[0], app);
    render_lr(frame, charts[1], app);
    let body = crate::layout::horizontal(
        chunks[2],
        &[Constraint::Percentage(55), Constraint::Percentage(45)],
    );
    render_levels(frame, body[0], app);
    render_checkpoints(frame, body[1], app);
}

fn render_status(frame: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.snapshot();
    let state = if app.paused {
        "JEDA"
    } else if metrics.running {
        "BERJALAN"
    } else {
        "HENTI"
    };
    let text = format!(
        "{} · {} · loss {} · lr {} · {} tok/s · mem {} MB · {} · p=jeda c=checkpoint",
        state,
        metrics.summary(),
        fmt(metrics.latest_loss(), 4),
        fmt(metrics.latest_lr(), 6),
        fmt(metrics.latest_throughput(), 0),
        fmt(metrics.latest_memory(), 0),
        metrics.eta_text(),
    );
    frame.render_widget(
        Paragraph::new(text)
            .block(panel("Latih"))
            .style(Style::default().fg(app.palette().fg)),
        area,
    );
}

fn render_loss(frame: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.snapshot();
    let points: Vec<(f64, f64)> = metrics
        .losses
        .iter()
        .enumerate()
        .map(|(i, loss)| (i as f64, *loss))
        .collect();
    if points.is_empty() {
        frame.render_widget(
            Paragraph::new("belum ada loss · metrics.jsonl").block(panel("Loss")),
            area,
        );
        return;
    }
    frame.render_widget(
        chart(&points, "Loss", app.palette().loss, "step", "loss"),
        area,
    );
}

fn render_lr(frame: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.snapshot();
    let chunks = vertical(
        area,
        &[
            Constraint::Percentage(45),
            Constraint::Percentage(35),
            Constraint::Percentage(20),
        ],
    );
    frame.render_widget(
        sparkline(&metrics.lrs, app.palette().lr).block(panel("LR")),
        chunks[0],
    );
    let throughput = metrics.latest_throughput().unwrap_or(0.0);
    let memory = metrics.latest_memory().unwrap_or(0.0);
    frame.render_widget(
        LineGauge::default()
            .block(panel("Throughput"))
            .ratio((throughput / 2_000.0).clamp(0.0, 1.0))
            .label(format!("{throughput:.0} tok/s"))
            .style(Style::default().fg(app.palette().lr)),
        chunks[1],
    );
    frame.render_widget(
        LineGauge::default()
            .block(panel("Memori"))
            .ratio((memory / 8_192.0).clamp(0.0, 1.0))
            .label(format!("{memory:.0} MB"))
            .style(Style::default().fg(app.palette().accent)),
        chunks[2],
    );
}

fn render_levels(frame: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.snapshot();
    let mut chunks = vertical(
        area,
        &[Constraint::Percentage(55), Constraint::Percentage(45)],
    );
    let max_loss = metrics
        .levels
        .iter()
        .map(|level| level.loss)
        .fold(0.0_f64, f64::max);
    let lines: Vec<Line> = metrics
        .levels
        .iter()
        .map(|level| {
            Line::from(vec![
                Span::raw(format!("{:<5}", truncate(&level.name, 5))),
                Span::styled(
                    bar(level.loss, max_loss.max(0.001), 18),
                    Style::default().fg(app.palette().loss),
                ),
                Span::raw(format!(" {:.4}", level.loss)),
            ])
        })
        .collect();
    if lines.is_empty() {
        chunks = vertical(area, &[Constraint::Percentage(100)]);
        frame.render_widget(
            Paragraph::new("level belum tersedia").block(panel("Level & codebook")),
            chunks[0],
        );
        return;
    }
    frame.render_widget(
        Paragraph::new(lines).block(panel("Loss per level")),
        chunks[0],
    );
    render_heatmap(frame, chunks[1], app);
}

fn render_heatmap(frame: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.snapshot();
    let mut lines = Vec::new();
    for level in &metrics.levels {
        let mut spans = vec![Span::raw(format!("{:<5}", truncate(&level.name, 5)))];
        if level.codebook.is_empty() {
            spans.push(Span::styled("—", Style::default().fg(app.palette().muted)));
        } else {
            for value in level.codebook.iter().take(24) {
                spans.push(Span::styled("▁", heat_style(*value)));
            }
        }
        lines.push(Line::from(spans));
    }
    if lines.is_empty() {
        lines.push(Line::from("codebook belum tersedia"));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("Utilisasi codebook"))
            .style(Style::default().fg(app.palette().fg)),
        area,
    );
}

fn render_checkpoints(frame: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.snapshot();
    let chunks = vertical(
        area,
        &[Constraint::Percentage(72), Constraint::Percentage(28)],
    );
    frame.render_widget(
        List::new(checkpoint_items(metrics)).block(panel("Checkpoint")),
        chunks[0],
    );
    frame.render_widget(
        Paragraph::new(gate_lines(metrics))
            .block(panel("Gerbang"))
            .style(Style::default().fg(app.palette().ok)),
        chunks[1],
    );
}

fn checkpoint_items(metrics: &crate::metrics::MetricsSnapshot) -> Vec<ListItem<'_>> {
    if metrics.checkpoints.is_empty() {
        return vec![ListItem::new("belum ada checkpoint")];
    }
    metrics
        .checkpoints
        .iter()
        .map(|checkpoint| {
            let step = checkpoint
                .step
                .map(|s| s.to_string())
                .unwrap_or_else(|| "-".into());
            let loss = checkpoint
                .loss
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".into());
            ListItem::new(format!(
                "{} · step {} · loss {}",
                truncate(&checkpoint.name, 24),
                step,
                loss
            ))
        })
        .collect()
}

fn gate_lines(metrics: &crate::metrics::MetricsSnapshot) -> Vec<Line<'static>> {
    metrics
        .gates
        .iter()
        .map(|gate| key_value(&gate.name, &format!("{} {}", gate.state, gate.detail)))
        .collect()
}

fn fmt(value: Option<f64>, digits: usize) -> String {
    value
        .map(|v| format!("{v:.digits$}"))
        .unwrap_or_else(|| "-".into())
}
