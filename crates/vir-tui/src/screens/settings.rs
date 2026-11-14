//! Layar Setelan: formulir Config, validasi inline, tema, dan peta tombol.

use crate::app::App;
use crate::layout::{LayoutMode, vertical};
use crate::settings::Row;
use crate::widgets::{key_value, panel};
use ratatui::prelude::*;
use ratatui::widgets::{List, ListItem, ListState, Paragraph};

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let mode = LayoutMode::from_width(area.width);
    let chunks = vertical(
        area,
        &[
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ],
    );
    render_header(frame, chunks[0], app);
    let body = settings_body(chunks[1], mode);
    render_form(frame, body[0], app);
    render_preview(frame, body[1], app);
    render_footer(frame, chunks[2], app);
}

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let text = format!(
        "Config · tema {} · {} · {} · s simpan",
        app.settings.config.theme.label(),
        app.settings.config.color_mode.label(),
        app.settings.validation_text()
    );
    frame.render_widget(
        Paragraph::new(text)
            .block(panel("Setelan"))
            .style(Style::default().fg(app.palette().fg)),
        area,
    );
}

fn settings_body(area: Rect, mode: LayoutMode) -> Vec<Rect> {
    if mode == LayoutMode::Compact {
        vertical(
            area,
            &[Constraint::Percentage(60), Constraint::Percentage(40)],
        )
        .to_vec()
    } else {
        crate::layout::horizontal(
            area,
            &[Constraint::Percentage(55), Constraint::Percentage(45)],
        )
        .to_vec()
    }
}

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let text = format!(
        "↑↓ pilih · ←→ ubah · Enter ubah · s simpan · {}",
        app.settings.validation_text()
    );
    frame.render_widget(
        Paragraph::new(text)
            .block(panel("Validasi"))
            .style(if app.settings.error.is_some() {
                Style::default().fg(app.palette().error)
            } else {
                Style::default().fg(app.palette().ok)
            }),
        area,
    );
}

fn render_form(frame: &mut Frame, area: Rect, app: &App) {
    let items: Vec<ListItem> = Row::ALL
        .iter()
        .map(|row| {
            let label = row.label();
            let value = app.settings.value(*row);
            let style = if app.settings.row() == *row {
                Style::default()
                    .fg(app.palette().accent)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(app.palette().fg)
            };
            ListItem::new(Line::from(vec![
                Span::styled(format!("{label:<22}"), style),
                Span::styled(value, style),
            ]))
        })
        .collect();
    let mut state = ListState::default().with_selected(Some(app.settings.selected));
    frame.render_stateful_widget(
        List::new(items).block(panel("Form")).highlight_symbol("› "),
        area,
        &mut state,
    );
}

fn render_preview(frame: &mut Frame, area: Rect, app: &App) {
    let lines = preview_lines(app);
    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("Pratinjau"))
            .style(Style::default().fg(app.palette().fg)),
        area,
    );
}

fn preview_lines(app: &App) -> Vec<Line<'static>> {
    let palette = app.palette();
    vec![
        swatches(palette),
        Line::from(""),
        key_value("theme", app.settings.config.theme.label()),
        key_value("color_mode", app.settings.config.color_mode.label()),
        key_value(
            "colorblind",
            &app.settings.config.colorblind_safe.to_string(),
        ),
        key_value("fps", &app.settings.config.fps.to_string()),
        key_value(
            "low_bandwidth",
            &app.settings.config.low_bandwidth.to_string(),
        ),
        key_value("palette", &app.settings.config.palette_key),
        key_value(
            "target_length",
            &app.settings.model.target_length.to_string(),
        ),
        key_value("num_levels", &app.settings.model.num_levels.to_string()),
        key_value("hidden_dim", &app.settings.model.hidden_dim.to_string()),
        key_value("num_heads", &app.settings.model.num_heads.to_string()),
        key_value(
            "codebook_size",
            &app.settings.model.codebook_size.to_string(),
        ),
        key_value("ema_decay", &format!("{:.3}", app.settings.model.ema_decay)),
    ]
}

fn swatches(palette: crate::theme::Palette) -> Line<'static> {
    Line::from(vec![
        Span::styled("  bg  ", Style::default().bg(palette.bg)),
        Span::styled("  panel  ", Style::default().bg(palette.panel)),
        Span::styled("  fg  ", Style::default().bg(palette.fg).fg(palette.bg)),
        Span::styled("  loss  ", Style::default().bg(palette.loss)),
        Span::styled("  lr  ", Style::default().bg(palette.lr)),
        Span::styled("  ok  ", Style::default().bg(palette.ok)),
        Span::styled("  warn  ", Style::default().bg(palette.warn)),
        Span::styled("  error  ", Style::default().bg(palette.error)),
    ])
}
