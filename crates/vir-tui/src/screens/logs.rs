//! Layar Log: penampil `tracing` dengan filter level dan modul.

use crate::app::App;
use crate::layout::vertical;
use ratatui::prelude::*;
use ratatui::widgets::{List, ListItem, Paragraph};

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = vertical(area, &[Constraint::Length(3), Constraint::Min(5)]);
    let filter = if app.filter_mode {
        format!("Saring modul: {}_", app.filter_buffer)
    } else {
        format!(
            "{} · L ganti level · / saring modul · c bersihkan · {} baris",
            app.log_filter.summary(),
            app.logs.len()
        )
    };
    frame.render_widget(
        Paragraph::new(filter)
            .block(ratatui::widgets::Block::bordered().title("Log"))
            .style(Style::default().fg(app.palette().fg)),
        chunks[0],
    );
    let records = app.logs.filtered(&app.log_filter);
    let items: Vec<ListItem> = records
        .iter()
        .rev()
        .take(180)
        .map(|record| {
            ListItem::new(Line::from(Span::styled(
                record.line(),
                Style::default().fg(app.log_level_color(record.level)),
            )))
        })
        .collect();
    frame.render_widget(
        List::new(items).block(ratatui::widgets::Block::bordered().title("Jejak")),
        chunks[1],
    );
}
