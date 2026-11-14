//! Overlay palet perintah dan bantuan.

use crate::app::App;
use crate::keymap;
use crate::widgets::panel;
use ratatui::prelude::*;
use ratatui::widgets::{Clear, List, ListItem, ListState, Paragraph};

fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width.saturating_sub(2)).max(20);
    let height = height.min(area.height.saturating_sub(2)).max(6);
    Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    }
}

pub fn render_palette(frame: &mut Frame, area: Rect, app: &App) {
    let rect = centered(area, 58, 18);
    frame.render_widget(Clear, rect);
    let items: Vec<ListItem> = app
        .palette
        .filtered()
        .iter()
        .map(|item| ListItem::new(item.label.clone()))
        .collect();
    let mut state = ListState::default().with_selected(Some(app.palette.selected));
    frame.render_stateful_widget(
        List::new(items)
            .block(panel(format!("Palet › {}", app.palette.query)))
            .highlight_style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("› "),
        rect,
        &mut state,
    );
}

pub fn render_help(frame: &mut Frame, area: Rect, app: &App) {
    let rect = centered(area, 54, 17);
    frame.render_widget(Clear, rect);
    let lines: Vec<Line> = keymap::HELP
        .iter()
        .map(|hint| {
            Line::from(vec![
                Span::styled(
                    format!("{:<14}", hint.keys),
                    Style::default().fg(app.palette().accent),
                ),
                Span::raw(hint.desc),
            ])
        })
        .collect();
    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("Bantuan"))
            .style(Style::default().fg(app.palette().fg)),
        rect,
    );
}
