//! Widget kecil bersama: panel, sparkline, kurva, bar, dan heatmap.

use ratatui::prelude::*;
use ratatui::widgets::{Axis, Block, Chart, Dataset, GraphType, Sparkline};

pub fn panel(title: impl Into<String>) -> Block<'static> {
    Block::bordered().title(title.into())
}

pub fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

pub fn sparkline(values: &[f64], color: Color) -> Sparkline<'static> {
    let data: Vec<u64> = values
        .iter()
        .map(|v| (v.abs() * 100.0).clamp(0.0, 100.0) as u64)
        .collect();
    Sparkline::default()
        .data(&data)
        .max(100)
        .style(Style::default().fg(color))
}

pub fn chart<'a>(
    points: &'a [(f64, f64)],
    title: &'a str,
    color: Color,
    x_label: &'a str,
    y_label: &'a str,
) -> Chart<'a> {
    let (xmin, xmax) = bounds(points, 0);
    let (ymin, ymax) = bounds(points, 1);
    let dataset = Dataset::default()
        .name(title)
        .data(points)
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(color));
    Chart::new(vec![dataset])
        .block(panel(title))
        .x_axis(
            Axis::default()
                .title(x_label)
                .bounds([xmin, xmax])
                .labels([format!("{xmin:.0}"), format!("{xmax:.0}")]),
        )
        .y_axis(
            Axis::default()
                .title(y_label)
                .bounds([ymin, ymax])
                .labels([format!("{ymin:.2}"), format!("{ymax:.2}")]),
        )
}

fn bounds(points: &[(f64, f64)], index: usize) -> (f64, f64) {
    let mut values = points.iter().filter_map(|p| {
        let value = if index == 0 { p.0 } else { p.1 };
        value.is_finite().then_some(value)
    });
    let Some(first) = values.next() else {
        return (0.0, 1.0);
    };
    let mut min = first;
    let mut max = first;
    for value in values {
        min = min.min(value);
        max = max.max(value);
    }
    if (max - min).abs() < f64::EPSILON {
        (min - 1.0, max + 1.0)
    } else {
        (min, max)
    }
}

pub fn bar(value: f64, max: f64, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let ratio = if max <= 0.0 {
        0.0
    } else {
        (value / max).clamp(0.0, 1.0)
    };
    let filled = (ratio * width as f64) as usize;
    let mut out = "█".repeat(filled);
    out.push_str(&"·".repeat(width.saturating_sub(filled)));
    out
}

pub fn heat_style(value: f64) -> Style {
    let v = value.clamp(0.0, 1.0);
    let color = if v < 0.25 {
        Color::Rgb(30, 45, 70)
    } else if v < 0.5 {
        Color::Rgb(36, 95, 120)
    } else if v < 0.75 {
        Color::Rgb(72, 160, 150)
    } else {
        Color::Rgb(130, 210, 150)
    };
    Style::default().fg(color)
}

pub fn key_value(key: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{key:<12}"), Style::default().fg(Color::DarkGray)),
        Span::raw(value.to_string()),
    ])
}

pub fn status_style(text: &str) -> Style {
    let lower = text.to_ascii_lowercase();
    if lower.contains("error") || lower.contains("gagal") {
        Style::default().fg(Color::Red)
    } else if lower.contains("warn") || lower.contains("tunggu") {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::Green)
    }
}
