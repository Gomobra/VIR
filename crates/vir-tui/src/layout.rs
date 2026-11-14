//! Tata letak responsif: <60 satu panel, 60..100 dua panel, >100 tiga panel.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutMode {
    Compact,
    Medium,
    Wide,
}

impl LayoutMode {
    pub fn from_width(width: u16) -> Self {
        match width {
            0..=59 => LayoutMode::Compact,
            60..=100 => LayoutMode::Medium,
            _ => LayoutMode::Wide,
        }
    }

    pub fn columns(self) -> u8 {
        match self {
            LayoutMode::Compact => 1,
            LayoutMode::Medium => 2,
            LayoutMode::Wide => 3,
        }
    }
}

pub fn vertical(area: Rect, constraints: &[Constraint]) -> Rc<[Rect]> {
    Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints.to_vec())
        .split(area)
}

pub fn horizontal(area: Rect, constraints: &[Constraint]) -> Rc<[Rect]> {
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints.to_vec())
        .split(area)
}

pub fn body(area: Rect, mode: LayoutMode, left: Constraint, right: Constraint) -> Rc<[Rect]> {
    match mode {
        LayoutMode::Compact => vertical(area, &[left, right]),
        LayoutMode::Medium | LayoutMode::Wide => horizontal(area, &[left, right]),
    }
}
