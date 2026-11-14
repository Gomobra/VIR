//! Palet perintah dengan pencocokan suburutan (tanpa dependensi fuzzy).

use crate::keymap::{self, command_label};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandItem {
    pub id: String,
    pub label: String,
}

impl CommandItem {
    fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            label: command_label(id),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PaletteState {
    pub open: bool,
    pub query: String,
    pub selected: usize,
}

impl PaletteState {
    pub fn toggle(&mut self) {
        self.open = !self.open;
        if self.open {
            self.query.clear();
            self.selected = 0;
        }
    }

    pub fn close(&mut self) {
        self.open = false;
    }

    pub fn push(&mut self, c: char) {
        self.query.push(c);
        self.selected = 0;
    }

    pub fn pop(&mut self) {
        self.query.pop();
        self.selected = 0;
    }

    pub fn move_selection(&mut self, delta: isize, count: usize) {
        if count == 0 {
            self.selected = 0;
            return;
        }
        let current = self.selected as isize + delta;
        let wrapped = current.rem_euclid(count as isize);
        self.selected = wrapped as usize;
    }

    pub fn filtered(&self) -> Vec<CommandItem> {
        filter_commands(&self.query)
    }

    pub fn current(&self) -> Option<String> {
        self.filtered()
            .get(self.selected)
            .map(|item| item.id.clone())
    }
}

pub fn filter_commands(query: &str) -> Vec<CommandItem> {
    let mut scored: Vec<(i32, CommandItem)> = keymap::COMMANDS
        .iter()
        .filter_map(|id| {
            let item = CommandItem::new(id);
            fuzzy_score(query, &item.label).map(|score| (score, item))
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.label.cmp(&b.1.label)));
    scored.into_iter().take(12).map(|(_, item)| item).collect()
}

/// Skor suburutan: makin rapat dan makin awal, makin tinggi.
pub fn fuzzy_score(query: &str, target: &str) -> Option<i32> {
    if query.is_empty() {
        return Some(0);
    }
    let query: Vec<char> = query.chars().flat_map(char::to_lowercase).collect();
    let target: Vec<char> = target.chars().flat_map(char::to_lowercase).collect();
    let mut ti = 0;
    let mut score = 0;
    let mut streak = 0;
    for qi in query {
        let mut found = false;
        while ti < target.len() {
            let hit = target[ti] == qi;
            let boundary = ti == 0 || target[ti - 1] == ' ' || target[ti - 1] == ':';
            ti += 1;
            if hit {
                found = true;
                streak += 1;
                score += 10 + streak * 2 + usize::from(boundary) * 8;
                if boundary {
                    score += 6;
                }
                break;
            }
            streak = 0;
        }
        if !found {
            return None;
        }
    }
    Some(i32::try_from(score).unwrap_or(i32::MAX))
}
