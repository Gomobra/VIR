//! Layar Generate: prompt, parameter, keluaran per level, kandidat MBR.

use crate::json::Json;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Candidate {
    pub text: String,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateView {
    pub prompt: String,
    pub n_loops: usize,
    pub cfg_scale: f32,
    pub n_candidates: usize,
    pub output: String,
    pub level: usize,
    pub halting: String,
    pub confidence: Vec<f64>,
    pub candidates: Vec<Candidate>,
    pub status: String,
}

impl Default for GenerateView {
    fn default() -> Self {
        Self {
            prompt: "Tulis penjelasan singkat tentang memori hierarkis.".into(),
            n_loops: 8,
            cfg_scale: 1.0,
            n_candidates: 4,
            output: "Belum ada keluaran level; menunggu generate-events.jsonl.".into(),
            level: 0,
            halting: "idle".into(),
            confidence: Vec::new(),
            candidates: Vec::new(),
            status: "siap".into(),
        }
    }
}

impl GenerateView {
    pub fn push(&mut self, c: char) {
        self.prompt.push(c);
    }

    pub fn pop(&mut self) {
        self.prompt.pop();
    }

    pub fn adjust_loops(&mut self, delta: isize) {
        let next = self.n_loops as isize + delta;
        self.n_loops = next.clamp(1, 64) as usize;
    }

    pub fn adjust_candidates(&mut self, delta: isize) {
        let next = self.n_candidates as isize + delta;
        self.n_candidates = next.clamp(1, 16) as usize;
    }

    pub fn apply_line(&mut self, line: &str) {
        let Some(value) = Json::parse(line) else {
            return;
        };
        match value.get("event").and_then(Json::as_str) {
            Some("level") => self.apply_level(&value),
            Some("candidate") => self.apply_candidate(&value),
            Some("final") => {
                self.halting = "final".into();
                self.status = "selesai".into();
            }
            _ => {}
        }
    }

    fn apply_level(&mut self, value: &Json) {
        self.level = value
            .get("level")
            .and_then(Json::as_usize)
            .unwrap_or(self.level + 1);
        if let Some(text) = value.get("text").and_then(Json::as_str) {
            self.output = text.to_string();
        }
        self.confidence = value
            .get("confidence")
            .and_then(Json::as_array)
            .map(|items| items.iter().filter_map(Json::as_f64).collect())
            .unwrap_or_default();
        self.halting = value
            .get("halting")
            .and_then(Json::as_str)
            .unwrap_or("running")
            .to_string();
        self.status = format!("level {}", self.level);
    }

    fn apply_candidate(&mut self, value: &Json) {
        self.candidates.push(Candidate {
            text: value
                .get("text")
                .and_then(Json::as_str)
                .unwrap_or("")
                .to_string(),
            score: value.get("score").and_then(Json::as_f64).unwrap_or(0.0),
        });
        self.candidates.truncate(8);
    }
}

#[derive(Debug)]
pub struct GenerateReader {
    path: String,
    pos: u64,
    view: GenerateView,
}

impl GenerateReader {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            pos: 0,
            view: GenerateView::default(),
        }
    }

    pub fn view(&self) -> &GenerateView {
        &self.view
    }

    pub fn view_mut(&mut self) -> &mut GenerateView {
        &mut self.view
    }

    pub fn poll(&mut self) {
        let Ok(file) = File::open(&self.path) else {
            return;
        };
        let Ok(len) = file.metadata().map(|m| m.len()) else {
            return;
        };
        if len < self.pos {
            self.pos = 0;
        }
        let mut reader = BufReader::new(file);
        if reader.seek(SeekFrom::Start(self.pos)).is_err() {
            return;
        }
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => self.view.apply_line(line.trim()),
                Err(_) => break,
            }
        }
        self.pos = reader.stream_position().unwrap_or(self.pos);
    }
}
