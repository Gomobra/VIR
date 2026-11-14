//! Pembaca evaluasi: baris pass@k, kurva `n_loops`, dan panjang konteks.

use crate::json::Json;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};

const MAX_ROWS: usize = 512;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct EvalRow {
    pub task: String,
    pub n: usize,
    pub c: usize,
    pub pass: Vec<f64>,
    pub n_loops: Option<usize>,
    pub context_len: Option<usize>,
    pub accuracy: Option<f64>,
}

impl EvalRow {
    pub fn pass_at(&self, index: usize) -> f64 {
        self.pass
            .get(index)
            .copied()
            .unwrap_or_else(|| pass_at_k(self.n, self.c, index + 1))
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct EvalSnapshot {
    pub rows: Vec<EvalRow>,
    pub source: String,
    pub error: Option<String>,
}

impl EvalSnapshot {
    pub fn loop_curve(&self) -> Vec<(f64, f64)> {
        curve(&self.rows, |row| {
            row.n_loops
                .map(|v| v as f64)
                .or_else(|| row.accuracy.map(|_| 0.0))
        })
    }

    pub fn context_curve(&self) -> Vec<(f64, f64)> {
        curve(&self.rows, |row| row.context_len.map(|v| v as f64))
    }

    pub fn average_pass(&self, index: usize) -> Option<f64> {
        if self.rows.is_empty() {
            return None;
        }
        let sum: f64 = self.rows.iter().map(|row| row.pass_at(index)).sum();
        Some(sum / self.rows.len() as f64)
    }
}

#[derive(Debug)]
pub struct EvalReader {
    path: String,
    pos: u64,
    snapshot: EvalSnapshot,
}

impl EvalReader {
    pub fn new(path: impl Into<String>) -> Self {
        let path = path.into();
        Self {
            snapshot: EvalSnapshot {
                source: path.clone(),
                ..EvalSnapshot::default()
            },
            pos: 0,
            path,
        }
    }

    pub fn snapshot(&self) -> &EvalSnapshot {
        &self.snapshot
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
            self.snapshot.rows.clear();
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
                Ok(_) => {
                    if let Some(row) = parse_row(line.trim()) {
                        self.snapshot.rows.push(row);
                        if self.snapshot.rows.len() > MAX_ROWS {
                            self.snapshot.rows.remove(0);
                        }
                    }
                }
                Err(_) => break,
            }
        }
        self.pos = reader.stream_position().unwrap_or(self.pos);
    }
}

pub fn parse_row(line: &str) -> Option<EvalRow> {
    let value = Json::parse(line)?;
    let n = value.get("n").and_then(Json::as_usize).unwrap_or(0);
    let c = value.get("c").and_then(Json::as_usize).unwrap_or(0);
    let pass = parse_pass(&value, n, c);
    Some(EvalRow {
        task: value
            .get("task")
            .or_else(|| value.get("name"))
            .and_then(Json::as_str)
            .unwrap_or("eval")
            .to_string(),
        n,
        c,
        pass,
        n_loops: value.get("n_loops").and_then(Json::as_usize),
        context_len: value
            .get("context_len")
            .or_else(|| value.get("context"))
            .and_then(Json::as_usize),
        accuracy: value.get("accuracy").and_then(Json::as_f64),
    })
}

fn parse_pass(value: &Json, n: usize, c: usize) -> Vec<f64> {
    if let Some(values) = value.get("pass_at_k").and_then(Json::as_array) {
        return values.iter().filter_map(Json::as_f64).collect();
    }
    if let Some(map) = value.get("pass_at_k") {
        let mut out = [0.0; 3];
        for (key, item) in match map {
            Json::Object(fields) => fields.iter().map(|(k, v)| (k.as_str(), v)).collect(),
            _ => Vec::new(),
        } {
            if let Some(index) = pass_index(key) {
                out[index] = item.as_f64().unwrap_or(0.0);
            }
        }
        if out.iter().any(|v| *v != 0.0) {
            return out.to_vec();
        }
    }
    [1, 5, 10]
        .iter()
        .map(|k| {
            value
                .get(&format!("pass{k}"))
                .and_then(Json::as_f64)
                .unwrap_or_else(|| pass_at_k(n, c, *k))
        })
        .collect()
}

fn pass_index(key: &str) -> Option<usize> {
    let key = key.trim().to_ascii_lowercase();
    let key = key.strip_prefix("pass@").unwrap_or(&key);
    match key {
        "1" => Some(0),
        "5" => Some(1),
        "10" => Some(2),
        _ => None,
    }
}

fn curve<F>(rows: &[EvalRow], value: F) -> Vec<(f64, f64)>
where
    F: Fn(&EvalRow) -> Option<f64>,
{
    let mut points: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|row| {
            let x = value(row)?;
            let y = row.accuracy.or_else(|| row.pass.first().copied())?;
            Some((x, y))
        })
        .collect();
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    points.dedup_by(|a, b| a.0.total_cmp(&b.0).is_eq());
    points
}

pub fn pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    let c = c.min(n);
    if k == 0 {
        return 1.0;
    }
    if n - c < k {
        return 1.0;
    }
    let mut product = 1.0;
    for i in (n - c + 1)..=n {
        product *= 1.0 - k as f64 / i as f64;
    }
    1.0 - product
}
