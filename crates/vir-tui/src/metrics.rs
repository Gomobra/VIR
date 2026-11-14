//! Pembaca metrik latihan: JSONL loss/LR, level, codebook, gerbang, checkpoint.

use crate::json::Json;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAX_POINTS: usize = 240;
const MAX_CHECKPOINTS: usize = 8;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GateStatus {
    pub name: String,
    pub state: String,
    pub detail: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LevelMetric {
    pub name: String,
    pub loss: f64,
    pub codebook: Vec<f64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Checkpoint {
    pub name: String,
    pub step: Option<usize>,
    pub loss: Option<f64>,
    pub path: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricsSnapshot {
    pub losses: Vec<f64>,
    pub lrs: Vec<f64>,
    pub throughput: Vec<f64>,
    pub memory_mb: Vec<f64>,
    pub steps: usize,
    pub backend: String,
    pub eta_secs: Option<f64>,
    pub running: bool,
    pub error: Option<String>,
    pub gates: Vec<GateStatus>,
    pub levels: Vec<LevelMetric>,
    pub checkpoints: Vec<Checkpoint>,
}

impl MetricsSnapshot {
    pub fn latest_loss(&self) -> Option<f64> {
        self.losses.last().copied()
    }

    pub fn latest_lr(&self) -> Option<f64> {
        self.lrs.last().copied()
    }

    pub fn latest_throughput(&self) -> Option<f64> {
        self.throughput.last().copied()
    }

    pub fn latest_memory(&self) -> Option<f64> {
        self.memory_mb.last().copied()
    }

    pub fn summary(&self) -> String {
        format!(
            "step {} · loss {} · lr {} · {} tok/s · {}",
            self.steps,
            opt_fmt(self.latest_loss(), 4),
            opt_fmt(self.latest_lr(), 6),
            opt_fmt(self.latest_throughput(), 0),
            self.eta_text()
        )
    }

    pub fn eta_text(&self) -> String {
        self.eta_secs
            .filter(|s| s.is_finite() && *s > 0.0)
            .map(|s| format!("ETA {s:.0}s"))
            .unwrap_or_else(|| "ETA -".into())
    }
}

#[derive(Debug)]
pub struct MetricsReader {
    path: String,
    pos: u64,
    snapshot: MetricsSnapshot,
}

impl MetricsReader {
    pub fn new(path: impl Into<String>) -> Self {
        let path = path.into();
        Self {
            snapshot: MetricsSnapshot {
                backend: "cpu".into(),
                running: true,
                ..MetricsSnapshot::default()
            },
            pos: 0,
            path,
        }
    }

    pub fn snapshot(&self) -> &MetricsSnapshot {
        &self.snapshot
    }

    pub fn poll(&mut self) {
        self.read_lines();
        self.scan_checkpoints();
    }

    fn read_lines(&mut self) {
        let Ok(file) = File::open(&self.path) else {
            return;
        };
        let Ok(len) = file.metadata().map(|m| m.len()) else {
            return;
        };
        if len < self.pos {
            self.pos = 0;
            self.snapshot.losses.clear();
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
                Ok(_) => self.apply_line(&line),
                Err(_) => break,
            }
        }
        self.pos = reader.stream_position().unwrap_or(self.pos);
    }

    fn apply_line(&mut self, line: &str) {
        let Some(value) = Json::parse(line.trim()) else {
            return;
        };
        self.apply_json(&value);
    }

    fn apply_json(&mut self, value: &Json) {
        if let Some(name) = value.get("event").and_then(Json::as_str) {
            if name == "checkpoint" {
                self.push_checkpoint(value);
                return;
            }
        }
        if let Some(loss) = value.get("loss").and_then(Json::as_f64) {
            push_bounded(&mut self.snapshot.losses, loss, MAX_POINTS);
        }
        if let Some(lr) = value.get("lr").and_then(Json::as_f64) {
            push_bounded(&mut self.snapshot.lrs, lr, MAX_POINTS);
        }
        if let Some(tp) = value.get("throughput").and_then(Json::as_f64) {
            push_bounded(&mut self.snapshot.throughput, tp, MAX_POINTS);
        }
        if let Some(mem) = value.get("mem_mb").and_then(Json::as_f64) {
            push_bounded(&mut self.snapshot.memory_mb, mem, MAX_POINTS);
        }
        if let Some(step) = value.get("step").and_then(Json::as_usize) {
            self.snapshot.steps = step;
        }
        if let Some(eta) = value.get("eta_secs").and_then(Json::as_f64) {
            self.snapshot.eta_secs = Some(eta);
        }
        if let Some(backend) = value.get("backend").and_then(Json::as_str) {
            self.snapshot.backend = backend.to_string();
        }
        self.apply_levels(value);
        self.apply_gate(value);
        if value.get("done").and_then(Json::as_bool) == Some(true)
            || value.get("status").and_then(Json::as_str) == Some("done")
        {
            self.snapshot.running = false;
        }
    }

    fn apply_levels(&mut self, value: &Json) {
        if let Some(items) = value.get("levels").and_then(Json::as_array) {
            let levels = items
                .iter()
                .enumerate()
                .map(|(i, item)| parse_level(item, i))
                .collect();
            self.snapshot.levels = levels;
        }
        let Some(codebook) = value.get("codebook_util").and_then(Json::as_array) else {
            return;
        };
        for (i, item) in codebook.iter().enumerate() {
            let level = self.snapshot.levels.get_mut(i);
            let values = parse_f64_array(item);
            if let Some(level) = level {
                level.codebook = values;
            } else {
                self.snapshot.levels.push(LevelMetric {
                    name: format!("L{i}"),
                    loss: 0.0,
                    codebook: values,
                });
            }
        }
    }

    fn apply_gate(&mut self, value: &Json) {
        if let Some(gates) = value.get("gates").and_then(Json::as_array) {
            self.snapshot.gates = gates.iter().filter_map(parse_gate).collect();
            return;
        }
        if let Some(gate) = value.get("gate") {
            if let Some(gate) = parse_gate(gate) {
                self.snapshot.gates = vec![gate];
            }
        }
    }

    fn push_checkpoint(&mut self, value: &Json) {
        let name = value
            .get("name")
            .and_then(Json::as_str)
            .unwrap_or("checkpoint")
            .to_string();
        let path = value
            .get("path")
            .and_then(Json::as_str)
            .unwrap_or(&name)
            .to_string();
        self.snapshot.checkpoints.push(Checkpoint {
            name,
            step: value.get("step").and_then(Json::as_usize),
            loss: value.get("loss").and_then(Json::as_f64),
            path,
        });
        sort_checkpoints(&mut self.snapshot.checkpoints);
    }

    fn scan_checkpoints(&mut self) {
        let Some(dir) = checkpoint_dir(&self.path) else {
            return;
        };
        let Ok(entries) = fs::read_dir(&dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if !looks_like_checkpoint(name) {
                continue;
            }
            let checkpoint = Checkpoint {
                name: name.to_string(),
                step: parse_step(name),
                loss: parse_loss_name(name),
                path: path.display().to_string(),
            };
            if !self
                .snapshot
                .checkpoints
                .iter()
                .any(|c| c.path == checkpoint.path)
            {
                self.snapshot.checkpoints.push(checkpoint);
            }
        }
        sort_checkpoints(&mut self.snapshot.checkpoints);
    }
}

fn parse_level(value: &Json, index: usize) -> LevelMetric {
    let codebook = value
        .get("codebook_util")
        .map(parse_f64_array)
        .unwrap_or_default();
    let name = value
        .get("name")
        .and_then(Json::as_str)
        .unwrap_or("")
        .to_string();
    LevelMetric {
        name: if name.is_empty() {
            format!("L{index}")
        } else {
            name
        },
        loss: value.get("loss").and_then(Json::as_f64).unwrap_or(0.0),
        codebook,
    }
}

fn parse_f64_array(value: &Json) -> Vec<f64> {
    match value {
        Json::Array(items) => items.iter().filter_map(Json::as_f64).collect(),
        Json::Object(fields) => fields.iter().filter_map(|(_, v)| v.as_f64()).collect(),
        _ => Vec::new(),
    }
}

fn parse_gate(value: &Json) -> Option<GateStatus> {
    if let Some(state) = value.as_str() {
        return Some(GateStatus {
            name: "gate".into(),
            state: state.into(),
            detail: String::new(),
        });
    }
    Some(GateStatus {
        name: value
            .get("name")
            .and_then(Json::as_str)
            .unwrap_or("gate")
            .into(),
        state: value
            .get("state")
            .or_else(|| value.get("status"))
            .and_then(Json::as_str)
            .unwrap_or("unknown")
            .into(),
        detail: value
            .get("detail")
            .and_then(Json::as_str)
            .unwrap_or("")
            .into(),
    })
}

fn push_bounded(values: &mut Vec<f64>, value: f64, max: usize) {
    if !value.is_finite() {
        return;
    }
    values.push(value);
    if values.len() > max {
        values.remove(0);
    }
}

fn sort_checkpoints(checkpoints: &mut Vec<Checkpoint>) {
    checkpoints.sort_by_key(|c| std::cmp::Reverse(c.step.unwrap_or(0)));
    checkpoints.truncate(MAX_CHECKPOINTS);
}

fn checkpoint_dir(path: &str) -> Option<PathBuf> {
    let path = Path::new(path);
    if path.is_dir() {
        return Some(path.to_path_buf());
    }
    path.parent().map(Path::to_path_buf)
}

fn looks_like_checkpoint(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("checkpoint")
        || lower.contains("ckpt")
        || lower.ends_with(".safetensors")
        || lower.ends_with(".pt")
        || lower.ends_with(".bin")
}

pub fn parse_step(name: &str) -> Option<usize> {
    name.split(|c: char| !c.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse().ok())
        .max()
}

pub fn parse_loss_name(name: &str) -> Option<f64> {
    let key = "loss=";
    let start = name.find(key)? + key.len();
    let rest = &name[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn opt_fmt(value: Option<f64>, digits: usize) -> String {
    value
        .map(|v| format!("{v:.digits$}"))
        .unwrap_or_else(|| "-".into())
}
