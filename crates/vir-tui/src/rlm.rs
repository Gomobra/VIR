//! Jejak RLM: pohon rekursi, meter anggaran, sel REPL, dan panggilan memori.

use crate::json::Json;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};

const MAX_NODES: usize = 256;
const MAX_OUTPUT: usize = 12_000;

#[derive(Debug, Clone, PartialEq)]
pub struct RlmNode {
    pub id: usize,
    pub parent: Option<usize>,
    pub label: String,
    pub depth: usize,
    pub calls: usize,
    pub code: String,
    pub output: String,
    pub mem: bool,
    pub done: bool,
}

impl Default for RlmNode {
    fn default() -> Self {
        Self {
            id: 0,
            parent: None,
            label: "tugas".into(),
            depth: 0,
            calls: 0,
            code: String::new(),
            output: String::new(),
            mem: false,
            done: false,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RlmSnapshot {
    pub nodes: Vec<RlmNode>,
    pub selected: usize,
    pub max_depth: usize,
    pub max_calls: usize,
    pub calls: usize,
    pub depth: usize,
    pub status: String,
    pub final_answer: Option<String>,
    pub error: Option<String>,
}

impl RlmSnapshot {
    pub fn new() -> Self {
        Self {
            nodes: vec![RlmNode::default()],
            status: "idle".into(),
            ..Self::default()
        }
    }

    pub fn selected(&self) -> RlmNode {
        self.nodes
            .get(self.selected)
            .or_else(|| self.nodes.first())
            .cloned()
            .unwrap_or_default()
    }

    pub fn depth_ratio(&self) -> f64 {
        ratio(self.depth, self.max_depth)
    }

    pub fn calls_ratio(&self) -> f64 {
        ratio(self.calls, self.max_calls)
    }

    pub fn visible_rows(&self) -> Vec<(usize, &RlmNode)> {
        let mut rows = Vec::new();
        for node in &self.nodes {
            if node.id == 0 {
                rows.push((0, node));
            }
        }
        for node in self.nodes.iter().filter(|node| node.id != 0) {
            let depth = node.depth.min(12);
            rows.push((depth, node));
        }
        rows
    }

    pub fn repl_lines(&self) -> Vec<String> {
        let node = self.selected();
        let mut lines = Vec::new();
        if !node.code.is_empty() {
            for line in node.code.lines().take(16) {
                lines.push(format!("$ {line}"));
            }
        }
        for line in node.output.lines().take(12) {
            lines.push(truncate(line, 160));
        }
        lines
    }

    pub fn mem_calls(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|node| node.mem)
            .map(|node| node.id)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct RlmReader {
    path: String,
    pos: u64,
    snapshot: RlmSnapshot,
}

impl RlmReader {
    pub fn new(path: impl Into<String>) -> Self {
        let path = path.into();
        Self {
            path,
            pos: 0,
            snapshot: RlmSnapshot::new(),
        }
    }

    pub fn snapshot(&self) -> &RlmSnapshot {
        &self.snapshot
    }

    pub fn snapshot_mut(&mut self) -> &mut RlmSnapshot {
        &mut self.snapshot
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
            self.snapshot = RlmSnapshot::new();
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
                Ok(_) => self.apply(line.trim()),
                Err(_) => break,
            }
        }
        self.pos = reader.stream_position().unwrap_or(self.pos);
    }

    fn apply(&mut self, line: &str) {
        let Some(value) = Json::parse(line) else {
            return;
        };
        match value
            .get("event")
            .or_else(|| value.get("type"))
            .and_then(Json::as_str)
        {
            Some("budget") => self.apply_budget(&value),
            Some("final") => self.apply_final(&value),
            Some("error") => self.snapshot.error = Some(message(&value, "RLM gagal")),
            Some("step") | Some("node") | None => self.apply_step(&value),
            Some(_) => {}
        }
    }

    fn apply_budget(&mut self, value: &Json) {
        if let Some(v) = value.get("max_depth").and_then(Json::as_usize) {
            self.snapshot.max_depth = v;
        }
        if let Some(v) = value.get("max_calls").and_then(Json::as_usize) {
            self.snapshot.max_calls = v;
        }
        self.snapshot.status = "running".into();
    }

    fn apply_step(&mut self, value: &Json) {
        let depth = value.get("depth").and_then(Json::as_usize).unwrap_or(0);
        let calls = value.get("calls").and_then(Json::as_usize).unwrap_or(0);
        let id = value
            .get("id")
            .and_then(Json::as_usize)
            .unwrap_or(self.snapshot.nodes.len());
        let parent = value
            .get("parent")
            .and_then(Json::as_usize)
            .or_else(|| find_parent(&self.snapshot.nodes, depth));
        let node = node_from_json(value, id, parent, depth, calls);
        self.push_node(node);
        self.snapshot.depth = self.snapshot.depth.max(depth);
        self.snapshot.calls = self.snapshot.calls.max(calls);
        self.snapshot.status = "running".into();
    }

    fn push_node(&mut self, node: RlmNode) {
        self.snapshot.nodes.push(node);
        self.snapshot.selected = self.snapshot.nodes.len() - 1;
        if self.snapshot.nodes.len() > MAX_NODES {
            self.snapshot.nodes.remove(1);
        }
    }

    fn apply_final(&mut self, value: &Json) {
        self.snapshot.final_answer = Some(message(value, "FINAL"));
        self.snapshot.status = "final".into();
        if let Some(node) = self.snapshot.nodes.last_mut() {
            node.done = true;
        }
    }
}

fn node_from_json(
    value: &Json,
    id: usize,
    parent: Option<usize>,
    depth: usize,
    calls: usize,
) -> RlmNode {
    let label = value
        .get("node")
        .or_else(|| value.get("label"))
        .and_then(Json::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("call {id}"));
    RlmNode {
        id,
        parent,
        label,
        depth,
        calls,
        code: value
            .get("code")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string(),
        output: truncate(
            value.get("output").and_then(Json::as_str).unwrap_or(""),
            MAX_OUTPUT,
        ),
        mem: value
            .get("mem")
            .and_then(Json::as_bool)
            .unwrap_or_else(|| value.get("tool").and_then(Json::as_str) == Some("mem")),
        done: value.get("done").and_then(Json::as_bool) == Some(true),
    }
}

fn find_parent(nodes: &[RlmNode], depth: usize) -> Option<usize> {
    if depth == 0 {
        return None;
    }
    nodes
        .iter()
        .rev()
        .find(|node| node.depth + 1 == depth)
        .map(|node| node.id)
}

fn message(value: &Json, fallback: &str) -> String {
    value
        .get("answer")
        .or_else(|| value.get("message"))
        .and_then(Json::as_str)
        .unwrap_or(fallback)
        .to_string()
}

fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut out = text.chars().take(max.saturating_sub(1)).collect::<String>();
    out.push('…');
    out
}

fn ratio(value: usize, max: usize) -> f64 {
    if max == 0 {
        0.0
    } else {
        (value as f64 / max as f64).clamp(0.0, 1.0)
    }
}
