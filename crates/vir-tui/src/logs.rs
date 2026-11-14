//! Pembaca log `tracing`: JSONL atau teks, dengan filter level/modul.

use crate::json::Json;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};

const MAX_RECORDS: usize = 1_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn label(self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "trace" => Some(LogLevel::Trace),
            "debug" => Some(LogLevel::Debug),
            "info" => Some(LogLevel::Info),
            "warn" | "warning" => Some(LogLevel::Warn),
            "error" => Some(LogLevel::Error),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogRecord {
    pub time: String,
    pub level: LogLevel,
    pub target: String,
    pub message: String,
}

impl LogRecord {
    pub fn line(&self) -> String {
        format!(
            "{} {:<5} {}: {}",
            self.time,
            self.level.label(),
            self.target,
            self.message
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterLevel {
    All,
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl FilterLevel {
    pub fn label(self) -> &'static str {
        match self {
            FilterLevel::All => "ALL",
            FilterLevel::Trace => "TRACE",
            FilterLevel::Debug => "DEBUG",
            FilterLevel::Info => "INFO",
            FilterLevel::Warn => "WARN",
            FilterLevel::Error => "ERROR",
        }
    }

    pub fn next(self) -> Self {
        match self {
            FilterLevel::All => FilterLevel::Trace,
            FilterLevel::Trace => FilterLevel::Debug,
            FilterLevel::Debug => FilterLevel::Info,
            FilterLevel::Info => FilterLevel::Warn,
            FilterLevel::Warn => FilterLevel::Error,
            FilterLevel::Error => FilterLevel::All,
        }
    }

    fn accepts(self, level: LogLevel) -> bool {
        match self {
            FilterLevel::All => true,
            FilterLevel::Trace => level >= LogLevel::Trace,
            FilterLevel::Debug => level >= LogLevel::Debug,
            FilterLevel::Info => level >= LogLevel::Info,
            FilterLevel::Warn => level >= LogLevel::Warn,
            FilterLevel::Error => level >= LogLevel::Error,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogFilter {
    pub level: FilterLevel,
    pub module: String,
    pub query: String,
}

impl Default for LogFilter {
    fn default() -> Self {
        Self {
            level: FilterLevel::All,
            module: String::new(),
            query: String::new(),
        }
    }
}

impl LogFilter {
    pub fn accepts(&self, record: &LogRecord) -> bool {
        if !self.level.accepts(record.level) {
            return false;
        }
        let module_ok = self.module.is_empty()
            || record
                .target
                .to_ascii_lowercase()
                .contains(&self.module.to_ascii_lowercase());
        let query_ok = self.query.is_empty()
            || record
                .message
                .to_ascii_lowercase()
                .contains(&self.query.to_ascii_lowercase());
        module_ok && query_ok
    }

    pub fn summary(&self) -> String {
        format!(
            "level={} module={} q={}",
            self.level.label(),
            if self.module.is_empty() {
                "-"
            } else {
                &self.module
            },
            if self.query.is_empty() {
                "-"
            } else {
                &self.query
            }
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct LogStore {
    path: String,
    pos: u64,
    records: VecDeque<LogRecord>,
    pub error: Option<String>,
}

impl LogStore {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            pos: 0,
            records: VecDeque::new(),
            error: None,
        }
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
            self.records.clear();
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
                    if let Some(record) = parse_line(&line) {
                        self.records.push_back(record);
                    }
                }
                Err(_) => break,
            }
        }
        while self.records.len() > MAX_RECORDS {
            self.records.pop_front();
        }
        self.pos = reader.stream_position().unwrap_or(self.pos);
    }

    pub fn filtered(&self, filter: &LogFilter) -> Vec<&LogRecord> {
        self.records
            .iter()
            .filter(|record| filter.accepts(record))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

pub fn parse_line(line: &str) -> Option<LogRecord> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('{') {
        return parse_json(trimmed);
    }
    Some(parse_plain(trimmed))
}

fn parse_json(line: &str) -> Option<LogRecord> {
    let value = Json::parse(line)?;
    Some(LogRecord {
        time: value
            .get("time")
            .or_else(|| value.get("timestamp"))
            .or_else(|| value.get("ts"))
            .and_then(Json::as_str)
            .unwrap_or("-")
            .to_string(),
        level: value
            .get("level")
            .or_else(|| value.get("severity"))
            .and_then(Json::as_str)
            .and_then(LogLevel::parse)
            .unwrap_or(LogLevel::Info),
        target: value
            .get("target")
            .or_else(|| value.get("module"))
            .and_then(Json::as_str)
            .unwrap_or("log")
            .to_string(),
        message: value
            .get("message")
            .or_else(|| value.get("msg"))
            .and_then(Json::as_str)
            .unwrap_or(line)
            .to_string(),
    })
}

fn parse_plain(line: &str) -> LogRecord {
    let (level, rest) = split_level(line);
    let (target, message) = rest.split_once(": ").unwrap_or(("log", rest));
    LogRecord {
        time: "-".into(),
        level,
        target: target.trim().to_string(),
        message: message.trim().to_string(),
    }
}

fn split_level(line: &str) -> (LogLevel, &str) {
    for level in [
        LogLevel::Error,
        LogLevel::Warn,
        LogLevel::Info,
        LogLevel::Debug,
        LogLevel::Trace,
    ] {
        let token = level.label();
        if let Some(pos) = line.find(token) {
            let rest = line[pos + token.len()..].trim();
            return (level, rest);
        }
    }
    (LogLevel::Info, line)
}
