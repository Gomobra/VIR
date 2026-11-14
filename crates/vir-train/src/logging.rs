//! Log metrik JSONL per langkah (PRD F1.7 langkah 7; dibaca langsung oleh TUI).
//!
//! Satu baris = satu objek JSON. `record` di bawah adalah bentuk yang dibaca
//! `vir-tui` (layar Train) dan artefak CI: `stage`, `step`, `lr`, `loss_total`,
//! `per_level`, `utilization`, `grad_norm`.

use crate::metrics::LevelStat;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use vir_common::VirError;

/// Satu baris log per langkah.
#[derive(Debug, Clone, Serialize)]
pub struct StepRecord {
    /// Tahap: 1 = VQ, 2 = generator.
    pub stage: u32,
    /// Langkah global.
    pub step: u64,
    /// Langkah sejak masuk tahap.
    pub stage_step: u64,
    /// Laju pembelajaran yang dipakai.
    pub lr: f32,
    /// Loss total.
    pub loss_total: f32,
    /// Loss per level (sudah berbobot).
    pub per_level: Vec<f32>,
    /// Token valid per level.
    pub level_tokens: Vec<f32>,
    /// Utilisasi codebook per level (0..1).
    pub utilization: Vec<f32>,
    /// Perplexity codebook per level.
    pub perplexity: Vec<f32>,
    /// Norma gradien global sebelum clipping.
    pub grad_norm: f32,
    /// Laju korupsi sebenarnya pada batch ini (bila korupsi aktif).
    pub corruption_rate: Option<f32>,
    /// Metrik rekonstruksi untuk gerbang tahap 1 (bila dilaporkan).
    pub recon: Option<f32>,
}

impl StepRecord {
    /// Rekam dari statistik level (utilisasi/perplexity).
    pub fn with_stats(mut self, stats: &[LevelStat]) -> Self {
        self.utilization = stats.iter().map(|s| s.utilization).collect();
        self.perplexity = stats.iter().map(|s| s.perplexity).collect();
        self
    }
}

/// Peristiwa penting (transisi tahap, gerbang lulus, ringkasan).
#[derive(Debug, Clone, Serialize)]
pub struct EventRecord {
    /// Nama peristiwa: `stage_start`, `gate_passed`, `stage_end`, `summary`.
    pub event: String,
    /// Tahap saat peristiwa.
    pub stage: u32,
    /// Langkah global.
    pub step: u64,
    /// Keterangan singkat.
    pub message: String,
    /// Nilai metrik terkait (bila ada).
    pub value: Option<f32>,
}

impl EventRecord {
    /// Peristiwa baru.
    pub fn new(event: &str, stage: u32, step: u64, message: &str, value: Option<f32>) -> Self {
        Self {
            event: event.to_string(),
            stage,
            step,
            message: message.to_string(),
            value,
        }
    }
}

/// Penulis JSONL: satu objek per baris, di-flush setiap baris agar TUI bisa
/// mengikuti berkas yang sedang ditulis.
#[derive(Debug)]
pub struct JsonlLogger {
    w: BufWriter<File>,
    path: String,
    lines: u64,
}

impl JsonlLogger {
    /// Buka (atau timpa) berkas log.
    pub fn new(path: &str) -> Result<Self, VirError> {
        let f = File::create(path)?;
        Ok(Self {
            w: BufWriter::new(f),
            path: path.to_string(),
            lines: 0,
        })
    }

    /// Tambah ke berkas log (bila ada) pada direktori checkpoint.
    pub fn in_dir(dir: &str, name: &str) -> Result<Self, VirError> {
        Self::new(&format!("{dir}/{name}"))
    }

    /// Buka berkas log untuk *append* (dipakai saat resume agar riwayat tidak
    /// hilang).
    pub fn append(path: &str) -> Result<Self, VirError> {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            w: BufWriter::new(f),
            path: path.to_string(),
            lines: 0,
        })
    }

    /// Jumlah baris yang sudah ditulis.
    pub fn lines(&self) -> u64 {
        self.lines
    }

    /// Path berkas log.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Tulis satu objek yang bisa diserialisasi.
    pub fn log<T: Serialize>(&mut self, rec: &T) -> Result<(), VirError> {
        let line = serde_json::to_string(rec).map_err(|e| VirError::Internal(e.to_string()))?;
        writeln!(self.w, "{line}")?;
        self.w.flush()?;
        self.lines += 1;
        Ok(())
    }

    /// Tulis satu baris JSON mentah (mis. output `LevelStat::to_json`).
    pub fn log_str(&mut self, line: &str) -> Result<(), VirError> {
        writeln!(self.w, "{line}")?;
        self.w.flush()?;
        self.lines += 1;
        Ok(())
    }

    /// Paksa tulis buffer ke disk.
    pub fn flush(&mut self) -> Result<(), VirError> {
        self.w.flush()?;
        Ok(())
    }
}

/// Uji asap logger: tulis dua baris lalu baca kembali.
pub fn logging_smoke() -> Result<(), VirError> {
    let dir = std::env::temp_dir().join("vir_train_log_smoke");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("steps.jsonl");
    let path = path
        .to_str()
        .ok_or_else(|| VirError::Internal("path tidak valid".into()))?;
    {
        let mut log = JsonlLogger::new(path)?;
        log.log(&EventRecord::new("stage_start", 1, 0, "mulai", None))?;
        log.log(&EventRecord::new("gate_passed", 1, 10, "lulus", Some(0.8)))?;
        if log.lines() != 2 {
            return Err(VirError::Internal("jumlah baris log salah".into()));
        }
    }
    let text = std::fs::read_to_string(path)?;
    let n = text.lines().filter(|l| !l.trim().is_empty()).count();
    if n != 2 {
        return Err(VirError::Internal("isi berkas log salah".into()));
    }
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
