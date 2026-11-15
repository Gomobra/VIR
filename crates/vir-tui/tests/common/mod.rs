//! Alat bantu bersama tes integrasi `vir-tui` (PRD F3.8.h, R9).
#![allow(dead_code)]

mod data;

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use vir_tui::app::{App, AppPaths};
use vir_tui::screens;

/// Tiga ukuran snapshot wajib menurut F3.8.h.
pub const SIZES: [(u16, u16); 3] = [(40, 20), (80, 24), (120, 40)];

/// Seluruh layar yang harus dirender dan diuji.
pub const SCREENS: [vir_tui::Screen; 8] = [
    vir_tui::Screen::Dashboard,
    vir_tui::Screen::Train,
    vir_tui::Screen::Generate,
    vir_tui::Screen::Memory,
    vir_tui::Screen::Rlm,
    vir_tui::Screen::Eval,
    vir_tui::Screen::Logs,
    vir_tui::Screen::Settings,
];

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Direktori sementara unik per proses; terhapus saat di-drop.
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |delta| delta.as_nanos());
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("vir-tui-{}-{nanos}-{id}", std::process::id()));
        fs::create_dir_all(&path).expect("buat direktori sementara");
        Self { path }
    }

    pub fn write(&self, name: &str, content: &str) {
        let path = self.path.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("buat direktori induk");
        }
        fs::write(path, content).expect("tulis berkas sementara");
    }

    pub fn file(&self, name: &str) -> String {
        self.path.join(name).display().to_string()
    }

    pub fn exists(&self, name: &str) -> bool {
        self.path.join(name).exists()
    }
}

impl Default for TempDir {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// `App` dengan path tepercaya: tidak pernah menyentuh repo kerja.
pub fn paths(temp: &TempDir) -> AppPaths {
    AppPaths {
        metrics: temp.file("metrics.jsonl"),
        logs: temp.file("logs/vir.log"),
        eval: temp.file("eval.jsonl"),
        rlm: temp.file("rlm-events.jsonl"),
        generate: temp.file("generate-events.jsonl"),
        memory: temp.file("memori.db"),
        settings: temp.file("tui.toml"),
    }
}

/// Aplikasi hidup dengan direktori sementara miliknya.
pub struct Fixture {
    pub temp: TempDir,
    pub app: App,
}

impl Fixture {
    /// Tanpa berkas data: jalur "belum ada data" di semua layar.
    pub fn empty() -> Self {
        let temp = TempDir::new();
        let app = App::with_paths(paths(&temp));
        Self { temp, app }
    }

    /// Dengan JSONL contoh agar setiap layar punya isi nyata untuk snapshot.
    pub fn with_data() -> Self {
        let temp = TempDir::new();
        temp.write("logs/vir.log", data::LOG_LINES);
        temp.write("metrics.jsonl", data::METRICS_LINES);
        temp.write("eval.jsonl", data::EVAL_LINES);
        temp.write("rlm-events.jsonl", data::RLM_LINES);
        temp.write("generate-events.jsonl", data::GENERATE_LINES);
        let app = App::with_paths(paths(&temp));
        Self { temp, app }
    }

    pub fn on(&mut self, screen: vir_tui::Screen) -> &mut App {
        self.app.screen = screen;
        &mut self.app
    }
}

impl Default for Fixture {
    fn default() -> Self {
        Self::empty()
    }
}

/// Render satu frame pada ukuran `width`×`height`, lalu pulangkan teks buffer.
pub fn draw(app: &App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal uji");
    terminal
        .draw(|frame| screens::render(frame, app))
        .expect("render satu frame");
    buffer_text(terminal.backend().buffer())
}

/// Teks seluruh sel buffer, satu baris per baris terminal.
pub fn buffer_text(buffer: &Buffer) -> String {
    let area = buffer.area();
    let mut out = String::new();
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let symbol = buffer.cell((x, y)).map_or(" ", |cell| cell.symbol());
            out.push_str(symbol);
        }
        out.push('\n');
    }
    out
}

/// Jumlah sel yang benar-benar berisi glif (bukan spasi).
pub fn visible_cells(text: &str) -> usize {
    text.chars().filter(|c| !c.is_whitespace()).count()
}

/// Nomor baris pertama yang memuat `needle`.
pub fn row_of(text: &str, needle: &str) -> Option<usize> {
    text.lines().position(|line| line.contains(needle))
}
