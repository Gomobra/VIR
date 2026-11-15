//! Model–Update–View: status aplikasi, pesan, dan pembaruan murni.

use crate::Screen;
use crate::eval::EvalReader;
use crate::filter::FilterTarget;
use crate::generate::GenerateReader;
use crate::logs::{LogFilter, LogStore};
use crate::memory_view::MemoryView;
use crate::palette::PaletteState;
pub use crate::paths::AppPaths;
use crate::rlm::RlmReader;
use crate::settings::SettingsView;
use crate::theme::Palette;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Toast {
    pub text: String,
    pub ticks: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Msg {
    Tick,
    Key(KeyEvent),
    Command(String),
    Resize(u16, u16),
}

pub struct App {
    pub screen: Screen,
    pub palette: PaletteState,
    pub help: bool,
    pub metrics: crate::metrics::MetricsReader,
    pub logs: LogStore,
    pub log_filter: LogFilter,
    pub eval: EvalReader,
    pub rlm: RlmReader,
    pub generate: GenerateReader,
    pub memory: MemoryView,
    pub settings: SettingsView,
    pub filter_mode: bool,
    pub filter_target: FilterTarget,
    pub filter_buffer: String,
    pub toast: Option<Toast>,
    pub paused: bool,
    pub status: String,
    pub ticks: u64,
    pub quit: bool,
    pub paths: AppPaths,
}

impl App {
    pub fn new() -> Self {
        Self::with_paths(AppPaths::default())
    }

    pub fn with_paths(paths: AppPaths) -> Self {
        let mut app = Self {
            screen: Screen::Dashboard,
            palette: PaletteState::default(),
            help: false,
            metrics: crate::metrics::MetricsReader::new(paths.metrics.clone()),
            logs: LogStore::new(paths.logs.clone()),
            log_filter: LogFilter::default(),
            eval: EvalReader::new(paths.eval.clone()),
            rlm: RlmReader::new(paths.rlm.clone()),
            generate: GenerateReader::new(paths.generate.clone()),
            memory: MemoryView::new(paths.memory.clone()),
            settings: SettingsView::new(paths.settings.clone()),
            filter_mode: false,
            filter_target: FilterTarget::Logs,
            filter_buffer: String::new(),
            toast: None,
            paused: false,
            status: "siap".into(),
            ticks: 0,
            quit: false,
            paths,
        };
        app.refresh_data();
        app
    }

    pub fn refresh_data(&mut self) {
        self.metrics.poll();
        self.logs.poll();
        self.eval.poll();
        self.rlm.poll();
        self.generate.poll();
        if self.ticks % 15 == 0 {
            self.memory.refresh();
        }
    }

    pub fn update(&mut self, msg: Msg) {
        match msg {
            Msg::Tick => self.tick(),
            Msg::Key(key) => self.on_key(key),
            Msg::Command(id) => self.execute(&id),
            Msg::Resize(_, _) => self.status = "ukuran terminal berubah".into(),
        }
    }

    fn tick(&mut self) {
        self.ticks = self.ticks.wrapping_add(1);
        if self.toast.as_ref().is_some_and(|toast| toast.ticks <= 1) {
            self.toast = None;
        } else if let Some(toast) = self.toast.as_mut() {
            toast.ticks -= 1;
        }
        if self.ticks % 3 == 0 {
            self.refresh_data();
        }
    }

    pub fn palette(&self) -> Palette {
        Palette::new(
            self.settings.config.theme,
            crate::theme::resolve_mode(self.settings.config.color_mode),
            self.settings.config.colorblind_safe,
        )
    }

    pub fn notify(&mut self, text: impl Into<String>) {
        self.toast = Some(Toast {
            text: text.into(),
            ticks: 30,
        });
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        // Ctrl+P selalu mengalihkan palet, termasuk saat overlay lain terbuka
        // (F3.8: palet perintah lewat `Ctrl+P` atau `:`).
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('p') {
            self.help = false;
            self.palette.toggle();
            return;
        }
        if self.palette.open {
            self.on_palette_key(key);
            return;
        }
        if self.help {
            self.help = false;
            return;
        }
        if self.filter_mode {
            self.on_filter_key(key);
            return;
        }
        match key.code {
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Char('?') => self.help = true,
            KeyCode::Char(':') => self.palette.toggle(),
            KeyCode::Char(c) if c.is_ascii_digit() => {
                if let Some(index) = c.to_digit(10).and_then(|n| usize::try_from(n).ok()) {
                    if (1..=8).contains(&index) {
                        self.screen = Screen::from_index(index - 1);
                    }
                }
            }
            KeyCode::Left => self.screen = Screen::from_index((self.screen.index() + 7) % 8),
            KeyCode::Right => self.screen = Screen::from_index((self.screen.index() + 1) % 8),
            KeyCode::Up => self.on_up(),
            KeyCode::Down => self.on_down(),
            KeyCode::Enter => self.on_enter(),
            KeyCode::Esc => self.escape(),
            _ => self.on_char(key.code),
        }
    }

    fn on_char(&mut self, code: KeyCode) {
        let KeyCode::Char(c) = code else {
            return;
        };
        match (self.screen, c) {
            (Screen::Train, 'p') => {
                self.paused = !self.paused;
                self.status = if self.paused {
                    "latih dijeda"
                } else {
                    "latih lanjut"
                }
                .into();
            }
            (Screen::Train, 'c') => self.status = "checkpoint manual diminta".into(),
            (Screen::Memory, 'r') => self.memory.refresh(),
            (Screen::Memory, '/') => self.open_filter(FilterTarget::Memory),
            (Screen::Rlm, 'n') => self.status = "langkah RLM diminta".into(),
            (Screen::Rlm, 'x') => self.status = "RLM dibatalkan".into(),
            (Screen::Logs, 'l') | (Screen::Logs, 'L') => {
                self.log_filter.level = self.log_filter.level.next();
                self.notify(self.log_filter.summary());
            }
            (Screen::Logs, 'c') => {
                self.log_filter = LogFilter::default();
                self.notify("filter log dibersihkan");
            }
            (Screen::Logs, '/') => self.open_filter(FilterTarget::Logs),
            (Screen::Settings, 's') => self.execute("settings:save"),
            _ => {}
        }
    }

    fn on_up(&mut self) {
        match self.screen {
            Screen::Memory => self.memory.move_selection(-1),
            Screen::Rlm => self.step_rlm_selection(-1),
            Screen::Settings => self.settings.move_selection(-1),
            _ => {}
        }
    }

    fn on_down(&mut self) {
        match self.screen {
            Screen::Memory => self.memory.move_selection(1),
            Screen::Rlm => self.step_rlm_selection(1),
            Screen::Settings => self.settings.move_selection(1),
            _ => {}
        }
    }

    fn on_enter(&mut self) {
        match self.screen {
            Screen::Memory => self.memory.open_selected(),
            Screen::Settings => self.settings.adjust(1),
            _ => {}
        }
    }

    fn escape(&mut self) {
        if self.help {
            self.help = false;
        } else if self.palette.open {
            self.palette.close();
        } else {
            self.filter_mode = false;
        }
    }

    fn open_filter(&mut self, target: FilterTarget) {
        self.filter_target = target;
        self.filter_mode = true;
        self.filter_buffer = match target {
            FilterTarget::Logs => self.log_filter.module.clone(),
            FilterTarget::Memory => self.memory.search.clone(),
        };
    }

    fn on_filter_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => self.filter_mode = false,
            KeyCode::Enter => {
                self.filter_mode = false;
                match self.filter_target {
                    FilterTarget::Logs => {
                        self.log_filter.module = self.filter_buffer.clone();
                        self.log_filter.query.clear();
                    }
                    FilterTarget::Memory => {
                        self.memory.set_search(self.filter_buffer.clone());
                        self.memory.search();
                    }
                }
                self.notify(self.log_filter.summary());
            }
            KeyCode::Backspace => {
                self.filter_buffer.pop();
            }
            KeyCode::Char(c) => self.filter_buffer.push(c),
            _ => {}
        }
    }

    fn on_palette_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => self.palette.close(),
            KeyCode::Enter => {
                if let Some(id) = self.palette.current() {
                    self.palette.close();
                    self.execute(&id);
                }
            }
            KeyCode::Up => self
                .palette
                .move_selection(-1, self.palette.filtered().len()),
            KeyCode::Down => self
                .palette
                .move_selection(1, self.palette.filtered().len()),
            KeyCode::Backspace => self.palette.pop(),
            KeyCode::Char(c) => self.palette.push(c),
            _ => {}
        }
    }

    fn step_rlm_selection(&mut self, delta: isize) {
        let len = self.rlm.snapshot().nodes.len() as isize;
        if len == 0 {
            return;
        }
        let current = self.rlm.snapshot().selected as isize;
        let next = (current + delta).rem_euclid(len) as usize;
        self.rlm.snapshot_mut().selected = next;
    }

    pub fn execute(&mut self, id: &str) {
        match id {
            "quit" => self.quit = true,
            "help:toggle" => self.help = !self.help,
            _ => self.execute_action(id),
        }
    }

    fn execute_action(&mut self, id: &str) {
        match id {
            "theme:next" => {
                self.settings.config.theme = self.settings.config.theme.next();
                self.settings.dirty = true;
                self.settings.validate();
                self.notify(format!("tema {}", self.settings.config.theme.label()));
            }
            "train:pause" => self.toggle_pause(),
            "train:checkpoint" => self.status = "checkpoint manual diminta".into(),
            "memory:refresh" => {
                self.memory.refresh();
                self.notify("memori disegarkan");
            }
            "memory:search" => {
                self.memory.search();
                self.notify(format!("{} hasil", self.memory.hits.len()));
            }
            "rlm:step" => self.status = "langkah RLM diminta".into(),
            "rlm:cancel" => self.status = "RLM dibatalkan".into(),
            "settings:save" => {
                if self.settings.save() {
                    self.notify("setelan disimpan");
                }
            }
            _ if id.starts_with("go:") => {
                if let Some(index) = screen_index(id) {
                    self.screen = Screen::from_index(index);
                }
            }
            _ => {}
        }
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        self.status = if self.paused {
            "latih dijeda"
        } else {
            "latih lanjut"
        }
        .into();
    }

    pub fn filter_level_label(&self) -> &'static str {
        self.log_filter.level.label()
    }

    pub fn filter_level_next(&mut self) {
        self.log_filter.level = self.log_filter.level.next();
    }

    pub fn log_level_color(&self, level: crate::logs::LogLevel) -> ratatui::style::Color {
        match level {
            crate::logs::LogLevel::Error => self.palette().error,
            crate::logs::LogLevel::Warn => self.palette().warn,
            crate::logs::LogLevel::Info => self.palette().fg,
            crate::logs::LogLevel::Debug => self.palette().muted,
            crate::logs::LogLevel::Trace => self.palette().lr,
        }
    }
}

fn screen_index(id: &str) -> Option<usize> {
    match id.strip_prefix("go:")? {
        "dashboard" => Some(0),
        "train" => Some(1),
        "generate" => Some(2),
        "memory" => Some(3),
        "rlm" => Some(4),
        "eval" => Some(5),
        "logs" => Some(6),
        "settings" => Some(7),
        _ => None,
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
