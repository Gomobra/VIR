//! Setelan: formulir `Config`, validasi inline, tema, dan penyimpanan TOML.

use crate::theme::{ColorMode, ThemeName};
use std::fs;
use std::path::Path;
use vir_common::ModelConfig;

const DEFAULT_PATH: &str = "configs/tui.toml";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiConfig {
    pub theme: ThemeName,
    pub color_mode: ColorMode,
    pub colorblind_safe: bool,
    pub low_bandwidth: bool,
    pub fps: u16,
    pub low_bandwidth_fps: u16,
    pub palette_key: String,
    pub help_key: String,
    pub quit_key: String,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            theme: ThemeName::Dark,
            color_mode: ColorMode::Truecolor,
            colorblind_safe: true,
            low_bandwidth: false,
            fps: 30,
            low_bandwidth_fps: 5,
            palette_key: "ctrl+p".into(),
            help_key: "?".into(),
            quit_key: "q".into(),
        }
    }
}

impl TuiConfig {
    pub fn load(path: &Path) -> Self {
        let Ok(text) = fs::read_to_string(path) else {
            return Self::default();
        };
        parse_toml(&text).unwrap_or_default()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.fps == 0 || self.fps > 120 {
            return Err("fps harus 1..=120".into());
        }
        if self.low_bandwidth_fps == 0 || self.low_bandwidth_fps > self.fps {
            return Err("fps hemat harus 1..=fps".into());
        }
        if self.palette_key.trim().is_empty() || self.quit_key.trim().is_empty() {
            return Err("peta tombol tidak boleh kosong".into());
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        fs::write(path, serialize_toml(self)).map_err(|e| e.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Row {
    Theme,
    ColorMode,
    Colorblind,
    LowBandwidth,
    Fps,
    LowFps,
    Palette,
    Help,
    Quit,
    ModelTargetLength,
    ModelLevels,
    ModelHidden,
    ModelHeads,
    ModelCodebook,
    ModelEma,
}

impl Row {
    pub const ALL: [Self; 15] = [
        Row::Theme,
        Row::ColorMode,
        Row::Colorblind,
        Row::LowBandwidth,
        Row::Fps,
        Row::LowFps,
        Row::Palette,
        Row::Help,
        Row::Quit,
        Row::ModelTargetLength,
        Row::ModelLevels,
        Row::ModelHidden,
        Row::ModelHeads,
        Row::ModelCodebook,
        Row::ModelEma,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Row::Theme => "theme",
            Row::ColorMode => "color_mode",
            Row::Colorblind => "colorblind_safe",
            Row::LowBandwidth => "low_bandwidth",
            Row::Fps => "fps",
            Row::LowFps => "low_bandwidth_fps",
            Row::Palette => "keymap.palette",
            Row::Help => "keymap.help",
            Row::Quit => "keymap.quit",
            Row::ModelTargetLength => "model.target_length",
            Row::ModelLevels => "model.num_levels",
            Row::ModelHidden => "model.hidden_dim",
            Row::ModelHeads => "model.num_heads",
            Row::ModelCodebook => "model.codebook_size",
            Row::ModelEma => "model.ema_decay",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SettingsView {
    pub path: String,
    pub config: TuiConfig,
    pub model: ModelConfig,
    pub selected: usize,
    pub error: Option<String>,
    pub dirty: bool,
    pub saved: bool,
}

impl SettingsView {
    pub fn new(path: impl Into<String>) -> Self {
        let path = path.into();
        let config = TuiConfig::load(Path::new(&path));
        Self {
            path,
            config,
            model: ModelConfig::default(),
            selected: 0,
            error: None,
            dirty: false,
            saved: false,
        }
    }

    pub fn row(&self) -> Row {
        Row::ALL[self.selected.min(Row::ALL.len() - 1)]
    }

    pub fn value(&self, row: Row) -> String {
        match row {
            Row::Theme => self.config.theme.label().into(),
            Row::ColorMode => self.config.color_mode.label().into(),
            Row::Colorblind => self.config.colorblind_safe.to_string(),
            Row::LowBandwidth => self.config.low_bandwidth.to_string(),
            Row::Fps => self.config.fps.to_string(),
            Row::LowFps => self.config.low_bandwidth_fps.to_string(),
            Row::Palette => self.config.palette_key.clone(),
            Row::Help => self.config.help_key.clone(),
            Row::Quit => self.config.quit_key.clone(),
            Row::ModelTargetLength => self.model.target_length.to_string(),
            Row::ModelLevels => self.model.num_levels.to_string(),
            Row::ModelHidden => self.model.hidden_dim.to_string(),
            Row::ModelHeads => self.model.num_heads.to_string(),
            Row::ModelCodebook => self.model.codebook_size.to_string(),
            Row::ModelEma => format!("{:.3}", self.model.ema_decay),
        }
    }

    pub fn move_selection(&mut self, delta: isize) {
        let len = Row::ALL.len() as isize;
        self.selected = (self.selected as isize + delta).rem_euclid(len) as usize;
    }

    pub fn adjust(&mut self, delta: i32) {
        let row = self.row();
        self.dirty = true;
        self.saved = false;
        self.adjust_row(row, delta);
        self.validate();
    }

    fn adjust_row(&mut self, row: Row, delta: i32) {
        match row {
            Row::Theme => self.config.theme = cycle_theme(self.config.theme, delta),
            Row::ColorMode => self.config.color_mode = cycle_mode(self.config.color_mode, delta),
            Row::Colorblind => self.config.colorblind_safe = !self.config.colorblind_safe,
            Row::LowBandwidth => self.config.low_bandwidth = !self.config.low_bandwidth,
            Row::Fps => adjust_u16(&mut self.config.fps, delta, 1, 120),
            Row::LowFps => adjust_u16(
                &mut self.config.low_bandwidth_fps,
                delta,
                1,
                self.config.fps,
            ),
            Row::Palette | Row::Help | Row::Quit => {}
            _ => self.adjust_model(row, delta),
        }
    }

    fn adjust_model(&mut self, row: Row, delta: i32) {
        match row {
            Row::ModelTargetLength => {
                adjust_usize(&mut self.model.target_length, delta, 1, 1_000_000)
            }
            Row::ModelLevels => adjust_usize(&mut self.model.num_levels, delta, 1, 1_000),
            Row::ModelHidden => adjust_usize(&mut self.model.hidden_dim, delta * 64, 64, 1_000_000),
            Row::ModelHeads => adjust_usize(&mut self.model.num_heads, delta, 1, 1_000),
            Row::ModelCodebook => {
                adjust_usize(&mut self.model.codebook_size, delta * 64, 1, 1_000_000)
            }
            Row::ModelEma => {
                self.model.ema_decay =
                    (self.model.ema_decay + delta as f32 * 0.01).clamp(0.01, 0.999)
            }
            _ => {}
        }
    }

    pub fn validate(&mut self) {
        self.error = None;
        if let Err(error) = self.config.validate() {
            self.error = Some(error);
            return;
        }
        if let Err(error) = self.model.validate() {
            self.error = Some(error.to_string());
        }
    }

    pub fn save(&mut self) -> bool {
        self.validate();
        if self.error.is_some() {
            return false;
        }
        match self.config.save(Path::new(&self.path)) {
            Ok(()) => {
                self.dirty = false;
                self.saved = true;
                true
            }
            Err(error) => {
                self.error = Some(error);
                false
            }
        }
    }

    pub fn validation_text(&self) -> String {
        self.error.clone().unwrap_or_else(|| {
            if self.dirty {
                "ubah belum disimpan · s untuk simpan".into()
            } else {
                "valid".into()
            }
        })
    }
}

fn cycle_theme(value: ThemeName, delta: i32) -> ThemeName {
    if delta < 0 {
        value.prev()
    } else {
        value.next()
    }
}

fn cycle_mode(value: ColorMode, delta: i32) -> ColorMode {
    if delta < 0 {
        value.prev()
    } else {
        value.next()
    }
}

fn adjust_u16(value: &mut u16, delta: i32, min: u16, max: u16) {
    let step = i16::try_from(delta).unwrap_or(i16::MAX);
    *value = value.saturating_add_signed(step).clamp(min, max.max(min));
}

fn adjust_usize(value: &mut usize, delta: i32, min: usize, max: usize) {
    let step = usize::try_from(delta.unsigned_abs()).unwrap_or(usize::MAX);
    *value = if delta < 0 {
        value.saturating_sub(step).max(min)
    } else {
        value.saturating_add(step).min(max)
    };
}

fn parse_toml(text: &str) -> Option<TuiConfig> {
    let mut config = TuiConfig::default();
    let mut section = String::new();
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            section = name.trim().to_string();
            continue;
        }
        let (key, value) = line.split_once('=')?;
        apply_pair(&mut config, &section, key.trim(), value.trim());
    }
    Some(config)
}

fn apply_pair(config: &mut TuiConfig, section: &str, key: &str, value: &str) {
    let value = value.trim_matches('"');
    match (section, key) {
        ("theme", "name") => {
            if let Some(theme) = ThemeName::parse(value) {
                config.theme = theme;
            }
        }
        ("theme", "color_mode") => {
            config.color_mode = match value {
                "256" | "ansi256" => ColorMode::Ansi256,
                "none" | "no_color" => ColorMode::None,
                _ => ColorMode::Truecolor,
            };
        }
        ("theme", "colorblind_safe") => config.colorblind_safe = value == "true",
        ("input", "low_bandwidth") => config.low_bandwidth = value == "true",
        ("input", "fps") => config.fps = value.parse().unwrap_or(config.fps),
        ("input", "low_bandwidth_fps") => {
            config.low_bandwidth_fps = value.parse().unwrap_or(config.low_bandwidth_fps);
        }
        ("keymap", "palette") => config.palette_key = value.to_string(),
        ("keymap", "help") => config.help_key = value.to_string(),
        ("keymap", "quit") => config.quit_key = value.to_string(),
        _ => {}
    }
}

fn serialize_toml(config: &TuiConfig) -> String {
    format!(
        "[theme]\nname = \"{}\"\ncolor_mode = \"{}\"\ncolorblind_safe = {}\n\n\
         [input]\nlow_bandwidth = {}\nfps = {}\nlow_bandwidth_fps = {}\n\n\
         [keymap]\npalette = \"{}\"\nhelp = \"{}\"\nquit = \"{}\"\n",
        config.theme.label(),
        config.color_mode.label(),
        config.colorblind_safe,
        config.low_bandwidth,
        config.fps,
        config.low_bandwidth_fps,
        config.palette_key,
        config.help_key,
        config.quit_key,
    )
}

pub fn default_path() -> &'static str {
    DEFAULT_PATH
}
