//! Tema dan palet: gelap, terang, kontras tinggi, ramah buta warna, NO_COLOR.

use ratatui::style::Color;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeName {
    Dark,
    Light,
    HighContrast,
    Colorblind,
}

impl ThemeName {
    pub const ALL: [Self; 4] = [
        ThemeName::Dark,
        ThemeName::Light,
        ThemeName::HighContrast,
        ThemeName::Colorblind,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ThemeName::Dark => "dark",
            ThemeName::Light => "light",
            ThemeName::HighContrast => "high-contrast",
            ThemeName::Colorblind => "colorblind",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "dark" => Some(Self::Dark),
            "light" => Some(Self::Light),
            "high-contrast" | "high_contrast" | "hc" => Some(Self::HighContrast),
            "colorblind" | "cb" => Some(Self::Colorblind),
            _ => None,
        }
    }

    pub fn next(self) -> Self {
        let index = Self::ALL.iter().position(|t| *t == self).unwrap_or(0);
        Self::ALL[(index + 1) % Self::ALL.len()]
    }

    pub fn prev(self) -> Self {
        let index = Self::ALL.iter().position(|t| *t == self).unwrap_or(0);
        Self::ALL[(index + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    Truecolor,
    Ansi256,
    None,
}

impl ColorMode {
    pub fn label(self) -> &'static str {
        match self {
            ColorMode::Truecolor => "truecolor",
            ColorMode::Ansi256 => "256",
            ColorMode::None => "none",
        }
    }

    pub fn next(self) -> Self {
        match self {
            ColorMode::Truecolor => ColorMode::Ansi256,
            ColorMode::Ansi256 => ColorMode::None,
            ColorMode::None => ColorMode::Truecolor,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            ColorMode::Truecolor => ColorMode::None,
            ColorMode::Ansi256 => ColorMode::Truecolor,
            ColorMode::None => ColorMode::Ansi256,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Palette {
    pub bg: Color,
    pub panel: Color,
    pub fg: Color,
    pub muted: Color,
    pub accent: Color,
    pub loss: Color,
    pub lr: Color,
    pub ok: Color,
    pub warn: Color,
    pub error: Color,
    pub selection: Color,
}

impl Palette {
    pub fn new(theme: ThemeName, mode: ColorMode, colorblind: bool) -> Self {
        let base = match theme {
            ThemeName::Dark => Self::dark(),
            ThemeName::Light => Self::light(),
            ThemeName::HighContrast => Self::high_contrast(),
            ThemeName::Colorblind => Self::colorblind(),
        };
        Self::adapt(base, mode, colorblind)
    }

    fn dark() -> Self {
        Self {
            bg: Color::Rgb(12, 14, 18),
            panel: Color::Rgb(20, 24, 32),
            fg: Color::Rgb(226, 232, 240),
            muted: Color::Rgb(120, 132, 150),
            accent: Color::Rgb(86, 190, 255),
            loss: Color::Rgb(255, 138, 101),
            lr: Color::Rgb(129, 212, 250),
            ok: Color::Rgb(102, 204, 153),
            warn: Color::Rgb(255, 204, 102),
            error: Color::Rgb(255, 107, 107),
            selection: Color::Rgb(45, 62, 82),
        }
    }

    fn light() -> Self {
        Self {
            bg: Color::Rgb(248, 250, 252),
            panel: Color::Rgb(255, 255, 255),
            fg: Color::Rgb(24, 30, 40),
            muted: Color::Rgb(100, 112, 128),
            accent: Color::Rgb(0, 105, 190),
            loss: Color::Rgb(190, 60, 30),
            lr: Color::Rgb(0, 115, 160),
            ok: Color::Rgb(20, 120, 80),
            warn: Color::Rgb(150, 100, 0),
            error: Color::Rgb(175, 30, 30),
            selection: Color::Rgb(210, 228, 245),
        }
    }

    fn high_contrast() -> Self {
        Self {
            bg: Color::Black,
            panel: Color::Black,
            fg: Color::White,
            muted: Color::Rgb(190, 190, 190),
            accent: Color::Yellow,
            loss: Color::LightRed,
            lr: Color::Cyan,
            ok: Color::LightGreen,
            warn: Color::Yellow,
            error: Color::LightMagenta,
            selection: Color::Rgb(60, 60, 60),
        }
    }

    fn colorblind() -> Self {
        Self {
            bg: Color::Rgb(16, 18, 22),
            panel: Color::Rgb(26, 30, 38),
            fg: Color::Rgb(236, 240, 244),
            muted: Color::Rgb(140, 150, 165),
            accent: Color::Rgb(86, 156, 214),
            loss: Color::Rgb(230, 159, 0),
            lr: Color::Rgb(0, 158, 115),
            ok: Color::Rgb(0, 158, 115),
            warn: Color::Rgb(240, 228, 66),
            error: Color::Rgb(213, 94, 0),
            selection: Color::Rgb(50, 62, 78),
        }
    }

    fn adapt(mut self, mode: ColorMode, colorblind: bool) -> Self {
        if colorblind && self.loss == Self::dark().loss {
            self.loss = Self::colorblind().loss;
            self.lr = Self::colorblind().lr;
        }
        match mode {
            ColorMode::Truecolor => self,
            ColorMode::Ansi256 => Self {
                bg: Color::Indexed(234),
                panel: Color::Indexed(236),
                fg: Color::Indexed(255),
                muted: Color::Indexed(245),
                accent: Color::Indexed(75),
                loss: Color::Indexed(209),
                lr: Color::Indexed(81),
                ok: Color::Indexed(78),
                warn: Color::Indexed(221),
                error: Color::Indexed(203),
                selection: Color::Indexed(238),
            },
            ColorMode::None => Self {
                bg: Color::Reset,
                panel: Color::Reset,
                fg: Color::Reset,
                muted: Color::Reset,
                accent: Color::Reset,
                loss: Color::Reset,
                lr: Color::Reset,
                ok: Color::Reset,
                warn: Color::Reset,
                error: Color::Reset,
                selection: Color::Reset,
            },
        }
    }
}

pub fn detect_mode(env: bool) -> ColorMode {
    if env {
        ColorMode::None
    } else {
        ColorMode::Truecolor
    }
}
