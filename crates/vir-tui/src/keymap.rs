//! Peta tombol dan bantuan; semuanya punya alternatif tanpa tombol fungsi.

#[derive(Debug, Clone, Copy)]
pub struct KeyHint {
    pub keys: &'static str,
    pub desc: &'static str,
}

pub const HELP: &[KeyHint] = &[
    KeyHint {
        keys: "1-8",
        desc: "pindah layar",
    },
    KeyHint {
        keys: "← →",
        desc: "layar sebelumnya/berikutnya",
    },
    KeyHint {
        keys: "Ctrl+P / :",
        desc: "palet perintah",
    },
    KeyHint {
        keys: "?",
        desc: "bantuan",
    },
    KeyHint {
        keys: "L",
        desc: "saring level log",
    },
    KeyHint {
        keys: "/",
        desc: "cari memori/log",
    },
    KeyHint {
        keys: "Enter",
        desc: "pilih/buka",
    },
    KeyHint {
        keys: "s",
        desc: "simpan setelan",
    },
    KeyHint {
        keys: "q",
        desc: "keluar",
    },
];

pub const COMMANDS: &[&str] = &[
    "go:dashboard",
    "go:train",
    "go:generate",
    "go:memory",
    "go:rlm",
    "go:eval",
    "go:logs",
    "go:settings",
    "theme:next",
    "help:toggle",
    "train:pause",
    "train:checkpoint",
    "memory:refresh",
    "memory:search",
    "rlm:step",
    "rlm:cancel",
    "settings:save",
    "quit",
];

pub fn command_label(id: &str) -> String {
    let text = id.replace(':', " · ");
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}
