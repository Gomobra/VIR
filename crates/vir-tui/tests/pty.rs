//! F3.8.h — tes PTY: mulai, render, lalu keluar bersih lewat pty sungguhan.
//!
//! `script` dipakai agar biner berjalan di terminal semu (raw mode + `/dev/tty`).
//! Bila `script` tidak tersedia (mis. Termux tanpa bsdutils) tes dilewati.

use std::io::Write;
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_vir-tui");
const TIMEOUT: Duration = Duration::from_secs(20);

fn script_available() -> bool {
    Command::new("sh")
        .args(["-c", "command -v script >/dev/null 2>&1"])
        .status()
        .is_ok_and(|status| status.success())
}

/// Perintah pty: set ukuran jendela dulu, lalu ganti proses dengan biner TUI.
fn pty_command() -> String {
    format!("stty rows 24 cols 80 2>/dev/null; exec {BIN}")
}

fn wait_for_exit(child: &mut Child) -> bool {
    let deadline = Instant::now() + TIMEOUT;
    while Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(_)) => return true,
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(_) => return false,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    false
}

/// Jalankan `command` di pty, kirim `keys`, lalu kumpulkan keluaran.
fn run_in_pty(command: &str, keys: &str, no_color: bool) -> Option<Output> {
    let mut cmd = Command::new("script");
    cmd.args(["-qec", command, "/dev/null"])
        .env("TERM", "xterm-256color")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if no_color {
        cmd.env("NO_COLOR", "1");
    } else {
        cmd.env_remove("NO_COLOR");
    }
    let mut child = cmd.spawn().ok()?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(keys.as_bytes());
        let _ = stdin.flush();
    }
    if !wait_for_exit(&mut child) {
        eprintln!("script tidak selesai dalam {TIMEOUT:?}");
        return None;
    }
    child.wait_with_output().ok()
}

#[test]
fn pty_mulai_render_lalu_keluar_bersih() {
    if !script_available() {
        eprintln!("lewati: perintah `script` tidak tersedia");
        return;
    }
    let output = run_in_pty(&pty_command(), "q\n", false).expect("spawn pty");
    assert!(output.status.success(), "keluar harus sukses: {output:?}");
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("VIR"), "dasbor harus tampil di pty:\n{text}");
}

#[test]
fn pty_no_color_mematikan_warna_truecolor() {
    if !script_available() {
        eprintln!("lewati: perintah `script` tidak tersedia");
        return;
    }
    let colored = run_in_pty(&pty_command(), "q\n", false).expect("pty warna");
    let plain = run_in_pty(&pty_command(), "q\n", true).expect("pty NO_COLOR");

    let colored = String::from_utf8_lossy(&colored.stdout);
    let plain = String::from_utf8_lossy(&plain.stdout);
    assert!(
        colored.contains("38;2;"),
        "tanpa NO_COLOR harus memakai truecolor:\n{colored}"
    );
    assert!(
        !plain.contains("38;2;"),
        "NO_COLOR harus mematikan RGB:\n{plain}"
    );
}

#[test]
fn tanpa_tty_biner_tidak_menggantung() {
    let mut child = Command::new(BIN)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn biner TUI");
    let exited = wait_for_exit(&mut child);
    let output = child.wait_with_output().expect("kumpulkan keluaran");
    assert!(exited, "TUI harus keluar tanpa terminal (F3.8.a pemulihan)");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("vir-tui"),
            "pesan galat harus menyebut biner: {stderr}"
        );
    }
}
