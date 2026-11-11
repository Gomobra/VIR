//! Sandbox eksekusi kode (dasar: subprocess + timeout + batas keluaran).

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use vir_common::VirError;

#[derive(Debug, Clone)]
pub struct Policy {
    pub timeout: Duration,
    pub max_output: usize,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            max_output: 64 * 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub truncated: bool,
}

pub fn run(program: &str, args: &[&str], policy: &Policy) -> Result<RunResult, VirError> {
    let mut child = Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| VirError::Internal(e.to_string()))?;

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if start.elapsed() >= policy.timeout {
                    let _ = child.kill();
                    // baca output yang ada
                    let out = child
                        .wait_with_output()
                        .map_err(|e| VirError::Internal(e.to_string()))?;
                    let (stdout, truncated_o) = cap(&out.stdout, policy.max_output);
                    let (stderr, truncated_e) = cap(&out.stderr, policy.max_output);
                    return Ok(RunResult {
                        stdout,
                        stderr,
                        exit_code: None,
                        timed_out: true,
                        truncated: truncated_o || truncated_e,
                    });
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => return Err(VirError::Internal(e.to_string())),
        }
    }
    let out = child
        .wait_with_output()
        .map_err(|e| VirError::Internal(e.to_string()))?;
    let (stdout, truncated_o) = cap(&out.stdout, policy.max_output);
    let (stderr, truncated_e) = cap(&out.stderr, policy.max_output);
    Ok(RunResult {
        stdout,
        stderr,
        exit_code: out.status.code(),
        timed_out: false,
        truncated: truncated_o || truncated_e,
    })
}

fn cap(bytes: &[u8], max: usize) -> (String, bool) {
    if bytes.len() <= max {
        (String::from_utf8_lossy(bytes).into_owned(), false)
    } else {
        (String::from_utf8_lossy(&bytes[..max]).into_owned(), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_echo() {
        let r = run("echo", &["hello"], &Policy::default()).unwrap();
        assert_eq!(r.stdout.trim(), "hello");
        assert_eq!(r.exit_code, Some(0));
        assert!(!r.timed_out);
    }

    #[test]
    fn run_timeout() {
        let pol = Policy {
            timeout: Duration::from_millis(100),
            ..Policy::default()
        };
        let r = run("sleep", &["10"], &pol).unwrap();
        assert!(r.timed_out);
    }

    #[test]
    fn run_truncate() {
        let pol = Policy {
            max_output: 4,
            ..Policy::default()
        };
        let r = run("printf", &["abcdefgh"], &pol).unwrap();
        assert!(r.truncated);
        assert_eq!(r.stdout.len(), 4);
    }
}
