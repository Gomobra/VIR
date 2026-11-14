//! REPL stateful di atas sandbox (F3.2).
//!
//! Proses CPython berumur panjang membaca perintah dari pipa. Protokol per
//! baris memakai JSON untuk perintah dan pesan kontrol; keluaran kode ditangkap
//! dan dikembalikan sebagai satu pesan `done` agar tidak tercampur dengan
//! keluaran protokol.
//!
//! Fungsi `mem` dan `llm_query` disuntikkan ke namespace Python dan memanggil
//! balik ke host Rust lewat RPC di pipa yang sama.
//!
//! Bila proses mati (crash, OOM, sinyal), [`Repl::exec`] menyalakan ulang proses
//! dan memulihkan status yang dapat diserialisasi.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStderr, ChildStdin};
use std::time::Duration;

use serde_json::{Value as JsonValue, json};
use vir_common::VirError;
use vir_sandbox::{Policy, spawn};

/// Awalan pesan RPC dari Python ke host.
const TAG_RPC: &str = "__VIR_RPC__ ";
/// Awalan pesan selesai (hasil exec).
const TAG_DONE: &str = "__VIR_DONE__ ";
/// Awalan snapshot status.
const TAG_STATE: &str = "__VIR_STATE__ ";

/// Permintaan RPC dari Python.
#[derive(Debug, Clone)]
pub struct RpcRequest {
    pub method: String,
    pub payload: JsonValue,
}

impl RpcRequest {
    fn from_value(value: &JsonValue) -> Self {
        Self {
            method: value
                .get("method")
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .to_string(),
            payload: value.get("payload").cloned().unwrap_or(JsonValue::Null),
        }
    }
}

/// Hasil satu eksekusi kode.
#[derive(Debug, Clone)]
pub struct ExecOutcome {
    pub output: String,
    pub error: Option<String>,
    /// Proses mati sebelum mengirim hasil.
    pub crashed: bool,
    /// Sesi dinyalakan ulang selama eksekusi ini.
    pub recovered: bool,
}

impl ExecOutcome {
    fn crashed(output: String) -> Self {
        Self {
            output,
            error: Some("proses REPL mati".to_string()),
            crashed: true,
            recovered: false,
        }
    }
}

/// Sesi REPL stateful.
pub struct Repl {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
    program: String,
    args: Vec<String>,
    policy: Policy,
    last_state: Option<JsonValue>,
}

impl Repl {
    /// Mulai REPL dengan `python3` dan kebijakan bawaan.
    pub fn start() -> Result<Self, VirError> {
        Self::start_with("python3", repl_policy())
    }

    /// Mulai REPL dengan program dan kebijakan khusus.
    pub fn start_with(program: &str, policy: Policy) -> Result<Self, VirError> {
        let args = vec!["-u".to_string(), "-c".to_string(), BOOTSTRAP.to_string()];
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let mut child = spawn(program, &refs, &policy)?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| VirError::Internal("stdin REPL tidak tersedia".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| VirError::Internal("stdout REPL tidak tersedia".into()))?;
        if let Some(stderr) = child.stderr.take() {
            std::thread::spawn(move || drain(stderr));
        }
        Ok(Self {
            child,
            stdin,
            reader: BufReader::new(stdout),
            program: program.to_string(),
            args,
            policy,
            last_state: None,
        })
    }

    /// Eksekusi kode. Bila proses mati, sesi dinyalakan ulang dan kode diulang.
    pub fn exec<H>(&mut self, code: &str, handler: &mut H) -> Result<ExecOutcome, VirError>
    where
        H: FnMut(&RpcRequest) -> Result<JsonValue, VirError>,
    {
        let first = self.send(code, handler)?;
        if !first.crashed {
            return Ok(first);
        }
        self.restart()?;
        let mut second = self.send(code, handler)?;
        second.recovered = true;
        Ok(second)
    }

    /// Nyalakan ulang proses dan pulihkan status terakhir.
    pub fn restart(&mut self) -> Result<(), VirError> {
        self.respawn()?;
        if let Some(state) = self.last_state.clone() {
            let json_text =
                serde_json::to_string(&state).map_err(|e| VirError::Internal(e.to_string()))?;
            // `json_text` adalah teks JSON; bungkus lagi sebagai literal string
            // JSON agar `json.loads` menerima string, bukan objek.
            let literal =
                serde_json::to_string(&json_text).map_err(|e| VirError::Internal(e.to_string()))?;
            let code = format!("globals().update(json.loads({literal}))");
            let mut noop = |_req: &RpcRequest| Ok(JsonValue::Null);
            let outcome = self.send(&code, &mut noop)?;
            if outcome.crashed {
                return Err(VirError::Internal(
                    "REPL tidak dapat dipulihkan setelah crash".into(),
                ));
            }
        }
        Ok(())
    }

    /// Matikan proses (tanpa menunggu).
    pub fn kill(&mut self) {
        let pid = self.child.id();
        vir_sandbox::limits::kill_group(pid);
        let _ = self.child.kill();
    }

    fn respawn(&mut self) -> Result<(), VirError> {
        let pid = self.child.id();
        vir_sandbox::limits::kill_group(pid);
        let _ = self.child.kill();
        let _ = self.child.wait();

        let refs: Vec<&str> = self.args.iter().map(String::as_str).collect();
        let child = spawn(&self.program, &refs, &self.policy)?;
        self.child = child;
        self.stdin = self
            .child
            .stdin
            .take()
            .ok_or_else(|| VirError::Internal("stdin REPL tidak tersedia".into()))?;
        let stdout = self
            .child
            .stdout
            .take()
            .ok_or_else(|| VirError::Internal("stdout REPL tidak tersedia".into()))?;
        self.reader = BufReader::new(stdout);
        if let Some(stderr) = self.child.stderr.take() {
            std::thread::spawn(move || drain(stderr));
        }
        Ok(())
    }

    fn send<H>(&mut self, code: &str, handler: &mut H) -> Result<ExecOutcome, VirError>
    where
        H: FnMut(&RpcRequest) -> Result<JsonValue, VirError>,
    {
        let request = json!({ "code": code });
        if !write_line(&mut self.stdin, &request.to_string()) {
            return Ok(ExecOutcome::crashed(String::new()));
        }
        let mut output = String::new();
        loop {
            let mut line = String::new();
            let n = match self.reader.read_line(&mut line) {
                Ok(n) => n,
                Err(_) => return Ok(ExecOutcome::crashed(output)),
            };
            if n == 0 {
                return Ok(ExecOutcome::crashed(output));
            }
            if let Some(rest) = line.strip_prefix(TAG_RPC) {
                let value: JsonValue = match serde_json::from_str(rest.trim()) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let rpc = RpcRequest::from_value(&value);
                let response = match handler(&rpc) {
                    Ok(v) => json!({ "result": v }),
                    Err(e) => json!({ "error": e.to_string() }),
                };
                if !write_line(&mut self.stdin, &response.to_string()) {
                    return Ok(ExecOutcome::crashed(output));
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix(TAG_STATE) {
                if let Ok(state) = serde_json::from_str::<JsonValue>(rest.trim()) {
                    self.last_state = Some(state);
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix(TAG_DONE) {
                let value: JsonValue = match serde_json::from_str(rest.trim()) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let ok = value
                    .get("ok")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let error = value
                    .get("error")
                    .and_then(JsonValue::as_str)
                    .map(str::to_string);
                if let Some(s) = value.get("stdout").and_then(JsonValue::as_str) {
                    output.push_str(s);
                }
                if let Some(s) = value.get("stderr").and_then(JsonValue::as_str) {
                    output.push_str(s);
                }
                return Ok(ExecOutcome {
                    output,
                    error: if ok {
                        None
                    } else {
                        Some(error.unwrap_or_else(|| "galat repl".to_string()))
                    },
                    crashed: false,
                    recovered: false,
                });
            }
            output.push_str(&line);
        }
    }
}

impl Drop for Repl {
    fn drop(&mut self) {
        let pid = self.child.id();
        vir_sandbox::limits::kill_group(pid);
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Tulis satu baris; `true` bila berhasil.
fn write_line(stdin: &mut ChildStdin, line: &str) -> bool {
    writeln!(stdin, "{line}").is_ok() && stdin.flush().is_ok()
}

fn drain(mut err: ChildStderr) {
    use std::io::Read;
    let mut buf = [0u8; 1024];
    while let Ok(n) = err.read(&mut buf) {
        if n == 0 {
            break;
        }
    }
}

fn repl_policy() -> Policy {
    Policy {
        timeout: Duration::from_secs(60),
        max_output: 16 * 1024 * 1024,
        workdir: None,
        clear_env: false,
        env: std::collections::BTreeMap::new(),
        limits: vir_sandbox::limits::ResourceLimits {
            cpu_secs: None,
            // Bionic (Android) memetakan shadow CFI yang besar; RLIMIT_AS apa
            // pun membuat linker gagal. Di luar Android, batasi 2 GiB.
            mem_bytes: if cfg!(target_os = "android") {
                None
            } else {
                Some(2 * 1024 * 1024 * 1024)
            },
            nofile: Some(1024),
            nproc: None,
        },
        allow_write: Vec::new(),
        landlock: false,
    }
}

/// Program bootstrap CPython yang dijalankan di dalam sandbox.
const BOOTSTRAP: &str = r#"import sys, json, io

TAG_RPC = "__VIR_RPC__ "
TAG_DONE = "__VIR_DONE__ "
TAG_STATE = "__VIR_STATE__ "
_RESERVED = {"mem", "llm_query", "_rpc", "_emit", "_is_state", "_snapshot", "_run", "sys", "json", "io", "TAG_RPC", "TAG_DONE", "TAG_STATE", "line", "req"}

def _emit(tag, obj):
    out = sys.__stdout__
    out.write(tag + json.dumps(obj) + "\n")
    out.flush()

def _rpc(method, payload):
    _emit(TAG_RPC, {"method": method, "payload": payload})
    resp = sys.stdin.readline()
    if not resp:
        raise RuntimeError("rpc: host menutup pipa")
    return json.loads(resp)

def mem(op, *args, **kwargs):
    resp = _rpc("mem", {"op": op, "args": list(args), "kwargs": kwargs})
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp.get("result")

def llm_query(prompt, **kwargs):
    resp = _rpc("llm_query", {"prompt": prompt, "kwargs": kwargs})
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp.get("result")

def _is_state(v):
    if isinstance(v, (type(None), bool, int, float, str)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_state(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_state(x) for k, x in v.items())
    return False

def _snapshot():
    out = {}
    for k, v in list(globals().items()):
        if k in _RESERVED or k.startswith("__"):
            continue
        if _is_state(v):
            out[k] = v
    return out

def _run(req):
    code = req["code"]
    buf_out, buf_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf_out, buf_err
    ok, err = True, None
    try:
        exec(compile(code, "<vir-repl>", "exec"), globals())
    except BaseException as exc:
        ok = False
        err = type(exc).__name__ + ": " + str(exc)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    _emit(TAG_DONE, {"ok": ok, "error": err, "stdout": buf_out.getvalue(), "stderr": buf_err.getvalue()})
    try:
        _emit(TAG_STATE, _snapshot())
    except BaseException:
        pass

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        req = json.loads(line)
    except Exception:
        continue
    if isinstance(req, dict) and "code" in req:
        _run(req)
"#;
