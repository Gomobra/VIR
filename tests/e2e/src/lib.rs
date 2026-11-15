//! F1.8 — validasi keluaran `vir smoke` dan gerbang G1.
//!
//! Modul ini tidak menjalankan model: ia memvalidasi skema JSON yang dihasilkan
//! perintah `vir smoke` (tokenizer → VQ → generator → decoder → sampling) dan
//! menegakkan kriteria gerbang G1 ketika field-nya tersedia. Dengan begitu CI
//! gagal bila rekonstruksi tidak naik per level, loss tidak turun, atau model
//! tidak mereproduksi 8 sampel persis (PRD F1.8 dan G1).

use serde_json::{Map, Value};

/// Statistik satu level kuantizer pada laporan `vir smoke`.
#[derive(Clone, Debug, Default)]
pub struct LevelStat {
    /// Indeks level `k` (1..=K).
    pub k: usize,
    /// Panjang kumulatif `ℓ_k` (posisi kode level ini).
    pub ell: usize,
    /// Akurasi rekonstruksi kumulatif pada level ini (0..=1).
    pub recon: f64,
    /// Loss generator (NCE) pada level ini.
    pub loss: f64,
}

/// Laporan `vir smoke` (F1.8). Field opsional hanya ada pada pipeline penuh.
#[derive(Clone, Debug, Default)]
pub struct SmokeReport {
    /// `"ok"` bila seluruh gerbang lulus.
    pub status: String,
    /// Info build (versi, target, profil).
    pub build: Option<String>,
    /// Uji roundtrip tokenizer (kompat lama).
    pub roundtrip_ok: Option<bool>,
    /// Model mereproduksi 8 sampel persis (G1).
    pub exact_match: Option<bool>,
    /// `Σℓ_k / (K·L)` menurut jadwal F1.1.
    pub recon_fraction: Option<f64>,
    /// Loss sebelum kepala dilatih.
    pub loss_before: Option<f64>,
    /// Loss sesudah kepala dilatih.
    pub loss_after: Option<f64>,
    /// Statistik per level (terurut `k` naik).
    pub levels: Vec<LevelStat>,
}

impl SmokeReport {
    /// Tegakkan kriteria G1 atas field yang tersedia (PRD F1.8).
    pub fn gate_g1_ok(&self) -> Result<(), String> {
        if self.status != "ok" {
            return Err(format!("status smoke '{}'", self.status));
        }
        if self.roundtrip_ok == Some(false) {
            return Err("roundtrip tokenizer gagal".to_string());
        }
        if self.exact_match == Some(false) {
            return Err("reproduksi persis 8 sampel gagal".to_string());
        }
        if let (Some(before), Some(after)) = (self.loss_before, self.loss_after) {
            if after >= before {
                return Err(format!("loss tidak turun: {before} → {after}"));
            }
        }
        self.check_recon_nondecreasing()
    }

    /// Rekonstruksi kumulatif tidak boleh menurun antar level (G1).
    fn check_recon_nondecreasing(&self) -> Result<(), String> {
        for pair in self.levels.windows(2) {
            if pair[1].recon < pair[0].recon {
                return Err(format!(
                    "rekonstruksi level {} tidak naik: {} → {}",
                    pair[1].k, pair[0].recon, pair[1].recon
                ));
            }
        }
        Ok(())
    }
}

/// Validasi skema JSON `vir smoke`.
pub fn validate_smoke_json(text: &str) -> Result<SmokeReport, String> {
    let value: Value =
        serde_json::from_str(text).map_err(|e| format!("JSON smoke tidak valid: {e}"))?;
    let obj = value.as_object().ok_or("smoke harus objek JSON")?;
    let status = obj
        .get("status")
        .and_then(Value::as_str)
        .ok_or("field 'status' (string) wajib")?
        .to_string();
    if status != "ok" && status != "fail" {
        return Err(format!("status '{}' tidak dikenal", status));
    }
    Ok(SmokeReport {
        build: opt_str(obj, "build"),
        roundtrip_ok: opt_bool(obj, "roundtrip_ok")?,
        exact_match: opt_bool(obj, "exact_match")?,
        recon_fraction: opt_f64(obj, "recon_fraction")?,
        loss_before: opt_f64(obj, "loss_before")?,
        loss_after: opt_f64(obj, "loss_after")?,
        levels: parse_levels(obj)?,
        status,
    })
}

/// Ambil field string opsional.
fn opt_str(obj: &Map<String, Value>, key: &str) -> Option<String> {
    obj.get(key).and_then(Value::as_str).map(str::to_string)
}

/// Ambil field f64 opsional; galat bila ada tetapi bukan angka.
fn opt_f64(obj: &Map<String, Value>, key: &str) -> Result<Option<f64>, String> {
    match obj.get(key) {
        None => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("'{key}' harus angka")),
    }
}

/// Ambil field bool opsional; galat bila ada tetapi bukan bool.
fn opt_bool(obj: &Map<String, Value>, key: &str) -> Result<Option<bool>, String> {
    match obj.get(key) {
        None => Ok(None),
        Some(v) => v
            .as_bool()
            .map(Some)
            .ok_or_else(|| format!("'{key}' harus bool")),
    }
}

/// Parse `levels` bila ada, lalu urutkan menurut `k`.
fn parse_levels(obj: &Map<String, Value>) -> Result<Vec<LevelStat>, String> {
    let Some(arr) = obj.get("levels") else {
        return Ok(Vec::new());
    };
    let arr = arr.as_array().ok_or("'levels' harus array")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        out.push(parse_level(item)?);
    }
    out.sort_by_key(|level| level.k);
    Ok(out)
}

/// Parse satu entri `levels`.
fn parse_level(value: &Value) -> Result<LevelStat, String> {
    let obj = value.as_object().ok_or("level harus objek")?;
    let k = obj
        .get("k")
        .and_then(Value::as_u64)
        .ok_or("level.k wajib")? as usize;
    Ok(LevelStat {
        k,
        ell: obj.get("ell").and_then(Value::as_u64).unwrap_or(0) as usize,
        recon: obj.get("recon").and_then(Value::as_f64).unwrap_or(0.0),
        loss: obj.get("loss").and_then(Value::as_f64).unwrap_or(0.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CURRENT: &str =
        r#"{"status":"ok","vocab_size":280,"roundtrip_ok":true,"build":"VIR Vbeta"}"#;
    const FULL: &str = r#"{
        "status": "ok", "build": "VIR Vbeta", "roundtrip_ok": true,
        "levels": [
            {"k": 1, "ell": 2, "recon": 0.10, "loss": 1.0},
            {"k": 2, "ell": 4, "recon": 0.55, "loss": 0.4}
        ],
        "recon_fraction": 0.75, "loss_before": 2.0, "loss_after": 0.4,
        "exact_match": true
    }"#;

    #[test]
    fn current_schema_parses_and_gate_ok() {
        let report = validate_smoke_json(CURRENT).expect("skema lama valid");
        assert_eq!(report.status, "ok");
        assert_eq!(report.roundtrip_ok, Some(true));
        assert!(report.gate_g1_ok().is_ok());
    }

    #[test]
    fn full_schema_gate_ok() {
        let report = validate_smoke_json(FULL).expect("skema penuh valid");
        assert_eq!(report.levels.len(), 2);
        assert_eq!(report.levels[1].k, 2);
        assert!(report.gate_g1_ok().is_ok());
    }

    #[test]
    fn missing_status_is_rejected() {
        assert!(validate_smoke_json("{}").is_err());
    }

    #[test]
    fn invalid_json_is_rejected() {
        assert!(validate_smoke_json("bukan json").is_err());
    }

    #[test]
    fn exact_match_false_fails_gate() {
        let report = SmokeReport {
            status: "ok".to_string(),
            exact_match: Some(false),
            ..Default::default()
        };
        assert!(report.gate_g1_ok().is_err());
    }

    #[test]
    fn decreasing_recon_fails_gate() {
        let report = SmokeReport {
            status: "ok".to_string(),
            levels: vec![
                LevelStat {
                    k: 1,
                    ell: 2,
                    recon: 0.6,
                    loss: 0.5,
                },
                LevelStat {
                    k: 2,
                    ell: 4,
                    recon: 0.3,
                    loss: 0.4,
                },
            ],
            ..Default::default()
        };
        assert!(report.gate_g1_ok().is_err());
    }
}
