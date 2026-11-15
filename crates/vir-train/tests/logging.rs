//! Tes nyata log JSONL per langkah (PRD F1.7 langkah 7): satu objek JSON valid
//! per baris, jumlah baris, append saat resume, dan `log_str`.

use vir_train::{EventRecord, JsonlLogger, LevelStat, StepRecord};

fn temp_dir(tag: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("vir_train_{tag}_{}_{nanos}", std::process::id()));
    path.to_str().expect("path utf8").to_string()
}

fn step_record(step: u64) -> StepRecord {
    StepRecord {
        stage: 1,
        step,
        stage_step: step,
        lr: 3e-4,
        loss_total: 1.25,
        per_level: vec![0.5, 0.75],
        level_tokens: vec![4.0, 8.0],
        utilization: vec![0.5, 0.75],
        perplexity: vec![2.0, 3.0],
        grad_norm: 0.9,
        corruption_rate: Some(0.25),
        recon: Some(0.8),
    }
}

#[test]
fn step_and_event_records_are_valid_json_lines() {
    let dir = temp_dir("jsonl");
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = format!("{dir}/steps.jsonl");
    {
        let mut log = JsonlLogger::new(&path).expect("open");
        log.log(&step_record(0)).expect("log step");
        log.log(&EventRecord::new("gate_passed", 1, 3, "lulus", Some(0.9)))
            .expect("log event");
        assert_eq!(log.lines(), 2);
        assert_eq!(log.path(), path.as_str());
        log.flush().expect("flush");
    }

    let text = std::fs::read_to_string(&path).expect("read");
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "berkas harus berisi dua baris");

    let step: serde_json::Value = serde_json::from_str(lines[0]).expect("json langkah");
    assert_eq!(step["stage"].as_u64(), Some(1));
    assert_eq!(step["step"].as_u64(), Some(0));
    assert!((step["loss_total"].as_f64().unwrap_or(0.0) - 1.25).abs() < 1e-6);
    assert_eq!(step["per_level"].as_array().map(|a| a.len()), Some(2));
    assert!(step["corruption_rate"].is_number());
    assert!(step["recon"].is_number());

    let event: serde_json::Value = serde_json::from_str(lines[1]).expect("json peristiwa");
    assert_eq!(event["event"].as_str(), Some("gate_passed"));
    assert_eq!(event["stage"].as_u64(), Some(1));
    assert_eq!(event["step"].as_u64(), Some(3));
    assert!((event["value"].as_f64().unwrap_or(0.0) - 0.9).abs() < 1e-6);

    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn append_keeps_previous_lines() {
    let dir = temp_dir("jsonl_append");
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = format!("{dir}/steps.jsonl");
    {
        let mut log = JsonlLogger::new(&path).expect("open");
        log.log(&step_record(0)).expect("log");
    }
    {
        let mut log = JsonlLogger::append(&path).expect("append");
        log.log(&EventRecord::new("summary", 2, 6, "selesai", None))
            .expect("log event");
        assert_eq!(log.lines(), 1, "penghitung append mulai dari nol");
    }

    let text = std::fs::read_to_string(&path).expect("read");
    assert_eq!(text.lines().count(), 2, "append menimpa baris lama");
    assert!(
        text.lines()
            .next()
            .expect("baris pertama")
            .contains("\"stage\":1")
    );
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn level_stat_json_line_parses() {
    let stat = LevelStat::from_indices(2, &[0, 1, 2, 3, 0], 8);
    let line = stat.to_json();
    let v: serde_json::Value = serde_json::from_str(&line).expect("json LevelStat");
    assert_eq!(v["level"].as_u64(), Some(2));
    assert_eq!(v["codes"].as_u64(), Some(8));
    assert_eq!(v["used"].as_u64(), Some(4));
    assert!((v["utilization"].as_f64().unwrap_or(0.0) - 0.5).abs() < 1e-4);
}
