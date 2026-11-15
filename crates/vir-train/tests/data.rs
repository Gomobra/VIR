//! Tes nyata `BatchSource` (PRD F2.5.a): deterministik pada seed sama dan
//! resume-safe lewat state/`seek_step`.

use vir_common::VirError;
use vir_data::write_shard;
use vir_train::data::{BatchSource, ShardBatchSource};

/// Token sintetis deterministik (nilai < 32).
fn tokens() -> Vec<u32> {
    (0..32u32).map(|i| (i * 7 + 3) % 32).collect()
}

fn temp_shard(tag: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!(
        "vir_train_{tag}_{}_{nanos}.shard",
        std::process::id()
    ));
    path.to_str().expect("path utf8").to_string()
}

fn source(path: &str, batch: usize, len: usize, seed: u64) -> ShardBatchSource {
    ShardBatchSource::from_shard(path, batch, len, seed, true).expect("buka shard")
}

fn take(src: &mut ShardBatchSource, n: usize) -> Vec<Vec<i64>> {
    (0..n).map(|_| src.next_batch().expect("batch").0).collect()
}

#[test]
fn batch_bentuk_dan_rentang() {
    let path = temp_shard("data_shape");
    write_shard(&path, &tokens()).expect("tulis shard");
    let mut src = source(&path, 4, 4, 7);
    let (flat, shape) = src.next_batch().expect("batch");
    assert_eq!(shape, [4, 4]);
    assert_eq!(flat.len(), 16);
    assert!(flat.iter().all(|t| (0..32).contains(t)));
    assert_eq!(src.produced(), 1);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn batch_deterministik_seed_sama() {
    let path = temp_shard("data_det");
    write_shard(&path, &tokens()).expect("tulis shard");
    let mut a = source(&path, 3, 4, 11);
    let mut b = source(&path, 3, 4, 11);
    for _ in 0..5 {
        assert_eq!(
            a.next_batch().expect("batch"),
            b.next_batch().expect("batch")
        );
    }
    // Seed berbeda → urutan blok berbeda (shuffle aktif).
    let mut c = source(&path, 3, 4, 12);
    assert_ne!(take(&mut a, 3), take(&mut c, 3));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn resume_state_melanjutkan_urutan_yang_sama() {
    let path = temp_shard("data_resume");
    write_shard(&path, &tokens()).expect("tulis shard");
    let mut full = source(&path, 2, 4, 5);
    let head = take(&mut full, 3);
    let state = full.state();
    let tail = take(&mut full, 3);

    let mut resumed = source(&path, 2, 4, 5);
    assert!(resumed.seek(state));
    assert_eq!(tail, take(&mut resumed, 3));

    let mut all = source(&path, 2, 4, 5);
    assert_eq!(head, take(&mut all, 3));
    assert_eq!(tail, take(&mut all, 3));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn seek_step_menyamakan_posisi_batch() {
    let path = temp_shard("data_seek");
    write_shard(&path, &tokens()).expect("tulis shard");
    let mut a = source(&path, 2, 4, 3);
    let _ = take(&mut a, 4);
    let tail_a = take(&mut a, 2);

    let mut b = source(&path, 2, 4, 3);
    b.seek_step(4).expect("seek");
    assert_eq!(tail_a, take(&mut b, 2));
    assert_eq!(b.produced(), 6);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn sumber_tanpa_blok_ditolak() {
    let path = temp_shard("data_empty");
    write_shard(&path, &[]).expect("tulis shard kosong");
    let mut src = source(&path, 4, 4, 1);
    let err: VirError = src.next_batch().expect_err("harus gagal");
    assert!(matches!(err, VirError::Internal(_)));
    let _ = std::fs::remove_file(&path);
}
