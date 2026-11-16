//! Tes verifier: seleksi saat tes (F4.2), pagar eksekusi (F4.4), dan verifier
//! kode yang berjalan di sandbox (`vir_sandbox::run`).
//!
//! Tes verifier kode memakai kebijakan tanpa konfinemen kernel (hanya rlimit +
//! timeout dinding) supaya jalan di host mana pun; isolasi Landlock/seccomp
//! adalah wilayah job `vir-sandbox`. Bila `python3` tidak ada, tesnya mencetak
//! `SKIP:` dengan alasan — tidak lulus diam-diam.

use std::time::Duration;

use vir_agent::code_verifier::CodeVerifier;
use vir_agent::verifier::{
    FnVerifier, GuardReport, ScriptedVerifier, Verifier, guard, select_candidate,
};
use vir_sandbox::{Policy, run};

const KANDIDAT_OK: &str = "def add(a, b):\n    return a + b\n";
const KANDIDAT_SALAH: &str = "def add(a, b):\n    return a - b\n";
const TES_OK: &str = r#"import unittest

from candidate import add


class AddTest(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_add_ganjil(self):
        self.assertEqual(add(-1, 1), 0)
"#;
/// Batas waktu wajar untuk tes fungsional.
const TIMEOUT_UJI: u64 = 10_000;
/// Batas waktu pendek untuk membuktikan timeout dinding.
const TIMEOUT_PENDEK: u64 = 700;

/// Kebijakan tanpa konfinemen kernel; lihat catatan di header berkas.
fn policy_fungsional(timeout_ms: u64) -> Policy {
    Policy {
        landlock: false,
        seccomp: false,
        timeout: Duration::from_millis(timeout_ms),
        ..Policy::default()
    }
}

/// Verifier kode untuk tes fungsional; `tests` = sumber modul `unittest`.
fn verifier_fungsional(tests: Option<&str>, timeout_ms: u64) -> CodeVerifier {
    let dasar = CodeVerifier::new().with_policy(policy_fungsional(timeout_ms));
    match tests {
        Some(source) => dasar.with_tests(source),
        None => dasar,
    }
}

/// `true` bila `python3` dapat dijalankan di host ini.
fn python3_tersedia() -> bool {
    let policy = policy_fungsional(TIMEOUT_UJI);
    match run("python3", &["-c", "pass"], &policy) {
        Ok(out) => out.exit_code == Some(0),
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// ScriptedVerifier dan FnVerifier
// ---------------------------------------------------------------------------

#[test]
fn scripted_verifier_skor_deterministik() {
    let verifier = ScriptedVerifier::new(&[("kandidat-a", 0.0), ("kandidat-b", 1.0)]);
    let skor_b = verifier.score("tugas", "kandidat-b");
    assert_eq!(skor_b.expect("skor"), 1.0);
    let skor_a = verifier.score("tugas", "kandidat-a");
    assert_eq!(skor_a.expect("skor"), 0.0);
    // Peta, bukan urutan pemanggilan: panggilan kedua memberi nilai sama.
    let ulang = verifier.score("tugas", "kandidat-b");
    assert_eq!(ulang.expect("skor"), 1.0);
}

#[test]
fn scripted_verifier_default_untuk_kandidat_tak_dikenal() {
    let dasar = ScriptedVerifier::new(&[("kandidat-a", 1.0)]);
    let verifier = dasar.clone().with_default(0.25);
    let skor = verifier.score("tugas", "lain");
    assert_eq!(skor.expect("skor"), 0.25);
    let tanpa_default = dasar.score("tugas", "lain");
    assert_eq!(tanpa_default.expect("skor"), 0.0);
}

#[test]
fn fn_verifier_memakai_predikat_kustom() {
    let verifier = FnVerifier::new(|task: &str, candidate: &str| {
        if task == "tambah" && candidate.contains("return a + b") {
            1.0
        } else {
            0.0
        }
    });
    let skor = verifier.score("tambah", KANDIDAT_OK);
    assert_eq!(skor.expect("skor"), 1.0);
    let skor = verifier.score("tambah", KANDIDAT_SALAH);
    assert_eq!(skor.expect("skor"), 0.0);
}

// ---------------------------------------------------------------------------
// select_candidate
// ---------------------------------------------------------------------------

#[test]
fn select_candidate_memilih_skor_tertinggi() {
    let verifier = ScriptedVerifier::new(&[("buruk", 0.0), ("sedang", 0.5), ("baik", 1.0)]);
    let kandidat = vec!["buruk", "sedang", "baik"];
    let hasil = select_candidate(&verifier, "tugas", &kandidat);
    let picked = hasil.expect("seleksi");
    assert_eq!(picked.index, 2);
    assert_eq!(picked.candidate, "baik");
    assert_eq!(picked.score, 1.0);
}

#[test]
fn select_candidate_seri_memilih_kandidat_pertama() {
    let verifier = ScriptedVerifier::new(&[("pertama", 0.5), ("kedua", 0.5)]);
    let kandidat = vec!["pertama", "kedua"];
    let hasil = select_candidate(&verifier, "tugas", &kandidat);
    let picked = hasil.expect("seleksi");
    assert_eq!(picked.index, 0, "seri: kandidat pertama");
    assert_eq!(picked.candidate, "pertama");
}

#[test]
fn select_candidate_kosong_adalah_galat() {
    let verifier = ScriptedVerifier::new(&[("baik", 1.0)]);
    let kandidat: Vec<&str> = Vec::new();
    let hasil = select_candidate(&verifier, "tugas", &kandidat);
    let err = hasil.expect_err("daftar kosong");
    assert!(err.to_string().contains("minimal satu kandidat"));
}

// ---------------------------------------------------------------------------
// guard (pagar eksekusi, F4.4)
// ---------------------------------------------------------------------------

#[test]
fn guard_kandidat_pertama_lulus_tidak_menyelamatkan() {
    let verifier = ScriptedVerifier::new(&[("baik", 1.0)]);
    let hasil = guard(&verifier, "tugas", 4, |_| Ok("baik".to_owned()));
    let report = hasil.expect("guard");
    assert_eq!(report.attempts, 1);
    assert_eq!(report.selected.as_deref(), Some("baik"));
    assert!(!report.rescued, "kandidat pertama lulus");
    assert_eq!(report.score, 1.0);
}

#[test]
fn guard_pagar_menyelamatkan_hasil() {
    let verifier = ScriptedVerifier::new(&[("buruk", 0.0), ("baik", 1.0)]);
    let kandidat = ["buruk", "baik"];
    let hasil = guard(&verifier, "tugas", 4, |i| Ok(kandidat[i].to_owned()));
    let report = hasil.expect("guard");
    assert_eq!(report.attempts, 2);
    assert!(report.rescued, "percobaan kedua lulus: pagar menyelamatkan");
    assert_eq!(report.selected.as_deref(), Some("baik"));
    assert_eq!(report.first_score, 0.0);
    assert_eq!(report.score, 1.0);
}

#[test]
fn guard_budget_habis_tanpa_kandidat_lulus() {
    let verifier = ScriptedVerifier::new(&[("buruk", 0.0)]);
    let hasil = guard(&verifier, "tugas", 3, |i| Ok(format!("kandidat-{i}")));
    let report = hasil.expect("guard");
    assert_eq!(report.attempts, 3);
    assert!(report.selected.is_none());
    assert!(!report.rescued);
    assert_eq!(report.score, 0.0);
}

#[test]
fn guard_anggaran_nol_adalah_galat() {
    let verifier = ScriptedVerifier::new(&[("baik", 1.0)]);
    let hasil = guard(&verifier, "tugas", 0, |_| Ok("baik".to_owned()));
    let err = hasil.expect_err("anggaran nol");
    assert!(err.to_string().contains("anggaran"));
}

/// Gabungan yang dipakai D4: dua kandidat gagal, yang ketiga lulus.
#[test]
fn guard_memilih_dari_kandidat_tak_lulus_lebih_dulu() {
    let verifier = ScriptedVerifier::new(&[("k1", 0.0), ("k2", 0.0), ("k3", 1.0)]);
    let kandidat = ["k1", "k2", "k3"];
    let hasil = guard(&verifier, "tugas", 3, |i| Ok(kandidat[i].to_owned()));
    let report = hasil.expect("guard");
    let harap = GuardReport {
        selected: Some("k3".to_owned()),
        score: 1.0,
        attempts: 3,
        rescued: true,
        first_score: 0.0,
    };
    assert_eq!(report, harap);
}

// ---------------------------------------------------------------------------
// CodeVerifier (sandbox)
// ---------------------------------------------------------------------------

/// Kebijakan bawaan harus menyalakan konfinemen kernel: verifier yang diam-diam
/// menjalankan kode tanpa Landlock/seccomp adalah lubang keamanan (F4.4 pagar).
#[test]
fn code_verifier_bawaan_menyalakan_konfinemen_kernel() {
    let verifier = CodeVerifier::new();
    let policy = verifier.policy();
    assert!(policy.landlock, "Landlock wajib aktif secara bawaan");
    assert!(
        policy.seccomp,
        "allowlist syscall wajib aktif secara bawaan"
    );
    assert!(policy.workdir.is_none(), "workdir disetel per kandidat");
}

#[test]
fn code_verifier_kandidat_lulus_unit_test() {
    if !python3_tersedia() {
        eprintln!("SKIP: python3 tidak dapat dijalankan di host ini");
        return;
    }
    let verifier = verifier_fungsional(Some(TES_OK), TIMEOUT_UJI);
    let hasil = verifier.run("tugas", KANDIDAT_OK);
    let report = hasil.expect("jalan");
    let stderr = report.stderr.clone();
    assert_eq!(report.tests_run, Some(2), "galat: {stderr:?}");
    assert!(report.passed, "kandidat harus lulus semua tes");
    let skor = verifier.score("tugas", KANDIDAT_OK);
    assert_eq!(skor.expect("skor"), 1.0);
}

#[test]
fn code_verifier_unit_test_gagal_memberi_skor_nol() {
    if !python3_tersedia() {
        eprintln!("SKIP: python3 tidak dapat dijalankan di host ini");
        return;
    }
    let verifier = verifier_fungsional(Some(TES_OK), TIMEOUT_UJI);
    let hasil = verifier.run("tugas", KANDIDAT_SALAH);
    let report = hasil.expect("jalan");
    assert_eq!(report.tests_run, Some(2));
    assert!(!report.passed);
    assert_eq!(report.exit_code, Some(1), "unittest gagal harus keluar 1");
    let skor = verifier.score("tugas", KANDIDAT_SALAH);
    assert_eq!(skor.expect("skor"), 0.0);
}

#[test]
fn code_verifier_tanpa_tes_memakai_eksekusi_skrip() {
    if !python3_tersedia() {
        eprintln!("SKIP: python3 tidak dapat dijalankan di host ini");
        return;
    }
    let verifier = verifier_fungsional(None, TIMEOUT_UJI);
    let lulus = "assert 1 + 1 == 2\nprint('ok')\n";
    let gagal = "assert 1 + 1 == 3\n";
    let skor = verifier.score("tugas", lulus);
    assert_eq!(skor.expect("skor"), 1.0);
    let skor = verifier.score("tugas", gagal);
    assert_eq!(skor.expect("skor"), 0.0);
}

#[test]
fn code_verifier_kandidat_gagal_parse_skor_nol() {
    if !python3_tersedia() {
        eprintln!("SKIP: python3 tidak dapat dijalankan di host ini");
        return;
    }
    let verifier = verifier_fungsional(Some(TES_OK), TIMEOUT_UJI);
    let kandidat = "def add(a, b)\n    return a + b\n";
    let hasil = verifier.run("tugas", kandidat);
    let report = hasil.expect("jalan");
    assert!(!report.passed);
    assert_eq!(report.tests_run, None, "kandidat gagal diimpor");
    let stderr = report.stderr.clone();
    assert!(stderr.contains("Error"), "galat impor: {stderr:?}");
}

/// Kumpulan tes tanpa metode tes apa pun tidak boleh memberi skor 1.0
/// (reward hacking lewat tes kosong, PRD F4.3 butir 5).
#[test]
fn code_verifier_tes_kosong_ditolak() {
    if !python3_tersedia() {
        eprintln!("SKIP: python3 tidak dapat dijalankan di host ini");
        return;
    }
    let verifier = verifier_fungsional(Some("import unittest\n"), TIMEOUT_UJI);
    let hasil = verifier.run("tugas", KANDIDAT_OK);
    let report = hasil.expect("jalan");
    assert_eq!(report.tests_run, Some(0), "unittest melaporkan 0 tes");
    assert!(!report.passed, "tes kosong harus ditolak");
    let skor = verifier.score("tugas", KANDIDAT_OK);
    assert_eq!(skor.expect("skor"), 0.0);
}

/// Timeout dinding sandbox berlaku untuk kandidat yang berputar tak hingga.
#[test]
fn code_verifier_timeout_didatangkan_sandbox() {
    if !python3_tersedia() {
        eprintln!("SKIP: python3 tidak dapat dijalankan di host ini");
        return;
    }
    let verifier = verifier_fungsional(None, TIMEOUT_PENDEK);
    let hasil = verifier.run("tugas", "while True:\n    pass\n");
    let report = hasil.expect("jalan");
    assert!(report.timed_out, "loop tak hingga harus timeout");
    assert_eq!(report.exit_code, None, "proses dibunuh");
    let skor = verifier.score("tugas", "while True:\n    pass\n");
    assert_eq!(skor.expect("skor"), 0.0);
}

/// Seleksi ujung ke ujung: dari tiga kandidat, hanya satu yang lulus unit test.
#[test]
fn select_candidate_lewat_code_verifier_di_sandbox() {
    if !python3_tersedia() {
        eprintln!("SKIP: python3 tidak dapat dijalankan di host ini");
        return;
    }
    let verifier = verifier_fungsional(Some(TES_OK), TIMEOUT_UJI);
    let kandidat = vec![
        KANDIDAT_SALAH.to_owned(),
        KANDIDAT_OK.to_owned(),
        "def add(a, b):\n    return 0\n".to_owned(),
    ];
    let hasil = select_candidate(&verifier, "tugas", &kandidat);
    let picked = hasil.expect("seleksi");
    assert_eq!(picked.index, 1);
    assert_eq!(picked.score, 1.0);
    assert_eq!(picked.candidate, KANDIDAT_OK);
}

/// Verifier harus menolak interpreter yang tidak ada, bukan mengaku lulus.
#[test]
fn code_verifier_interpreter_mising_adalah_galat() {
    let verifier = CodeVerifier::new().with_interpreter("vir-tak-ada");
    let hasil = verifier.score("tugas", KANDIDAT_OK);
    let err = hasil.expect_err("interpreter tidak ada");
    assert!(!err.to_string().is_empty());
}
