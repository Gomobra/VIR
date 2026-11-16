//! Verifier: skor kandidat, seleksi saat tes, dan pagar eksekusi (F4.2, F4.4).
//!
//! [`Verifier`] adalah satu-satunya antarmuka penilaian kandidat: skor
//! `0.0..1.0` dengan `1.0` berarti lulus/verified. [`select_candidate`] memilih
//! kandidat berskor tertinggi secara deterministik (seri dimenangkan kandidat
//! yang lebih awal), dan [`guard`] adalah pagar eksekusi: mencoba kandidat
//! berikutnya saat yang sekarang gagal lalu melaporkan apakah pagar menyelamatkan
//! hasil.
//!
//! Implementasi nyata ada di [`crate::code_verifier::CodeVerifier`] (unit test di
//! sandbox); [`ScriptedVerifier`] dan [`FnVerifier`] dipakai untuk tes
//! deterministik dan kebijakan skor kustom.

use vir_common::VirError;

/// Ambang skor yang dianggap lulus. Verifier kode bersifat biner, jadi hanya
/// skor penuh yang lolos; verifier berkelanjutan harus memetakan skornya ke
/// `{0.0, 1.0}` sendiri agar pagar eksekusi tetap memakai satu ambang.
pub const VERIFIED: f32 = 1.0;

/// Penilai kandidat: skor `candidate` pada `task`.
pub trait Verifier {
    /// Skor `0.0..1.0` untuk `candidate` pada `task`; `1.0` berarti lulus.
    fn score(&self, task: &str, candidate: &str) -> Result<f32, VirError>;
}

/// Verifier berbasis closure; pembungkus tipis untuk kebijakan skor kustom.
pub struct FnVerifier<F> {
    f: F,
}

impl<F> FnVerifier<F>
where
    F: Fn(&str, &str) -> f32,
{
    /// Bungkus closure `(task, kandidat) -> skor`.
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F> Verifier for FnVerifier<F>
where
    F: Fn(&str, &str) -> f32,
{
    fn score(&self, task: &str, candidate: &str) -> Result<f32, VirError> {
        Ok((self.f)(task, candidate))
    }
}

/// Verifier deterministik: peta `kandidat -> skor` untuk tes (F4.2).
///
/// Peta, bukan urutan pemanggilan, sehingga skor tidak bergantung pada berapa
/// kali verifier dipanggil: tes dapat memanggil `score` dua kali dan mendapat
/// nilai yang sama.
#[derive(Debug, Clone)]
pub struct ScriptedVerifier {
    scores: Vec<(String, f32)>,
    default: f32,
}

impl ScriptedVerifier {
    /// Peta skor dari pasangan `(kandidat, skor)`.
    pub fn new<S: AsRef<str>>(scores: &[(S, f32)]) -> Self {
        Self {
            scores: scores
                .iter()
                .map(|(kandidat, skor)| (kandidat.as_ref().to_owned(), *skor))
                .collect(),
            default: 0.0,
        }
    }

    /// Skor untuk kandidat yang tidak ada di daftar (bawaan `0.0`).
    pub fn with_default(mut self, default: f32) -> Self {
        self.default = default;
        self
    }
}

impl Verifier for ScriptedVerifier {
    fn score(&self, _task: &str, candidate: &str) -> Result<f32, VirError> {
        let skor = self
            .scores
            .iter()
            .find(|(kandidat, _)| kandidat == candidate)
            .map(|(_, skor)| *skor);
        Ok(skor.unwrap_or(self.default))
    }
}

/// Hasil seleksi satu kandidat; skor disimpan sebagai data RL (F4.2 butir 4).
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    /// Posisi kandidat pada daftar masukan.
    pub index: usize,
    pub candidate: String,
    pub score: f32,
}

/// Pilih kandidat berskor tertinggi; seri dimenangkan kandidat pertama.
///
/// Galat bila daftar kosong: seleksi tanpa kandidat bukan "tidak ada yang
/// lulus", melainkan pemakaian yang salah.
pub fn select_candidate<V, S>(
    verifier: &V,
    task: &str,
    candidates: &[S],
) -> Result<Selection, VirError>
where
    V: Verifier + ?Sized,
    S: AsRef<str>,
{
    let mut best: Option<Selection> = None;
    for (index, candidate) in candidates.iter().enumerate() {
        let candidate = candidate.as_ref();
        let score = verifier.score(task, candidate)?;
        if best.as_ref().is_none_or(|picked| score > picked.score) {
            best = Some(Selection {
                index,
                candidate: candidate.to_owned(),
                score,
            });
        }
    }
    let Some(picked) = best else {
        return Err(VirError::Config(
            "seleksi butuh minimal satu kandidat".into(),
        ));
    };
    Ok(picked)
}

/// Hasil pagar eksekusi (F4.4 butir 3 dan 4).
#[derive(Debug, Clone, PartialEq)]
pub struct GuardReport {
    /// Kandidat yang lulus; `None` bila anggaran habis tanpa kandidat lulus.
    pub selected: Option<String>,
    /// Skor kandidat terakhir yang diuji.
    pub score: f32,
    /// Jumlah kandidat yang diuji, termasuk yang gagal.
    pub attempts: usize,
    /// `true` bila kandidat pertama gagal tetapi kandidat berikutnya lulus:
    /// inilah "pagar menyelamatkan hasil" yang diukur F4.4.
    pub rescued: bool,
    /// Skor kandidat pertama; pembanding untuk `rescued`.
    pub first_score: f32,
}

impl GuardReport {
    /// Laporan awal: belum ada kandidat yang diuji.
    fn kosong() -> Self {
        Self {
            selected: None,
            score: 0.0,
            attempts: 0,
            rescued: false,
            first_score: 0.0,
        }
    }
}

/// Coba kandidat dari `next(attempt)` sampai ada yang lulus atau anggaran habis.
///
/// `next` menerima indeks percobaan (0 = kandidat pertama) sehingga pemanggil
/// dapat mengganti seed atau suhu; kegagalan model dikembalikan apa adanya.
pub fn guard<V, F>(
    verifier: &V,
    task: &str,
    budget: usize,
    mut next: F,
) -> Result<GuardReport, VirError>
where
    V: Verifier + ?Sized,
    F: FnMut(usize) -> Result<String, VirError>,
{
    if budget == 0 {
        return Err(VirError::Config(
            "anggaran pagar harus lebih dari nol".into(),
        ));
    }
    let mut report = GuardReport::kosong();
    for attempt in 0..budget {
        let candidate = next(attempt)?;
        report.score = verifier.score(task, &candidate)?;
        if attempt == 0 {
            report.first_score = report.score;
        }
        report.attempts = attempt + 1;
        if report.score >= VERIFIED {
            report.selected = Some(candidate);
            report.rescued = attempt > 0;
            return Ok(report);
        }
    }
    Ok(report)
}
