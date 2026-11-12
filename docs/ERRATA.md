# ERRATA — VIR Vbeta

Format entri:
## E-XXX — judul singkat
- Gejala:
- Akar masalah:
- Perbaikan:
- Tes regresi:
- Commit:

## E-001 — (pantau) Rekonstruksi eksak
- Gejala: akurasi level penuh < 99.8% pada korpus tes.
- Akar masalah: (belum)
- Perbaikan: (belum)
- Tes regresi: regression_e001_*
- Commit: -

## E-002 — (pantau) Jumlah level tetap
- Gejala: halting tidak adaptif, biaya boros.
- Akar masalah: (belum)
- Perbaikan: (belum)
- Tes regresi: regression_e002_*
- Commit: -

## E-003 — (pantau) Ketidakcocokan latihan dan inferensi
- Gejala: mode latih (corruption) vs inferensi (bersih) tidak sinkron.
- Akar masalah: (belum)
- Perbaikan: (belum)
- Tes regresi: regression_e003_*
- Commit: -

## E-004 — (pantau) Utilisasi codebook rendah
- Gejala: utilisasi < 60% pada level tertentu.
- Akar masalah: (belum)
- Perbaikan: (belum)
- Tes regresi: regression_e004_*
- Commit: -

## E-005 — Koreksi PRD: SNAC untuk teks
- Gejala: PRD F1.4.b menyebut SNAC sebagai kandidat Down/R_k untuk teks.
- Akar masalah: salah tafsir; kode NCM memakai `F.interpolate(area/linear)` untuk teks, SNAC hanya audio.
- Perbaikan: PRD F1.4.b diperbarui; ADR-003 dicatat.
- Tes regresi: N/A (dokumentasi)
- Commit: a50dcbf