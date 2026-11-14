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

## E-006 — kda_step kehilangan suku rank-1
- Gejala: `kda_step` lama hanya menerapkan diagonal `I − βkkᵀ`, kehilangan `−βk(α⊙k)ᵀS`.
- Akar masalah: implementasi tidak sesuai rumus PRD F5.2.
- Perbaikan: `kda_step` diperbaiki + `kda_step_chunkwise` (UT) + tes kesetaraan.
- Tes regresi: `kda_chunkwise_matches_naive`
- Commit: (nn-eng)

## E-007 — fallback fused_argmin_bias mengabaikan bias
- Gejala: fallback `argmax(z·eᵀ)` bukan `argmax(z·eᵀ − ½‖e‖²)`.
- Akar masalah: bias tidak ditambahkan di jalur fallback.
- Perbaikan: `+ bias.reshape([1, codes])` + validasi bentar/device.
- Tes regresi: `fallback_applies_bias`
- Commit: (nn-eng)