# ADR-002: Kuantizer (VQ-EMA vs FSQ vs iFSQ)

## Konteks
NCM memakai VQ-EMA (risiko codebook collapse, PRD risiko 12). FSQ/RFSQ/iFSQ
bebas collapse (tanpa codebook dipelajari, tanpa commitment loss). PRD F1.4.f
mensyaratkan ablasi VQ-EMA vs FSQ `[7,5,5,5,5]` vs iFSQ `[16,16,16]` pada
akurasi rekonstruksi eksak, utilisasi, dan stabilitas, lalu memilih default
Vbeta (kandidat utama: residual FSQ).

## Ablasi (mini, CPU, CI job `vir-vq` — `crates/vir-vq/tests/ablation.rs`)
Tokenizer mini (L=32, K=4, d=32, 8 sampel sintetis deterministik), 4 arm:

| Arm | Akurasi penuh | Level 1→K | Loss | Utilisasi | Perplexity | Stuck |
|---|---|---|---|---|---|---|
| vq-e32 | 0.6122 | 0.6663→0.6122 | 6.82 | 0.38 | 28.2 | 0 |
| vq-e4096 | 0.8379 | 0.7162→0.9786 | 5.28 | 0.47 | 285.9 | 0 |
| **fsq-7555** | **0.9708** | 0.9708→0.9708 | 4.49 | 0.97 | 445.2 | 0 |
| ifsq-16 | 0.6000 | 4.9075→46.7 | 7.31 | 0.60 | 185.4 | 0 |

**Temuan (jujur, skala mini):**
- **FSQ `[7,5,5,5,5]` unggul** pada akurasi rekonstruksi eksak (0.9708) dan
  stabilitas loss (4.49, turun), dengan utilisasi tinggi (0.97) — konsisten
  dengan PRD (FSQ bebas collapse, tanpa commitment loss).
- VQ-EMA 4096 (0.8379) di bawah FSQ di skala mini ini; VQ-EMA 32 (codebook
  sangat kecil) jauh lebih buruk (0.6122).
- iFSQ `[16,16,16]` di skala mini ini tidak stabil (loss naik tajam 4.9→46.7);
  hasil makalah iFSQ berasal dari gambar, bukan teks mini.

## Keputusan
- **Default Vbeta: residual FSQ `[7,5,5,5,5]`** (≈2¹² = 4375, dekat |V_k|=4096
  NCM). Alasan: akurasi rekonstruksi eksak tertinggi & stabil di ablasi mini,
  bebas collapse, tanpa commitment loss (PRD F1.4.f).
- VQ-EMA tetap sebagai **baseline** (match paper NCM) di belakang trait
  `Quantizer` yang sama, untuk verifikasi skala lebih besar.
- iFSQ `[16,16,16]` dan RFSQ (skala+LayerNorm) tetap sebagai kandidat; butuh
  verifikasi skala lebih besar (Kaggle) sebelum dipromosikan.

## Batas & verifikasi lanjutan
- Ablasi ini di **skala mini** (L=32, K=4, d=32). Hasil iFSQ negatif di sini
  belum tentu berlaku di skala NCM (L=128, K=12, d=384). Rencana: jalankan
  ablasi yang sama di Kaggle (skala NCM mini) dan perbarui ADR ini dengan angka
  skala lebih besar sebelum mengunci default produksi.
- Angka di tabel adalah hasil nyata CI (job `vir-vq`, test `ablation`), bukan
  konstanta.

## Status
**Final untuk Vbeta mini** (default FSQ `[7,5,5,5,5]`). Verifikasi skala besar
(Kaggle) tercatat sebagai tindak lanjut F1.4.f.