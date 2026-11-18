# ADR-010: Pengukuran throughput latihan yang adil & profil fase

## Konteks
 Klaim "performa superior vs PyTorch" (PRD R1/R10) butuh **pengukuran adil**.
Misur pertama (vir-kaggle-vs-pytorch v1) menampilkan **~350× lebih lambat**
dari PyTorch — ini **artefak**: (a) VIR diukur sekali jalan **tanpa warmup**,
(b) wall-clock VIR termasuk **kompilasi/autotune kernel langkah-1** (~146s),
(c) sedangkan PyTorch sudah di-warmup 3 langkah. Tanpa koreksi ini, keputusan
optimisasi jadi menebak-nebak.

## Keputusan
1. **Ukur steady-state, bukan total-time.** Tambah `--warmup N` pada
   `NativeTrainer` (W0): N langkah forward/backward/step yang memicu kompilasi +
   autotune, **tidak** dihitung/log. `tokens_per_sec()` dihitung sejak akhir
   warmup. `StepRecord.dt_ms` mencatat durasi per langkah; `RunOutcome.tok_s`
   melaporkan throughput steady-state di `--json`.
2. **Profil fase (R1: ukur dulu).** `train_step` mengukur terpisah
   `fwd_ms` / `bwd_ms` / `opt_ms` (EWMA), dilaporkan di `--json`. Ini menunjukkan
   **di mana** waktu per langkah dipakai sebelum menulis optimisasi.
3. **Bandingkan presisi sama** (PRD:636). Catat bahwa fp32 CubeK memakai
   **TF32 default** (`cubek.matmul.allow_tf32 = true`), sama seperti PyTorch
   fp32+TF32 — jadi perbandingan fp32 adil.
4. **Pisahkan cold-start dari steady-state** dalam analisis (cold start = had
   kompilasi sekali, bukan throughput).

## Hasil (Kaggle 2×T4, mini dim64 K5 L32, batch 64, L=32, warmup 5)

| Metrik | vir train --native fp32 | PyTorch setara fp32(TF32) |
|---|---|---|
| tok/s (steady-state) | **7 936 – 10 580** | **136 533** |
| waktu/langkah | 0.10–0.26 s | ~0.015 s |
| cold start (kompilasi) | ~146 s (sekali) | ~0 |

**Profil fase vir (EWMA, ms/langkah):**

| Fase | ms | % |
|---|---|---|
| forward | 50.2 | 27.3% |
| backward | 90.0 | 48.9% |
| optimizer (Muon+AdamW) | 43.9 | 23.8% |
| **total** | **184.1** | 100% |

## Temuan (jujur)
- **Gap nyata ≈ 12–17×** (bukan 350×). Setelah W0 (warmup) + W1 (buang 16 sync
  metrik/langkah) + B6 (RoPE sekali/blok), artefak 350× → 17× → ~(10580 vs
  136533) 13×.
- **Bottleneck = overhead per-op / dispatch**, bukan kernel VQ. Bandingkan:
  ADR-001 (kernel VQ murni) = **0.75× PyTorch** (gate G2 ≥0.5× terpenuhi).
  Full training loop jauh lebih lambat karena **ribuan op kecil** (30 blok
  transformer = 5 level × 6 blok, + VQ + Muon ~700 matmul) → tiap op kena
  overhead luncuran/dispatch Burn yang jauh di atas PyTorch eager.
- **TF32 sudah aktif** (default) → bukan sumber gap.
- **Muon** (Newton-Schulz 5 iter × ~48 param 2D ≈ 700 matmul) = 24% waktu,
  overhead struktural vs AdamW PyTorch (impl. optimizer alternatif, bukan
  sekadar tuning).
- **Backward (49%)** = turunan dari op-count forward; mengecilkan op forward
  (B4) mengecilkan backward secara proporsional.

## Rencana lanjut (high-effort/risk, butuh buy-in)
- **B4 — batch-level forward**: 5 level → 1 forward `[B·K, L, d]` + mask
  antar-level (30 blok → 6 blok). Potensi ~2–3× bila benar-benar launch-bound;
  butuh ekuivalensi numerik & hati-hati (recurrent depth, teacher forcing).
- **W4 — CUDA graph capture** (`burn-capture`) untuk forward: hilangkan overhead
  dispatch; **Evaluasi** (Burn 0.22 capture_inference-oriented, training input
  variabel).
- **Muon**: tak mengubah `ns_steps` (kualitas, PRD:604); opsi: bf16 matmul NS,
  atau fused kernel NS (CubeCL) — besar.
- **Autotune cache persisten** (B3): cold start ~146s → ~0 (total-time & etika
  perbandingan); API cubecl 0.11.4 belum terverifikasi.

## Status
**Metodologi: final & dipakai** (W0/W1/B6/profil). **Optimisasi penuh: berjalan**
— gap 12–17× terdokumentasi; B4/W4/Muon adalah kandidat berikutnya yang
high-risk. Tidak ada klaim ">PyTorch" tanpa ukur.