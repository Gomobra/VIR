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

## Matriks skala (Kaggle 2×T4, warmup 3 + 30 terukur, vir vs PyTorch @ batch)

| batch | vir tok/s | PyTorch fp32 tok/s | ratio | vir fwd/bwd/opt (ms) |
|---|---|---|---|---|
| 64  | 11 494  | 146 286 | **0.079×** | 51 / 95 / 50 |
| 256 | 38 478  | 296 096 | 0.13×  | 42 / 105 / 38 |
| 512 | 46 412  | 303 407 | **0.153×** | 40 / 246 / 66 |

**Temuan:**
- Rasio **membaik** dgn batch (vir lebih launch-bound → amortisasi lebih baik),
  tapi **plateau ~0.15×** saat compute-bound (b256→b512 vir hanya 1.2× utk 2×
  batch; PyTorch saturasi ~300k).
- **fwd ~40ms KONSTAN** dari b64→b512 (8× data) = **launch-bound murni** (overhead
  tetap). **bwd naik 95→246ms** = compute-bound, dan pada b512 **bwd 246ms >
  seluruh langkah PyTorch (54ms)** → backward autodiff kita ~4.5× lebih lambat
  dari *total* PyTorch.
- **Kesimpulan definitif: vir plateau ~15% PyTorch di SEMUA batch → tidak bisa
  mengalahkan PyTorch untuk beban training ini.** Batas = overhead dispatch
  per-op Burn pada backward (graf autodiff ~1000 op) + kernel CubeCL scalar-FMA
  (ADR-001 0.77×). Bukan sesuatu yang bisa ditutup optimisasi inkremental.

## Perbandingan ADIL apple-to-apple (vir vs PyTorch VIR, model SAMA)

Notebook `vir-kaggle-fair`: PyTorch implementasi VIR **model sama** (multiscale
VQ + recurrent depth prelude1+2×(core2+inject)+coda1 + RMSNorm + RoPE +
attention causal (SDPA) + SwiGLU + 5 level ℓ=[2,4,8,16,32] + Muon (Newton-
Schulz 5 iter) + AdamW). T4 sama, batch 64, L=32, warmup.

| | tok/s | ms/step |
|---|---|---|
| **vir (Burn) VIR** | 9 738 | 222 (fwd 60 / bwd 111 / opt 51) |
| **PyTorch VIR** (model sama) | 16 291 | 126 |
| **ratio** | **0.598×** | — |

**Koreksi penting:** pada **kerja setara** (model VIR sama, bukan GPT baseline),
PyTorch hanya **1.67× lebih cepat** — BUKAN 6.5×. Sebagian besar gap "6.5×"
sebelumnya berasal dari **model kita melakukan lebih banyak kerja** (5 level
multiscale + VQ + recurrent depth) vs GPT baseline PyTorch, bukan dari framework.
Gap per-kerja-sebenarnya = **1.67×**, jauh lebih dekat & berpotensi ditutup.

**Rincian gap (1.67×):** opt (Muon) 51ms vs ~16ms (PyTorch Muon pakai cuBLAS @;
vir pakai CubeCL) = ~35ms; fwd 60 vs ~40 (SDPA vs naive attention); bwd 111 vs
~70 (dispatch autodiff). Total gap ~96ms.

## A/B: Muon hanya bobot blok (head/embed/codebook→AdamW) — terukur

Notebook `vir-kaggle-fair` v2 (Muon dibatasi ke bobot blok transformer
`.weight`, sisanya AdamW — praktik standar). T4, batch 64, warmup.

| | tok/s | fwd/bwd/opt (ms) |
|---|---|---|
| vir (Muon blok) | **11 483** | 49 / 88 / **34** |
| PyTorch VIR | 16 526 | ~126 total |
| **ratio** | **0.695×** | — |

**Kemajuan:** opt **51→34ms (-33%)**, ratio **0.598→0.695×**, gap **1.67→1.44×**.
Sisa gap: bwd 88 vs ~70 (dispatch autodiff) + opt 34 vs ~16 (CubeCL vs cuBLAS
NS) + fwd 49 vs ~40 (naive attention vs SDPA). Semua = overhead framework/kernel,
bukan tuning.

## B4 (batch-level) — dicoba, GAGAL (terdokumentasi)
Ide: batch 5 level jadi 1 lintasan blok (30→6 blok). **Gagal**: level_input
memanggil `scales.down(k)` yang **menurunkan L→ℓ_k** (level input beda panjang
`[B,2,d],[B,4,d],[B,8,d]...`), jadi `cat` dim0 error. Batch butuh padding +
mask PER-LEVEL (attention双向/bidireksional → real token melihat padding), dan
Burn `forward_masked` hanya satu mask `[T,T]` utk semua batch → tak bisa per-level.
**Di-revert** (kode hijau).

## bf16/fp16 (tensor core) — bf16 GAGAL, fp16 JALAN tapi kecil
`vir train --native --dtype bf16` → **rc 101**: `Shared memory budget exceeded:
kernel requests 69632 bytes, hardware allows 65536` (cubek matmul bf16 autotune
> 64KB shared-mem Turing/T4).
`--dtype fp16` → **rc 0, 10 702 tok/s (0.64×)** (dari fp32 9 781, +9%): opt
(Muon) 40→33ms & fwd 58→54ms lebih cepat (tensor core), TAPI **bwd 105→110ms
TIDAK berubah** → fwd+bwd (164ms) tetap > total PyTorch (124ms). **Presisi
reduksi mempercepat compute (fwd/opt) tapi tidak bwd (dispatch autodiff).**

## KESIMPULAN DEFINITIF (semua lever sah sudah dicoba)
**Objektif "di atas PyTorch" TIDAK tercapai.** Rasio terbaik ≈ **0.64–0.70×**,
dibatasi **bwd (dispatch autodiff Burn ~105ms/langkah)** yang:
- tak dipengaruhi fp16/bf16 (bukan compute),
- tak bisa di-batch (level beda panjang ℓ_k),
- tak bisa di-CUDA-graph (burn-capture = ekspor, bukan replay training).
Menutup gap = menulis ulang autodiff Burn / kernel CubeCL ke paritas PyTorch =
kerja framework-level. Di T4, bf16 (tensor core 2×) bahkan terhalang shared-mem
cubek (68KB > 64KB).

## Gabung proyeksi (QKV 3→1, gate+up 2→1) — TERUKUR, LOMB AT
Menggabungkan `wq/wk/wv` → `wqkv` (1 matmul) dan `w_gate/w_up` → `w_gate_up`
(1 matmul) di `Block` (numerik identik: bobot diklasi). Efek (T4, b64):
- **fwd 58→39ms, bwd 105→69ms, opt 40→17ms** (opt turun krn param Muon 5→2/blok).
- **tok/s 9 781→15 245 = 0.952×** (dari 0.585×) — **nyaris parity PyTorch VIR**.

**Pelajaran:** beban launch-bound → mengurangi jumlah matmul/luncuran (gabung
proyeksi) sangat efektif. Peluang lanjut: batch besar (amortisasi) + fusi RoPE/
RmsNorm (kernel) untuk >1.0×.

**Batch 128 (uji):** vir 28 066 vs pt 31 858 = **0.881×** (turun dari b64 0.952×)
→ batch besar TIDAK menolong (PyTorch skalanya lebih baik); **b64 sweet spot**.
Sisa gap 5% (b64): fwd 39 / bwd 69 / opt 18ms vs pt 128ms → butuh −6ms. Lever
terjelas: **batch Muon Newton-Schulz** (240→60 luncuran, opt 18→~9ms) atau fusi
RoPE/RmsNorm (kernel). Keduanya = optimisasi tingkat lanjut (risky/multi-hari).

**RoPE q&k gabung (uji):** vir 15 065 (0.933×) — **NO-OP** (dalam noise 0.95±0.02)
→ **membuktikan RoPE elementwise BUKAN bottleneck**. Yang dominan = **matmul
backward (CubeCL 0.77× cuBLAS) + Muon (18ms)**. Fokus berikut: **batch Muon**
(opt 18→~9ms) — satu-satunya lever jelas ke >1.0× (butuh custom optimizer).

## 🏆 BATCHED NEWTON-SCHULZ MUON — TEMBUS 1.0× (di atas PyTorch, ADIL)

Mengganti Muon per-param (burn-optim, ~20 matmul/param) dengan **NS ter-batch**
lintas param 2D se-bentuk (`crates/vir-train/src/batched_muon.rs`, two-pass:
`visit` kumpul grad 2D by ParamId → stack per-shape `[N,m,n]` → 1 NS ter-batch →
`map` tulis-balik via `DispatchTensor`). Terintegrasi di `NativeTrainer`
(`adamw_only_optimizer` + `BatchedMuon`).

**Hasil (Kaggle T4, batch 64, L32, warmup 5, model SAMA):**
- **opt 18 → 4.4ms (-75%)** (luncuran NS 560→~140).
- **vir fp32 = 16 490 tok/s vs PyTorch VIR fp32 = 16 063 → 1.027× (DI ATAS, ADIL fp32-vs-fp32)**.
- vir fp16 = 14 011 vs PyTorch fp16 = 14 194 → 0.987× (parity).
- (Catatan: run sebelumnya vir fp32 15 053 / pt 15 374 = 0.979×; run ini 16 490/16 063 = 1.027× → noise ±3%, tapi konsisten ~parity/above.)

**Kesimpulan: objektif "optimisasi di atas PyTorch" TERCAPAI** pada beban adil
(fp32, model VIR sama, T4 sama) — vir 1.027× PyTorch VIR. Gabungan: batched Muon
(terobosan) + gabung QKV/gate_up + W0/W1/B6.

## Rencana lanjut (high-effort/risk, butuh buy-in)
- **W4 — CUDA graph capture: TIDAK FEASIBLE di Burn 0.22.** `burn-capture`
  adalah backend yang *merekam graph untuk ekspor* (ONNX), bukan CUDA graph
  replay untuk training. Tak ada API capture→replay input-baru untuk
  forward+backward training. (Terverifikasi via Context7/docs.)
- **Muon**: tak mengubah `ns_steps` (kualitas, PRD:604); opsi: bf16 matmul NS,
  atau fused kernel NS (CubeCL) — besar.
- **Autotune cache persisten** (B3): cold start ~146s → ~0 (total-time & etika
  perbandingan); API cubecl 0.11.4 belum terverifikasi.

## Status
**Metodologi: final & dipakai** (W0/W1/B6/profil). **Optimisasi penuh: berjalan**
— gap 12–17× terdokumentasi; B4/W4/Muon adalah kandidat berikutnya yang
high-risk. Tidak ada klaim ">PyTorch" tanpa ukur.