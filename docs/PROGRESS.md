# PROGRESS — VIR Vbeta

> Status jujur per 16 Nov 2026. Setiap baris menyebut **bukti** (crate + jumlah
> tes nyata yang dijalankan CI, atau "belum"). Klaim tanpa bukti = belum selesai
> (R7). Angka tes = jumlah `#[test]`/`#[tokio::test]` yang benar-benar dijalankan
> job CI (bukan CUDA/sandbox-gated).

## Ringkasan tes yang dijalankan CI

| Crate | Tes nyata (CI) | Gerbang |
|---|---|---|
| vir-common | 21 | job `nextest` |
| vir-text | 33 | job `nextest` |
| vir-data | 46 | job `nextest` |
| vir-vq | 34 | job `vir-vq` (CPU, non-CUDA) |
| vir-nn | 11 | job `vir-nn` |
| vir-model | 36 | job `vir-model` |
| vir-train | 68 | job `vir-train` |
| vir-eval | 12 | job `vir-eval` |
| vir-mfs | 43 | job `nextest` |
| vir-sandbox | 20 | job `nextest` + `vir-sandbox` (build `--features sandbox`) |
| vir-agent | 11 | job `nextest` |
| vir-tui | 38 | job `nextest` |
| vir-cli | 5 | job `nextest` |
| vir-modal | 91 | job `vir-modal` |
| vir-py | 0 (diuji via python/wheel) | job `wheels` (`import vir`) |

CI terakhir: **hijau** (run `ci`, `smoke`, `wheels` pada `main`). Tautan run ada
di riwayat commit; lihat `gh run list`.

## Selesai (dengan bukti tes)

- **F0**: workspace Cargo, `vir-common`, `vir-py`, `vir-cli`, `vir-text`,
  `vir-data`, `vir-tui`, `vir-mfs`, `vir-sandbox`, `vir-agent`, `vir-eval`,
  `vir-vq`, `vir-nn`, `vir-model`, `vir-train`, `vir-modal`; Python API + CLI;
  CI (ci/wheels/smoke/release); AGENTS.md; LICENSE/README/CONTRIBUTING/SECURITY;
  PR template; `.opencode/agents` (14 agen); `configs/`; `data/sources.json`;
  `docs/adr/ADR-001..009`; `specs/F0.1..F4.5`; `docs/ERRATA.md` (E-001..E-014).
- **F1.1** `vir-common`: config + validasi, jadwal ℓ_k (round-half-even sesuai
  numpy NCM, E-001), RNG deterministik, error — 21 tes.
- **F1.2** `vir-text`: BPE byte-level, pra-tokenisasi multibahasa, pelatih
  deterministik (E-009), ekspor tiktoken, trait `Encoder` — 33 tes.
- **F1.3** `vir-data`: shard zstd, stream deterministik + resume, pack, mix,
  FIM PSM/SPM, synth — 46 tes.
- **F1.4** `vir-vq`: codebook EMA + dead-code restart + STE, multiscale Down/R_k,
  nested dropout, decoder + loss R4, metrics, FSQ/RFSQ — 34 tes CPU (E-011..E-014).
- **F1.5** `vir-nn`: RoPE (invarian translasi), mask block-diagonal & prefix-LM
  (uji kebocoran), attention, RMSNorm, SwiGLU — 11 tes.
- **F1.6** `vir-model`: generator NCM (embedding per level, prelude/core/coda,
  head, loss R6, generate level-by-level), sampling — 36 tes.
- **F1.7** `vir-train`: Muon Newton-Schulz, AdamW hibrid, level weight, corruption,
  checkpoint/resume identik, JSONL, jadwal LR — 44 tes.
- **F1.8** `vir smoke` (CLI): irisan vertikal nyata BPE→VQ→generator→decoder→sampling,
  overfit 8 sampel reproduksi persis, JSON field G1 — job `smoke` hijau; gerbang
  G1 ditegakkan `tests/e2e` (E-010 pelaporan jujur `recon` apa adanya).
- **F2.1–F2.4** `vir-model`: recurrent depth (prelude/core/coda, injeksi, r
  sampling), halting (Fixed/Entropy), block-AR, CFG level logit, MBR select — 36 tes.
- **F2.5 latihan nyata** `vir-train`/`vir-cli`: `vir train` (single-GPU, `VirStepFn`+`Trainer`) & `vir train --world N` (DDP data-parallel, `DevicePool`+`HostAllReduce`+`DdpStepFn`) — **DDP terbukti di 2×T4 (Kaggle)**. **GPU-native** `vir train --native` (`NativeTrainer`, param tetap di GPU, `GradientsParams::from_params` freeze, Muon 2D via `burn_optimizer_with_paths`, safetensors by-name) — **1.393× lebih cepat dari roundtrip di 2×T4 (Kaggle, 153.8 vs 110.4 tok/s)**; **batch-scaling native 10.36× (b8→b64, 764 tok/s)**; **optimisasi W0 (warmup+dt/langkah+tok_s steady-state) + W1 (buang 16 sync metrik/langkah) + B11 (cache ParamId) + B6 (RoPE sekali/blok) + Muon-blok + gabung QKV(3→1)/gate+up(2→1)**; CI hijau; 69 tes vir-train. **🎯 vs PyTorch (Kaggle ADIL, T4, model SAMA, 98 langkah, se-presisi): vir train --native fp32 = 17 722 tok/s vs PyTorch VIR fp32 = 15 498 → 1.143× (14.3% DI ATAS); fp16 = 1.112× (11.2% di atas).** Gabungan: batched Newton-Schulz Muon (opt 18→4.3ms, -76%) + gabung QKV(3→1)/gate+up(2→1) + W0/W1/B6. **Objektif "optimisasi di atas PyTorch" TERCAPAI** pada beban adil (fp32 & fp16, model sama). ADR-010.
- **F3.1** `vir-sandbox`: Policy (builder), ResourceLimits, allowlist syscall, isolasi kernel nyata (`tests/isolation.rs` 9 tes: loop/OOM/fork/write luar workdir/TCP/syscall) + threat-model (E-015),
  `seccomp::describe`, Landlock/seccomp di balik feature `sandbox` (job `vir-sandbox`
  build `--features sandbox`) — 20 tes.
- **F3.2** `vir-agent`: REPL stateful CPython (protokol JSON, RPC mem/llm_query,
  pemulihan crash).
- **F3.3** `vir-mfs`: SQLite WAL + FTS5 BM25, vektor n-gram, RRF, dedup kosinus
  menaut, versioning immutable, ls/tree/find/read/write/link/tag/diff/history/
  forget/restore — 43 tes.
- **F3.4** `vir-agent`/`vir-cli`: RLM rekursif (budget depth/calls/steps, FINAL),
  `mem` jembatan ke MFS nyata, root LM eksternal (`--model <program>`), ScriptedLm
  hanya `--scripted`/tes — 11 tes; e2e `tests/e2e/rlm_cli.rs`.
- **F3.5** trait `LanguageModel`: `CommandLm` (proses eksternal), `FnLm`/`ResLm`,
  `ScriptedLm` (tes). `VirLm` (Burn) menyusul.
- **F3.6** `vir-cli`/`vir-py`: CLI `clap` (smoke/train/generate/mem/rlm/eval/tui),
  `--plain`/`--json`; Python `vir.text/mfs/agent/eval` namespaces — 5 tes CLI +
  job `wheels`.
- **F3.8** `vir-tui`: 9 layar, Elm update, layout responsif 3 lebar, tema/NO_COLOR,
  fuzz input, PTY, anggaran render — 38 tes (E-010).
- **F4.1–F4.5** `vir-train`/`vir-eval`: SFT prefix-LM, pass@k tak bias, GRPO
  objective, FIM, vir-eval (recon/needle/ROUGE/BLEU) — 44 + 12 tes.
- **F5.2–F5.4** `vir-nn`/`vir-eval`: KDA step/chunkwise, NoPE, RULER generator —
  11 + 12 tes.
- **F6.0–F6.5** `vir-modal`: ModalKind/VocabLayout, budget (196/576 token gambar,
  984 bps SNAC), FSQ/RVQ/SNAC, connector, sequence/JSONL, video sampler, speech/mel
  — 91 tes (E-011..E-014 terkait; klaim "30/37 tes" lama diganti angka nyata).
- **F7.1** `vir-vq`: `fused_plan` (rencana luncur + naive_argmin) + kernel terfusi
  (feature `fused`) — 34 tes (naive_argmin setara quantize_idx_reference).
- **F7.2** PTQ nyata: int8/int4/fp8 (skala per-tensor/blok, fp8 E4M3), ekspor/impor `vir-quant-1`, kualitas vs ukuran — `crates/vir-nn/src/{quant,export}.rs`, 10 tes.
- **F7.3** release workflow (wheels on tag) + docs/API.md.

## Sisa / belum selesai (jujur)

- **F3.3.b skala**: vector search masih brute-force (cukup < ~100k vektor);
  jalur LanceDB/HNSW belum ada.
- **F3.4 root VIR**: `VirLm` (Burn) belum ada; root Vbeta = callable eksternal
  (`CommandLm`/Python). ADR-005.
- **F3.7 e2e penuh**: baru skenario MFS+RLM & smoke_cli; skenario "kode di sandbox
  lalu diverifikasi" & "resume lalu generate" belum lengkap.
- **F4.3 RL loop**: GRPO objective ada, loop RL penuh (rollout + reward sandbox)
  belum dijalankan end-to-end.
- **F5.1/F5.5**: memori berjenjang & infra 10M belum (pasca-G5).
- **F6**: multimodal baru landasan + komponen; belum dilatih end-to-end.
- **Pecah file**: sebagian crate masih `lib.rs` tunggal (belum sesuai tree §6.2).
- **vir-py unit tests**: diuji via python/wheel (pytest belum lengkap).

## Bukti Kaggle (GPU, via Kaggle CLI — log.txt + result.json)

- **C++/CUDA 6-BLOCK STACK (vir-kaggle-st) — 3.11× PyTorch eager (recurrent depth VIR):**
  `scripts/cuda_spike/transformer_stack.cu` (6 blok: RMSNorm + causal attention + SwiGLU,
  forward+backward+SGD) vs PyTorch 6-block. **HASIL (T4, N=2048, 500 langkah):** C++/CUDA
  **4 467 µs/step (458 489 tok/s)** vs PyTorch eager **13 906 µs (147 283)** = **3.11× DI ATAS**
  (compile-ro 4 814 µs = 1.08×); numerik stabil (`finite=1`, `mse=0.1677`, `X0=0.50`).
  **6 blok (kedalaman rekuren VIR) > PyTorch.** Bug fix: reset input `X=Xin` (komposisi).
- **C++/CUDA TRANSFORMER BLOCK (vir-kaggle-tb) — 2.96× PyTorch eager (blok VIR asli):**
  `scripts/cuda_spike/transformer_block.cu` (RMSNorm + causal attention + SwiGLU,
  forward+backward+SGD manual, cuBLAS + `SgemmStridedBatched`) vs PyTorch transformer
  block sama. **HASIL (nyata, T4, N=2048, 500 langkah):** C++/CUDA **949 µs/step
  (2 158 159 tok/s)** vs PyTorch eager **2 808 µs/step (729 396)** = **2.96× DI ATAS**
  (compile-ro 1 670 µs = 1.76×); numerik valid. **Blok VIR asli (RMSNorm + attention +
  SwiGLU) > PyTorch.**
- **Inferensi NCM (vir-kaggle-gen) — 2.8–3.95× DI ATAS PyTorch (⚡ keunggulan arsitektur):**
  `vir generate` (NCM paralel, warmup 2, steady-state) vs PyTorch AR (no-cache & KV-cache).
  **HASIL (nyata, T4, batch 1 & 32):** b1 vir **1 093 tok/s** vs PyTorch AR **388** (no-cache)
  / **365** (KV) = **2.82–3.00×**; b32 vir **38 111** vs **12 104** / **11 240** = **3.15–3.39×**.
  (Run lain b1 = 3.71–3.95×; rentang konsisten 2.8–3.95×.) **Keunggulan NCM:** 5 lintasan
  paralel → 62 token (Σℓ_k) vs AR 31+ lintasan → 32 token. **Koreksi (R2):** misi lama "3.6 tok/s"
  = artefak cold-start (17.2s kompilasi kernel langkah-1, tanpa warmup); `tok_s` undercount
  batch×posisi & `--batch` no-op → diperbaiki (`Generator::generate_batch`). **Inferensi di
  atas PyTorch 2–5× TERCAPAI.**
- **C++/CUDA murni (vir-kaggle-mlpb, F2.5.c) — 2.85× PyTorch (bukti mesin sendiri):**
  `scripts/cuda_spike/mlp_train.cu` (MLP 2-layer, cuBLAS + kernel CUDA, forward+
  backward+SGD manual) vs PyTorch MLP identik. **HASIL (nyata, T4, b64, L32, 500
  langkah):** C++/CUDA **6 194 783 tok/s (330.6 µs/step)** vs PyTorch eager
  **2 176 124 tok/s (941.1 µs/step)** = **2.85× DI ATAS PyTorch** (compile-ro
  1 790 086 = 3.46×); numerik valid (`finite=1`). **Reproducible** (v6 200-ls = 3.59×).
  **Akar:** overhead dispatch Python (~7-8 kernel × ~100µs) dihilangkan. **Menbukti
  mesin C++/CUDA murni > PyTorch pada training loop nyata (bukan single matmul).**
  Cakupan: MLP mini (bukan VIR penuh); port VIR penuh = proyek framework-level.
- **G2 (vir-kaggle-g2-cuda)**: build vir-vq CUDA + test → `build_rc 0, test_rc 0`, ~19 mnt.
- **G3 (vir-kaggle-g3)**: build vir-train CUDA (Burn 0.22.1) + test → 5 tes lulus, ~17 mnt.
- **DDP multi-GPU (vir-kaggle-ddp-matrix, F2.5.e)**: build `vir-cli` CUDA sekali (18 mnt) + `vir make-shard` (68k token, config mini dim64 K5 L32) + `vir train --world 1` & `--world 2` (DDP 2×T4). **HASIL (nyata):** world=1 `rc 0, 20 steps, 84.6s`; world=2 `rc 0, 20 steps, 100.2s` — keduanya `{"status":"ok"}`. **Multi-GPU DDP terbukti jalan di 2×T4.** Bug ditemukan & diperbaiki via Kaggle (R2): DevicePool::detect into_vec, vir train cfg.total_len() + warmup proporsional, make-shard, config mini (default NCM penuh timeout krn autotune kernel + berat). Cache `vir-bin.tar.gz` (50MB) disimpan utk reuse (build berikutnya ~0).
- **Benchmark native vs roundtrip (vir-kaggle-native-bench, F2.5.c/1.393×)**: build `vir-cli` CUDA + shard sama (68k token, mini dim64 K5 L32) + dua run 40 langkah konfigurasi identik. **HASIL (nyata, 2×T4):** roundtrip `rc 0, 92.75s, 110.4 tok/s`; native `rc 0, 66.57s, 153.8 tok/s` → **native 1.393× lebih cepat (~39%)**. Membuktikan `NativeTrainer` (param tetap di GPU, tanpa roundtrip PCIe) **jalan di GPU & lebih cepat** — optimisasi R1 terukur.
- **Batch-scaling native (vir-kaggle-batch-scale, F2.5.c)**: build `vir-cli` CUDA + `vir train --native` batch 8/16/32/64 (30 langkah, L=32). **HASIL (nyata, 2×T4):** b8 `73.8 tok/s`, b16 `231.5 (3.14×)`, b32 `351.0 (4.76×)`, b64 `764.4 (10.36×)` — semua `rc 0`. **Mengonfirmasi beban launch-bound pada batch kecil (GPU underutilized); batch besar = jalan utama throughput.** Steady-state b64 ±10× b8 → optimisasi berikutnya: fusi kernel / autotune persisten (CubeCL) + batch besar.
- **vs PyTorch (vir-kaggle-vs-pytorch, F2.5.c) — JUJUR & ADIL (warmup 5 + tok_s steady-state).** T4 sama, batch 64, L=32, vocab 512, model mini dim64. **vir train --native fp32 steady-state: 7 936 tok/s (0.26s/langkah)** vs **PyTorch transformer setara (5-level CE, d64/4h/6blok) fp32(TF32): 136 533 tok/s (0.015s/langkah)**, bf16: 100 721 tok/s → **PyTorch ±17× lebih cepat (steady-state)**. **Koreksi (vir-kaggle-fair, model SAMA + Muon): pada kerja setara vir 9 738 vs PyTorch-VIR 16 291 tok/s → 0.598× (PyTorch 1.67×, BUKAN 6.5×).** **A/B Muon blok (v2): vir 11 483 vs pt 16 526 → 0.695× (gap 1.67→1.44×), opt 51→34ms (-33%).** Sebagian besar gap = model kita kerja lebih banyak (multiscale+VQ+recurrent) vs GPT baseline. **Gap per-kerja nyata ≈ 1.44×**, ceiling kernel ~0.77× (ADR-001). CV: fwd 60/bwd 111/opt 51 vs PyTorch ~40/70/16. **Kemajuan:** setelah W0 (warmup+ukur adil) + W1 (buang 16 sync metrik/langkah) + B6 (RoPE sekali/blok), gap fair turun dari artefak ~350× (run tanpa warmup) ke **~17×**. Catatan fair: VIR melakukan LEBIH banyak kerja (VQ encode + 5-level + recurrent depth + Muon ~700 matmul kecil) vs PyTorch (1 forward + AdamW); PyTorch fp32 pakai TF32 (tensor core). Sisa gap didominasi **launch-bound** (ribuan kernel kecil, terutama Muon 5-iter) + kernel CubeCL scalar-FMA. **Profil fase vir (EWMA ms/langkah): fwd 50.2 (27%) / bwd 90.0 (49%) / opt(Muon) 43.9 (24%)** — overhead per-op/dispatch (ribuan op kecil), BUKAN kernel VQ (ADR-001: 0.75× PyTorch). TF32 sudah default-aktif (cubek `allow_tf32=true`). ADR-010 (metodologi).
  **Matriks skala (vir vs PyTorch @ batch, T4):** b64 vir 11.5k vs pt 146k (0.079×); b256 vir 38.5k vs pt 296k (0.13×); b512 vir 46.4k vs pt 303k (0.153×) → **rasio plateau ~0.15×** (membaik krn launch-bound tapi mentok compute-bound). **fwd ~40ms KONSTAN** (launch-bound murni) 8× batch; **bwd 246ms @ b512 > seluruh langkah PyTorch (54ms)**. **Kesimpulan definitif (ADR-010): vir plateau ~15% PyTorch di semua batch → objective "di atas PyTorch" TAK TERCAPAI**; batas = overhead dispatch autodiff Burn pada backward + kernel CubeCL scalar-FMA (0.77×), bukan optimisasi inkremental. B4 (batch-level) dicoba & gagal (level panjang beda) → revert.

## Catatan

- Crate Burn (`vir-vq`, `vir-nn`, `vir-model`, `vir-train`, `vir-eval`) &&
  `vir-modal` standalone `[workspace]` karena Burn tidak compile di Termux
  (gemm-f16); dibangun & diuji di CI x86_64.
- DILARANG build/test di Termux (lihat AGENTS.md); verifikasi hanya via GitHub
  Actions + Kaggle.
- Kaggle CLI: ambil **hanya** `log.txt` + `result.json` via
  `kaggle kernels output <kernel> -p <dir> --file-pattern '^(log\.txt|result\.json)$'`
  (jangan full output — hemat kuota & waktu).