# INSTRUCTIONS — VIR (Model AI 1B dari Scratch, Zig)

> Dokumen instruksi lengkap untuk melanjutkan proyek. Berisi aturan, rencana, status implementasi, error yang pernah terjadi, dan sisa pekerjaan.

---

## 0. Aturan Kerja (Mengikat)

1. **Jangan debug dan test di HP lokal.** Semua build/test berjalan di GitHub Actions (runner Ubuntu).
2. **Akses GitHub via `gh` CLI** (akun `Gomobra`, repo `Gomobra/VIR`). Setiap PR/commit harus punya bukti run CI.
3. **Selalu bersihkan penyimpanan** setelah ada tambahan (build artifact, cache). Hapus `.zig-cache`, `zig-out`, file tes di `/tmp/scratchpad`.
4. **Gunakan Kaggle CLI** bila perlu GPU. Saat ini kuota GPU habis — pekerjaan GPU ditunda.
5. **Selesaikan semuanya langsung**, boleh pakai semua MCP (context7, exa, deepwiki, dll.).
6. **Jangan pakai kode lama (Rust VIR) sebagai main code.** Repo VIR sekarang adalah proyek Zig dari nol.
7. **Jangan buat repo baru.** Pakai repo yang ada: `Gomobra/VIR`.
8. **Pin Zig 0.17.1** (atau 0.16.0 per PRD, tapi CI pakai 0.17.1 karena tersedia). Jangan upgrade di tengah fase.
9. **Tidak ada stub/todo/unimplemented.** Setiap berkas harus langsung berfungsi.
10. **Setiap fitur lulus gerbang (gate)** dengan bukti CI sebelum fitur berikutnya.
11. **Integrasi dibangun bersamaan**, bukan di akhir. Tiap crate langsung dihubungkan ke CLI/tes.
12. **Commit message** mengikuti ID tasklist (mis. `T3.3-05: matmul tensor core`).
13. **Eksperimen:** satu catatan per ablasi di `docs/`.

### Konvensi Kode (Zig 0.17)
- Modul: huruf kecil dengan garis bawah (`tensor.zig`).
- Tipe: PascalCase. Fungsi: camelCase.
- **`var` hanya jika variabel dimutasi**, selain itu pakai `const` (Zig 0.17 menolak `var` yang tidak pernah dimutasi).
- Hindari nama yang shadow primitive Zig: `i0`, `i1`, `i2`, `i3` (gunakan `ii0`, `iidx`, `ia`, dll.).
- Hindari `std.heap.GeneralPurposeAllocator` (tidak ada di 0.17) → pakai `std.heap.page_allocator`.
- API lama yang dihapus di 0.17: `std.io.getStdOut()`, `std.fs.File`, `std.process.argsAlloc`, `std.os.argv` → gunakan `std.debug.print`, `std.os.argv` (kadang tidak ada, gunakan argumen main tanpa args).
- Array repeat: `[_]u8{0}**32` (tanpa spasi di sekitar `**`).
- For-loop dengan assignment: butuh braces `{ }`.
- `zig fmt` wajib lulus sebelum push.
- Tidak ada `unreachable`, `todo!()`, stub.

---

## 1. Overview Proyek

**VIR** = model AI 1B parameter dari scratch di **Zig**, multimodal (teks, gambar, audio, video).  
Repo: `https://github.com/Gomobra/VIR`  
PRD: `PRD.md` (1.945 baris, versi Zig; ada juga PRD Rust lama yang tidak dipakai).

### Kriteria Keberhasilan
1. Engine Zig benar secara numerik (gradient check + parity oracle).
2. Model teks terlatih sejauh anggaran memungkinkan.
3. Reasoning dasar terukur naik.
4. Setiap modalitas non-teks lulus minimal tier pertama.
5. Hasil reproducible dari repo + config + checkpoint.

### Keputusan Terkunci
| Hal | Keputusan |
|---|---|
| Bahasa | Zig 0.17.1 (CI), target 0.16.0 di PRD (disesuaikan) |
| Hardware | Kaggle 2×T4 gratis; GitHub Actions hanya CI |
| Tokenizer | Custom (SuperBPE/BPE) di Zig |
| Presisi | fp16 + loss scaling dinamis, master fp32 (T4 tanpa bf16) |
| Optimizer | Muon (matriks 2D) + AdamW (embedding/norm/skalar) |
| Arsitektur | Hybrid GDN + attention 3:1, Engram, BitLinear, looping, MTP |

---

## 2. Milestone (PRD §2.5)

| ID | Milestone | Status |
|---|---|---|
| M0 | Verifikasi lingkungan | ⏳ Belum (Kaggle kuota habis) |
| M1 | Engine CPU (tensor, kernel, autograd, optimizer) | ✅ Sebagian besar |
| M2 | Tokenizer | ✅ BPE dasar + encode/decode |
| M3 | Pipeline data | ✅ Filter, dedup, shard, decontam, mix, FIM, packing |
| M4 | Backend GPU dasar | ⏳ Bindings saja |
| M5 | Training GPU end-to-end | ⏳ |
| M6 | Checkpoint + watchdog | ✅ Protokol + watchdog |
| M7 | Kernel lanjut + multi-GPU | ⏳ |
| M8 | Proxy P2 + muP | ⏳ |
| M9 | Ablation ladder | ⏳ |
| M10 | Run 1B | ⏳ |
| M11 | Decay + reasoning | ⏳ |
| M12 | Multimodal fondasi + audio | ⏳ |
| M13 | Gambar | ⏳ |
| M14 | Video | ⏳ |

---

## 3. Status Implementasi (Modul yang Sudah Ada)

Semua modul di bawah ini **lulus CI** (fmt + build + test) di GitHub Actions.

### Engine Inti
- **tensor** (`src/tensor/tensor.zig`): DType, Tensor, dtypeSize, view, f32↔f16.
- **cpu** (`src/cpu/cpu.zig`): elementwise (add, mul, scale, relu, relu²), matmul naif+blocked+SIMD, softmax stabil, RMSNorm, embedding gather, attention kausal, cross-entropy fused, block-diagonal mask.
- **autograd** (`src/autograd/autograd.zig`): Tape, backward add/mul/scale/relu/matmul/softmax/rmsnorm/CE/embedding/RoPE/attention, gradCheckScalar, gradient-check tests.
- **optim** (`src/optim/optim.zig`): AdamW, Muon (Newton-Schulz5), muP init std + LR scaling, clipGradNorm.
- **nn** (`src/nn/nn.zig`): Linear, RMSNorm, RoPE (fwd+bwd), GDN (forward+backward+chunked+parity), Engram (forward+backward), BitLinear (BitNet b1.58, fwd+bwd).
- **model** (`src/model/model.zig`): Block (fwd+bwd), full Model (embed+blocks+head) with forward+backward+trainStep.

### Tokenizer & Data
- **tokenizer** (`src/tokenizer/tokenizer.zig`): Vocab, encodeBytes, BPETrainer (byte-level), encode/decode exact roundtrip.
- **data** (`src/data/data.zig`): ShardHeader, Sampler, quality filter, exact+MinHash dedup, ZTOK shard writer/reader, sampler resume, decontamination (n-gram), data mix weights, runPipeline, FIM (PSM/SPM), pipelineToShards, MixedSampler, sequence packing.

### Training & Eval
- **train** (`src/train/train.zig`): TrainConfig, Trainer, TinyModel (end-to-end CPU training step + greedy generate), WSD scheduler, runTrainingLoop with checkpoint, TST (multi-hot CE + bag), GradAccumulator, cosineLr, LossScaler (dynamic), ZeRO-1 sharding helper, LRScheduler (wsd/cosine/linear).
- **eval** (`src/eval/eval.zig`): bitsPerByte, mcAccuracy, needleHaystack, rougeL, perplexity, f1Score, exactMatch, passAtK, bleu.

### Inferensi & Modalitas
- **infer** (`src/infer/infer.zig`): argmax, temperatureScale, topK, topP, repetitionPenalty, logSoftmax, softmax, sampleCategorical, KVCache.
- **modality** (`src/modality/modality.zig`): Modality, VocabLayout, defaultLayout (73984), FSQ, RVQ, VQ (EMA + dead-code restart), Connector (fwd+bwd).

### Infrastruktur
- **checkpoint** (`src/checkpoint/checkpoint.zig`): writeF32Tensor (safetensors), atomic save/load/latest, verifyCommit, Watchdog, SIGTERM handler.
- **gpu** (`src/gpu/gpu.zig`): CUDA Driver API bindings + memory management (alloc/free/memcpy/stream).
- **kernels** (`src/kernels/kernels.zig`): PTX vector_add + tiled matmul placeholder.
- **config** (`src/config/config.zig`): RunConfig, ModelConfig, TrainConfig, paramCount, parseKeyValue, loadConfigFile, validateModelConfig.

### File Infrastruktur
- `build.zig` / `build.zig.zon` (Zig 0.17.1, fingerprint).
- `.github/workflows/ci.yml` (fmt + build + test).
- `src/main.zig`, `src/root.zig` (CLI: train demo).

---

## 4. Error yang Pernah Terjadi & Solusinya

| Error | Penyebab | Solusi |
|---|---|---|
| `expected enum literal` di build.zig.zon `.name` | Zig 0.17 memerlukan enum literal, bukan string | Ubah `.name = "vir-zig"` → `.name = .vir_zig` |
| `missing top-level 'fingerprint' field` | Zig 0.17 mewajibkan fingerprint di build.zig.zon | Tambah `.fingerprint = 0x13dc013569a979aa` |
| `root source file struct 'std' has no member named 'io'` | Zig 0.17 menghapus `std.io` | Ganti dengan `std.debug.print` |
| `struct 'fs' has no member named 'File'` | Zig 0.17 mengubah `std.fs.File` | Ganti dengan `std.debug.print` |
| `struct 'heap' has no member named 'GeneralPurposeAllocator'` | Zig 0.17 menghapus GPA | Pakai `std.heap.page_allocator` |
| `struct 'process' has no member named 'argsAlloc'` | Zig 0.17 mengubah API args | Ganti `main` tanpa args, pakai `std.os.argv` (kadang tidak ada → hapus args) |
| `struct 'os' has no member named 'argv'` | Zig 0.17 tidak expose `std.os.argv` di semua target | Hapus parsing args, langsung jalankan demo |
| `local variable is never mutated` (Zig 0.17) | Zig 0.17 menolak `var` yang tidak dimutasi | Ubah `var` → `const` untuk variabel yang tidak dimutasi |
| `name shadows primitive 'i0'/'i1'/'i2'/'i3'` | Zig 0.17 memiliki primitive dengan nama tersebut | Rename ke `ii0`, `iidx`, `ia`, `ii`, dll. |
| `binary operator '*' has whitespace on one side` | Zig fmt mewajibkan tanpa spasi di `**` | Ubah `[_]u8{0} ** 32` → `[_]u8{0}**32` |
| `expected ',' after field` / `EOF` | Heredoc (`<<EOF`) meninggalkan string "EOF" di akhir file | Hapus baris "EOF" yang tersisa di akhir file (via python) |
| `local constant shadows declaration of 'std'` | Redeclare `const std` di dalam test | Hapus `const std = @import("std")` dari test, pakai module-level |
| `local variable is never mutated` (alloc results) | `var x = try alloc(...)` tidak dimutasi pointer-nya | Ubah `var` → `const` |
| `duplicate struct member name 'ZeroShard'` | Menambahkan ZeroShard ke file yang sudah memilikinya (dari restore git) | Hapus duplikat (truncate ke occurrence pertama) |
| `duplicate test name 'ZeroShard range'` | Duplikat test | Hapus duplikat |
| `use of undeclared identifier 'dTables'` | Backward signature tidak menyertakan dTables param | Tambahkan `dTables` ke signature backward Engram |
| `redeclaration of local variable 'i'` | Variabel `i` dideklarasikan dua kali di scope yang sama | Rename salah satu (mis. `i3` → `ia`) |
| `name shadows primitive 'i3'` | `i3` juga shadow primitive | Rename ke `ia` |
| `unused local variable/constant/parameter` | Zig 0.17 menolak variabel tidak terpakai | Tambah `_ = var;` atau hapus |
| `invalid left-hand side to assignment` (for-loop) | Zig tidak izinkan assignment di dalam for-loop tanpa braces | Tambah braces `{ }` |
| `duplicate test name 'KVCache append/get'` | Duplikat test dari edit yang tidak sengaja | Hapus duplikat, simpan satu |
| Network error saat `gh run view` | Transien network | Retry setelah beberapa detik |

### Pola Umum yang Harus Dihindari
1. **Heredoc (`<<'EOF'`)** sering meninggalkan "EOF" di akhir file → selalu cek dan hapus baris stray "EOF".
2. **Menambahkan komponen yang sudah ada** (ZeroShard, KVCache) → selalu cek dulu apakah sudah ada sebelum menambah.
3. **`var` vs `const`**: Zig 0.17 sangat ketat; gunakan `const` kecuali benar-benar dimutasi.
4. **Nama variabel**: hindari `i0`, `i1`, `i2`, `i3` (shadow primitive).
5. **Array repeat spacing**: `[_]u8{0}**32` (tanpa spasi).
6. **For-loop dengan assignment**: butuh braces.

---

## 5. Struktur Repositori

```
repo/
├── build.zig
├── build.zig.zon          # pin Zig 0.17.1; fingerprint
├── src/
│   ├── tensor/            # Tensor, dtype, allocator, view
│   ├── cpu/               # kernel CPU: matmul, elementwise, reduksi, konv
│   ├── gpu/               # binding CUDA Driver API, alokator device
│   ├── kernels/           # kernel Zig → PTX (sm_75)
│   ├── autograd/          # tape, backward, gradient check
│   ├── nn/                # linear, norm, rope, attention, gdn, engram, bitlinear, mlp
│   ├── model/             # config, registri komponen, param_count, Model
│   ├── optim/             # adamw, muon, jadwal WSD, loss scaler, muP, clipGradNorm
│   ├── train/             # loop, dp/zero1, checkpoint, watchdog, spike, TST
│   ├── tokenizer/         # pra-tokenizer, trainer bpe/superbpe, encoder, decoder
│   ├── data/              # filter, dedup, shard, mix, sampler, packing, FIM
│   ├── eval/              # bpb, runner benchmark, metrik modalitas
│   ├── infer/             # forward-only, cache, sampling, ternary, KVCache
│   ├── modality/          # mimi, vq gambar, vq video, dekoder, FSQ, RVQ, VQ, Connector
│   ├── checkpoint/        # safetensors, resume, watchdog
│   ├── config/            # RunConfig, ModelConfig, TrainConfig
│   ├── main.zig
│   └── root.zig
├── tests/
│   └── golden/            # biner referensi dari oracle (dikomit)
├── oracle/                # skrip PyTorch pembuat data referensi
├── scripts/               # shell: kaggle, hub sync, glue media
├── configs/               # model/*.yaml, mix/*.yaml, run/*.yaml
└── docs/                  # PRD.md, catatan eksperimen
```

---

## 6. Tech Stack

| Lapisan | Pilihan |
|---|---|
| Bahasa utama | Zig 0.17.1 (CI) |
| Build | `build.zig` + `build.zig.zon` |
| Tensor/CPU | Kode sendiri: `@Vector` SIMD + thread pool statis |
| GPU | Kernel Zig → PTX (backend NVPTX, target sm_75) + CUDA Driver API |
| Presisi | fp16 + loss scaling dinamis, master fp32 |
| Multi-GPU | Data-parallel + ZeRO-1 buatan sendiri |
| Optimizer | AdamW + Muon (Newton-Schulz) |
| Hyperparameter | muP / CompleteP |
| Arsitektur | Hybrid GDN + attention GQA, Engram, BitLinear, looping, MTP |
| Tokenizer | Trainer dan encoder SuperBPE/BPE di Zig |
| Data | JSONL → shard ZTOK memmap |
| Checkpoint | safetensors ditulis manual + biner sendiri |
| Evaluasi | Runner BPB dan log-likelihood di Zig |
| Oracle tes | PyTorch (hanya membuat data referensi) |
| Orkestrasi | Shell + Kaggle CLI + YAML |
| CI | GitHub Actions (CPU) |

---

## 7. Cara Build & Test

```bash
# Format check
zig fmt --check src

# Build
zig build

# Test
zig build test

# Stubs check
bash scripts/forbid_stubs.sh
```

Di CI (GitHub Actions), semua ini dijalankan otomatis pada push/PR.

---

## 8. Pekerjaan yang Masih Tersisa (Sesuai PRD)

### Backend GPU (M4, M7)
- Binding CUDA Driver API sudah ada (bindings + memory management).
- Kernel nyata belum: matmul tensor core (`mma.m16n8k8`), flash attention, GDN chunked, konvolusi.
- PTX compilation bisa di laptop; **menjalankan** kernel hanya di Kaggle (kuota habis).
- Workaround zig_cuda: UBSan off, `pub fn` bukan `export fn`, 5 workaround lainnya.

### Autograd Penuh (M1)
- Tape + backward helpers sudah ada.
- Gradient check f64 untuk semua op belum lengkap (harness `gradCheckScalar` ada).
- Backward untuk attention, softmax, RMSNorm, CE, embedding, RoPE, matmul sudah ada.

### Tokenizer (M2)
- Vocab + BPE trainer + encode/decode sudah ada.
- SuperBPE dua tahap + sapuan titik transisi belum.
- Encoder cepat (cache, paralel) belum.
- Peta kompresi Engram belum.

### Pipeline Data (M3)
- Filter, dedup (exact+MinHash), decontam, shard, mix, FIM, packing sudah ada.
- Near-dup LSH (16 pita × 8 baris) belum dioptimasi.
- Unggah shard sebagai Kaggle Dataset belum.

### Checkpoint (M6)
- Safetensors writer + protokol atomik + watchdog sudah ada.
- Resume dari Hub belum.
- Uji mati paksa ≥ 10 kali belum.

### Training 1B (M10)
- WSD + TST + muP + LossScaler + ZeRO-1 helper sudah ada.
- Training loop end-to-end dengan Model (bukan TinyModel) belum.
- ZeRO-1 reduce-scatter/all-gather nyata belum.

### Multimodal (M12-M14)
- Fondasi (vocab layout, FSQ, RVQ, VQ, Connector) sudah ada.
- Mimi audio codec (port ke Zig) belum.
- VQ-VAE gambar/video belum dilatih.
- Generasi gambar/video/audio belum.

### Evaluasi
- BPB, mcAccuracy, needleHaystack, rougeL, perplexity, f1, exactMatch, pass@k, bleu sudah ada.
- Runner benchmark pilihan ganda (MMLU, ARC) belum.
- Set evaluasi reasoning/coding tertahan belum.

---

## 9. Cara Melanjutkan (Next Steps)

### Saat kuota Kaggle reset:
1. **M0**: Verifikasi lingkungan (Zig + libcuda, `vector_add` PTX, query spesifikasi GPU).
2. **M4**: Backend GPU dasar (elementwise, reduksi, matmul naif → tiled → tensor core).
3. **M5**: Training GPU end-to-end (P1 model).
4. **M7**: Kernel lanjut + multi-GPU + G-SCALE.

### CPU-side (bisa dikerjakan sekarang):
1. Selesaikan autograd gradient check untuk semua op.
2. Implementasi tokenizer BPE/SuperBPE pelatih + encoder cepat.
3. Pipeline data: integrasi filter→dedup→decontam→shard→mix end-to-end.
4. Checkpoint: resume dari Hub, uji mati paksa.
5. Training loop dengan Model (bukan hanya TinyModel).
6. Multimodal: port Mimi, implementasi VQ-VAE.
7. Evaluasi: runner benchmark, set tertahan.

### Prioritas (sesuai risiko R-18: beban engineering):
- Fokus pada **irisan vertikal** dulu: satu jalur penuh teks→teks yang bekerja nyata.
- Jangan sentuh fitur baru sebelum irisan vertikal lulus gerbang.

---

## 10. Kontak & Referensi

- **Repo:** https://github.com/Gomobra/VIR
- **PRD:** `PRD.md` (1.945 baris)
- **CI:** GitHub Actions (`ci.yml`)
- **Kaggle:** `scripts/vir_kaggle*.ipynb`

### Sumber Riset (PRD §1.9)
- Zig GPU: backend NVPTX, zig_cuda, Fucina, zgml, Zigrad.
- Arsitektur: Gated DeltaNet (arXiv 2412.06464), Qwen3-Next, Kimi Linear.
- Engram: arXiv 2601.07372.
- BitNet: arXiv 2504.12285, 2411.05882.
- Optimizer: Muon (Keller Jordan), Moonlight.
- Training: TST (arXiv 2605.06546), WSD (arXiv 2410.05192), CompleteP (arXiv 2512.22382).
- Tokenizer: SuperBPE (arXiv 2503.13423), Gigatoken.
- Reasoning: RLP (arXiv 2510.01265), Thinking Mid-training (Meta).
- Audio: Mimi/Moshi (Kyutai).
- Gambar: MicroDiT (arXiv 2407.15811).
- Video: MagViT-v2, LTX-Video, SANA-Video.

---

## 11. Catatan Penting

- **Jangan buat repo baru.** Pakai `Gomobra/VIR`.
- **Jangan pakai kode lama (Rust VIR) sebagai main code.**
- **Jangan debug/test di HP.** Gunakan GitHub Actions.
- **Selalu bersihkan `.zig-cache` dan `zig-out` setelah build.**
- **Kuota Kaggle habis** — pekerjaan GPU ditunda.
- **Setiap commit harus punya bukti CI hijau.**
- **Tidak ada stub/todo/unimplemented.**

---

*Dokumen ini dibuat pada 27 November 2026. Update sesuai progress.*
EOF
echo "written"