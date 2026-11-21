# ADR-011: Integrasi engine C++/CUDA ke CLI (`vir cuda-train`)

## Konteks
Engine C++/CUDA murni (ADR-010) mencapai **2.85–3.44× PyTorch** pada training (MLP,
transformer block, 6-block, full VIR). Untuk menjadikannya "mesin VIR" yang nyata (bukan
PoC Kaggle), engine harus diintegrasikan ke CLI `vir`. Tantangan: engine C++ (nvcc) dan
CLI Rust (cargo) adalah dua toolchain terpisah; checkpoint harus kompatibel.

## Keputusan
1. **Subcommand Rust `vir cuda-train`** (`crates/vir-cli/src/cmd_cuda.rs`):
   - Memuat shard (`vir_data::ShardReader`), mengekspor token ke berkas datar uint32
     (`<out>/tokens.bin`), lalu memanggil binari C++ `vir-cuda-train` (dari `VIR_CUDA_BIN`
     atau `PATH`).
   - Melaporkan status JSON.
2. **Binari C++/CUDA `vir-cuda-train`** (`scripts/cuda_spike/vir_cuda_train.cu`):
   - Membaca config (argv) + token (uint32), melatih model (VQ encode + 6-block +
     5-level multiscale heads + CE + Muon), menulis checkpoint (`manifest.json` +
     `model.bin`).
   - Dibangun terpisah (nvcc); `vir cuda-train` memerlukannya di `PATH`.
3. **Checkpoint format** `vir-cuda-1` (sementara): `manifest.json` (nama+shape tensor) +
  `model.bin` (float32 datar). Kompatibilitas penuh `vir generate` (safetensors Burn) =
  langkah lanjutan.

## Hasil (Kaggle T4, `vir-kaggle-vct`, 100 langkah, N=2048, 6 blok)
- **vir-cuda-train: 137 290 tok/s** (14 917 µs/step), `finite=1`, `loss=24 746` (sum,
  model untrained/uniform — grad averaged /M).
- **Checkpoint valid**: `manifest.json` (53 tensor, format `vir-cuda-1`) + `model.bin`
  (672 512 float, `match: True`), `rc 0`.
- **Bug diperbaiki (R2)**: (1) `#define NBLK 6` dari kernel section membayangi variabel
  runtime `NBLK` → `6=6`; (2) `K`/`V` (level count/vocab) bentrok tensor `K[b]`/`V[b]` →
  `NK`/`VOCAB`; (3) checkpoint `fwrite` menulis **pointer device** → segfault + 0 byte →
  copy device→host dulu; (4) manifest leading comma + heads separator; (5) `fflush` +
  `_Exit` (hindari cleanup segfault).

## Keterbatasan (jujur)
- Model C++ = VIR **versi ringkas**: codebook **fixed** (tanpa EMA update), 6 blok flat
  (bukan prelude/core/coda ber-struktur), tanpa adapter/proj/phis. Checkpoint belum
  kompatibel `vir generate` (Burn safetensors).
- Integrasi penuh (checkpoint Burn-compatible + `vir generate` baca model C++) = lanjutan.
- `vir cuda-train` butuh binari `vir-cuda-train` (nvcc) di `PATH` — belum dibangun di CI.

## Status
**Integrasi pertama (R7)**: plumbing nyata (Rust subcommand + C++ engine + checkpoint
valid). **Objective "mesin sendiri > PyTorch" (ADR-010) + integrasi CLI (ADR-011)** —
engine C++/CUDA 2.85–3.44× PyTorch, kini dapat dipanggil via `vir cuda-train`.