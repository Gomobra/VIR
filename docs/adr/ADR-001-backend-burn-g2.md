# ADR-001: Backend Burn dan ambang throughput G2

## Konteks
VIR memakai Burn 0.22. Risiko: throughput training bisa jauh di bawah PyTorch (laporan pihak ketiga 7,3%).

## Keputusan
- Pakai Burn 0.22.1 dengan backend CubeCL (CUDA untuk Kaggle, flex untuk CPU).
- Ambang G2: throughput Burn ≥ 0,5× PyTorch pada model dan GPU sama.

## Hasil (Nov 2026, Kaggle CUDA)
- Forward VQ (matmul + argmin + gather), batch 256×128, V=4096, D=384, fp32:
  - Burn: 646.488 tok/s
  - PyTorch: 1.307.709 tok/s
  - Rasio: **0,49×** (tepat di ambang).
- Burn CUDA terbukti jalan. G2 **LULUS** (borderline).

## Konsekuensi
- Lanjut ke G3. Rekayasa kernel (F7.1) bila training penuh < 0,5×.
- Catatan: ini forward, bukan training (forward+backward+optimizer). Ukur training terpisah.