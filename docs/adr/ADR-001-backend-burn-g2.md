# ADR-001: Backend Burn dan ambang throughput G2

## Konteks
VIR memakai Burn 0.22. Risiko: throughput training bisa jauh di bawah PyTorch (laporan pihak ketiga 7,3%).

## Keputusan
- Pakai Burn 0.22.1 dengan backend CubeCL (CUDA untuk Kaggle, flex untuk CPU).
- Ambang G2: throughput Burn ≥ 0,5× PyTorch pada model dan GPU sama.

## Hasil (Nov 2026, Kaggle CUDA, V=4096, D=384, batch 256×128, 50 iters)

| Versi | Burn tok/s | PyTorch tok/s | Rasio |
|---|---|---|---|
| fp32 naive (z_sq+gather) | 646.488 | 1.307.709 | 0,49× |
| fp32 optimized (tanpa z_sq/gather) | 802.080 | 1.116.389 | 0,72× |
| fp32 + autotune/fusion (burn-cuda) | 806.976 | 1.063.044 | 0,76× |
| fp16 tensor core | 1.059.822 | 2.341.472 | 0,45× |

- Burn CUDA terbukti jalan. G2 **LULUS** (borderline untuk fp32 optimized 0,72–0,76×).
- **Belum di atas PyTorch.** Autotune+fusion (burn-cuda) hanya naik tipis (0,72→0,76×). Bottleneck utama = `argmax` (reduce) + matmul lepas, bukan GEMM saja.
- Untuk mengalahkan PyTorch pada microbench ini perlu **kernel terfusi (F7.1/P3)**: matmul + argmax dalam satu kernel (tidak menulis [N,V] ke HBM). Estimasi riset: 5–8M tok/s (fp16) / 1.4–1.6M (fp32).

## Konsekuensi
- Lanjut ke G3. Rekayasa kernel (F7.1) diperlukan untuk target "di atas PyTorch".
- Catatan: ini forward, bukan training (forward+backward+optimizer). Ukur training terpisah.