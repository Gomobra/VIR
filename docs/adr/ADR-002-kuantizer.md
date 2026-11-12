# ADR-002: Kuantizer (VQ-EMA vs FSQ)

## Konteks
NCM memakai VQ-EMA. Risiko codebook collapse (risiko 12). FSQ/RFSQ/iFSQ bebas collapse.

## Keputusan (sementara)
- Baseline: VQ-EMA (match NCM, `commitment_cost=0.25`, decay 0.99).
- Ablasi G1: VQ-EMA vs FSQ `[7,5,5,5,5]` vs iFSQ `[16,16,16]`.
- Kriteria: akurasi rekonstruksi eksak, utilisasi, stabilitas.
- Pemenang ditetapkan setelah G1.

## Status
Belum diimplementasikan penuh (F1.4 skeleton). ADR final setelah ablasi.