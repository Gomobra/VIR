# ADR-004: Adapter recurrent depth

## Konteks
Recurrent depth (Huginn) pakai injeksi `e` tiap iterasi. Parcae menetapkan `A = Diag(-exp(logA))` agar `ρ(A)<1`.

## Keputusan
- Default: adapter konkatenasi `[s_i, e]` (Huginn).
- Alternatif: Parcae linear (ZOH) — ablasi.
- `n_loops` acak saat latih, tetap saat inferensi.
- BPTT terpotong k=8.

## Status
Belum diimplementasikan (F2.1). ADR final setelah ablasi.