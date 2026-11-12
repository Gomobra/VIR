# ADR-007: Linear attention / KDA

## Konteks
Kimi Linear: KDA hibrid 3:1 dengan attention penuh (NoPE). KDA kausal; NCM block-diagonal non-kausal.

## Keputusan
- Implementasi KDA naif (oracle) lalu chunkwise di `vir-nn`.
- Hibrid 3:1, NoPE pada lapisan penuh.
- Ablasi rasio; ukur 128K/512K/1M.
- Uji salin-persis + needle (kecocokan kausal vs non-kausal).

## Status
Belum diimplementasikan (F5.2).