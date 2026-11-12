# ADR-003: Operator Down_k / R_k

## Konteks
NCM tidak menyebut bentuk pasti. Kandidat: pooling/interpolasi, konvolusi (VAR), SNAC.

## Temuan riset (kode HazyResearch/ncm)
- Teks: `Down_k` = `F.interpolate(mode="area")` (k>0) / learned conv stride (level 0); `R_k`/up = `F.interpolate(mode="linear")` + residual conv `Phi`.
- SNAC (average pooling + nearest-neighbor) hanya untuk audio (F6.4), bukan teks.

## Keputusan
- Teks: area (down) + linear (up) + Phi residual conv. Cocok kode NCM.
- Audio: SNAC-style (F6.4).
- Koreksi PRD F1.4.b (SNAC untuk teks) dicatat di ERRATA.