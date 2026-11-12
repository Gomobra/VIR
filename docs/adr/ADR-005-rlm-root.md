# ADR-005: Root RLM untuk Vbeta

## Konteks
RLM butuh root yang bisa menulis kode. VIR belum tentu cukup baik.

## Keputusan
- Trait `LanguageModel`: `VirLm`, `PyCallableLm` (callable Python), `ScriptedLm` (tes).
- Default Vbeta: `PyCallableLm` (bisa pakai model eksternal) + `ScriptedLm` untuk tes.
- `VirLm` dipakai bila model VIR cukup baik.

## Status
`ScriptedLm` ada. `VirLm`/`PyCallableLm` belum.