# ADR-008: Representasi multimodal

## Konteks
Gambar/audio pakai kode diskret berlevel (sejajar teks). Encoder beku + konektor.

## Keputusan
- Gambar: tokenizer multiscale FSQ/iFSQ + decoder; encoder visi beku (SigLIP kandidat).
- Audio: codec residual (SNAC/Mimi/DAC); encoder ucapan beku (Whisper kandidat).
- Video: sampling frame -> token gambar + penanda waktu.
- Crate `vir-modal` dibuat hanya saat Fase 6.

## Status
Belum diimplementasikan (F6).