# Contributing

## Aturan
- Ikuti R1–R10 di `PRD.md`.
- Satu crate = satu wilayah tulis = satu subagen (lihat `AGENTS.md`).
- Tidak ada stub/`todo!`/`unimplemented!`/`unwrap` di kode pustaka.
- Setiap PR menyertakan **Bukti** (tautan run CI, benchmark, metrik).

## Alur
1. Buat spec `specs/<ID>.md` (tujuan, trait antarmuka, kriteria tes).
2. Implementasi di crate terkait.
3. `cargo fmt`, `cargo clippy -- -D warnings`, `cargo test`, `bash scripts/forbid_stubs.sh`.
4. Buka PR; reviewer menjalankan `cargo mutants -p <crate>`.
5. Merge setelah CI hijau + review.

## Commit
Format: `area: ringkasan` (contoh: `vq: tambah EMA update`).