# AGENTS.md — VIR Vbeta

## Tujuan
VIR = Variable-depth Iterative Reasoner. Inti Rust (Burn), API Python (PyO3+maturin), CLI `vir`, TUI `vir tui`. Lihat PRD.md.

## Aturan (mengikat)
R1 efektif (ukur dulu), R2 bug akar+tes regresi+ERRATA, R3 kode rapi (≤40 baris/fungsi, ≤400 baris/berkas, no unwrap di lib), R4 satu crate=satu subagen, R5 nama/versi, R6 tanpa HF Tokenizers, R7 selesai dulu (no stub), R8 integrasi lengkap, R9 UI/UX terminal, R10 teknologi terbaik dengan bukti.

## Peta crate
vir-common (config, error, rng, schedule), vir-py (binding), vir-cli (CLI). Crate lain ditambah sesuai fase.

## Perintah
- fmt: `cargo fmt --check`
- clippy: `cargo clippy --workspace --all-targets -- -D warnings`
- test: `cargo nextest run --workspace`
- stubs: `bash scripts/forbid_stubs.sh`
- build wheel: `maturin build` (di CI)

## Spec
Satu berkas per fitur di `specs/<ID>.md` dengan trait antarmuka sebelum implementasi.

## Bukti
Setiap PR menyertakan tautan run CI. Tanpa bukti, tugas belum selesai.