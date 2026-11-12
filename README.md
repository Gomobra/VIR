# VIR Vbeta

**VIR** = *Variable-depth Iterative Reasoner*. Arsitektur model bahasa dari nol: NCM (multiscale residual VQ-VAE + generator), recurrent depth, halting, block-AR, RLM, dan memori file manager. Inti Rust (Burn), API Python (PyO3 + maturin), CLI `vir`, TUI `vir tui`.

## Status
- Fondasi F0–F4 sebagian: workspace, CI hijau, wheel manylinux, Kaggle G2 (Burn CUDA 0.49× PyTorch pada forward VQ).
- Lihat `docs/PROGRESS.md` dan `PRD.md` untuk detail.

## Struktur
- `crates/` — crate Rust (vir-common, vir-text, vir-data, vir-vq, vir-nn, vir-model, vir-train, vir-eval, vir-mfs, vir-sandbox, vir-agent, vir-tui, vir-py, vir-cli).
- `python/vir/` — paket Python tipis.
- `scripts/` — skrip Kaggle + notebook G2.
- `docs/` — ERRATA, PROGRESS, ADR.
- `specs/` — spec per fitur.

## Build & Test
```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
bash scripts/forbid_stubs.sh
```

## Wheel
```bash
maturin build --release   # manylinux x86_64 di CI
pip install vir
python -c "import vir; print(vir.build_info())"
```

## Kaggle GPU
- `scripts/vir_kaggle.ipynb` — smoke (wheel dari GitHub Release).
- `scripts/vir_kaggle_g2.ipynb` — G2 CUDA smoke.
- `scripts/vir_kaggle_g2_bench.ipynb` — throughput Burn vs PyTorch.

## Lisensi
MIT (lihat `LICENSE`). Kode acuan NCM/ARLoop MIT.