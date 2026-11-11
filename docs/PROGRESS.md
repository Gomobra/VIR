# PROGRESS — VIR Vbeta

## Selesai
- F0.1 (sebagian): workspace Cargo, vir-common, vir-py, vir-cli, vir-text, vir-data, vir-tui, vir-vq, vir-nn, vir-model, vir-train (Burn, di luar workspace Termux, dibangun di CI), python/vir, pyproject, CI workflows, AGENTS.md, docs, specs/F0.1.
- F1.8 (sebagian): `vir smoke --json` menjalankan irisan BPE train+encode+decode roundtrip; `vir tui` membuka dasbor Ratatui.
- F2.x (sebagian): recurrent depth, halting (Fixed/Entropy), sampling (CFG, MBR), level weight, pass@k (tak bias) di vir-model/vir-train.
- F3.x (sebagian): vir-mfs (SQLite WAL, versioning, FTS5, forget/restore, RRF, tes invarian), vir-sandbox (subprocess + timeout + truncation), vir-agent (LanguageModel trait, ScriptedLm, RLM orchestrator + budget), vir-py bindings (Memory, rlm_run), CLI mem/rlm.
- CI: semua job hijau (fmt, clippy, nextest, forbid_stubs, deny, vir-vq, vir-nn, vir-model, vir-train, vir-eval); wheels manylinux x86_64 terbangun.
- F4 (sebagian): vir-eval (pass@k, recon accuracy, needle gen), FIM di vir-data, pass@k tak bias di vir-train, SFT pack, GRPO advantage, kaggle_train.py skeleton.
- CI: semua job hijau; wheels manylinux x86_64 terbangun.
- Gerbang lokal: cargo build/test/clippy/fmt hijau; wheel maturin terbangun; import vir OK; text_encode/decode OK; vir-data shard/pack/FIM/stream/synth OK.
- Tes: cargo test --workspace lulus (schedule proptest + config tests + vir-text roundtrip).
- Clippy -D warnings lulus; fmt lulus; forbid_stubs lulus.
- Wheel maturin terbangun (Android); import vir OK; text_encode/decode OK.

## Bukti
- `cargo test --workspace` lulus (1 tes properti schedule).
- `cargo clippy --workspace --all-targets -- -D warnings` lulus.
- `cargo fmt --check` lulus.
- `bash scripts/forbid_stubs.sh` lulus.
- `vir smoke --json` mengeluarkan build info.

## Sisa
- F0.2: lint config sudah ada; pemeriksa crate punya tes; cargo-mutants prosedur.
- F0.3: agen OpenCode di .opencode/agents (belum).
- F0.5: smoke.yml masih placeholder (vir-cli smoke belum memanggil pipeline nyata).
- F1.x: belum dimulai.