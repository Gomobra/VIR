# PROGRESS — VIR Vbeta

## Selesai (F0–F4 sebagian)
- **F0**: workspace Cargo, `vir-common`, `vir-py`, `vir-cli`, `vir-text`, `vir-data`, `vir-tui`, `vir-mfs`, `vir-sandbox`, `vir-agent`, `vir-eval`, `vir-vq`, `vir-nn`, `vir-model`, `vir-train`; Python API + CLI; CI (ci/wheels/smoke); AGENTS.md; LICENSE/README/CONTRIBUTING/SECURITY; PR template; `.opencode/agents` (14 agen); `configs/`; `data/sources.json`; `docs/adr/ADR-001..008`; `specs/F0.1..F4.5`.
- **F1**: BPE byte-level + ByteTokenizer; shard/pack/FIM/stream/synth; VQ codebook skeleton; NN block skeleton; generator skeleton; loss helpers; `vir smoke` BPE roundtrip; TUI dashboard v1.
- **F2**: recurrent/halting/sampling skeleton; level weight; pass@k.
- **F3**: MFS (SQLite WAL, FTS5, versioning, RRF); sandbox (subprocess); RLM (ScriptedLm + budget); Python bindings.
- **F4**: vir-eval (pass@k, recon, needle); SFT pack; GRPO advantage; kaggle_train.py.
- **F7.1**: fused matmul+argmax kernel (`crates/vir-vq/src/kernel.rs`, feature `fused`) — tidak menulis buffer [N,V] ke HBM; compile check CI lulus.
- **F1.4**: vir-vq lengkap (codebook EMA, multiscale, dropout, decoder, loss, metrics, FSQ) — 50+ tes lulus di CI.
- **F1.5**: vir-nn (RoPE nyata, block-diagonal + prefix-LM mask, attention, RMSNorm, SwiGLU, leakage tests) — tes lulus di CI.
- **F1.6**: vir-model (generator: codebook per level, proyeksi rekonstruksi, prelude/core/coda, head per level, loss R6, generate level-by-level) — tes lulus di CI.
- **F1.7**: vir-train (Muon Newton-Schulz, JSONL logger, pass@k tak bias, GRPO advantage, level_weight) — tes lulus di CI.
- **F2.1**: vir-model recurrent depth (prelude/core/coda, injection adapter, r sampling Poisson lognormal) — tes lulus di CI.
- **F2.2/F2.3**: halting (Fixed/Entropy) + entropy + block-AR loop — tes lulus di CI.
- **F2.4**: CFG logit + MBR select (similarity-based) — tes lulus di CI.
- **F3.4**: RLM recursive `llm_query` dengan budget kedalaman/panggilan — tes lulus di CI.

## Bukti
- CI semua job hijau (fmt, clippy, nextest, forbid_stubs, deny, vir-vq, vir-nn, vir-model, vir-train, vir-eval).
- Wheel manylinux x86_64 terbangun; `import vir` OK.
- Kaggle G2: Burn CUDA **0.49× PyTorch** (forward VQ) → LULUS borderline (ADR-001).
- Kaggle G2 smoke + bench: COMPLETE (log.txt + result.json).

## Sisa (belum selesai)
- **F1.4**: VQ penuh (EMA update, dead-code restart, nested dropout, decoder, loss, FSQ).
- **F1.5**: RoPE nyata, mask block-diagonal/prefix-LM, leakage tests.
- **F1.6**: generator penuh (embedding per level, head, loss/generate, sampling).
- **F1.7**: trainer 2 tahap, Muon, checkpoint/resume, JSONL.
- **F1.8**: irisan penuh (BPE→VQ→generator→decoder→sampling, overfit 8 sampel).
- **F1.10/F3.8**: TUI penuh (JSONL tail, chart, snapshot 3 ukuran, 9 layar).
- **F3.1**: Landlock + seccomp + rlimit.
- **F3.2**: REPL stateful.
- **F3.3**: vector index, dedup, konsolidasi, ls/tree/link/tag/diff.
- **F3.4**: RLM rekursif + llm_query + mem + FINAL.
- **F3.6**: API lengkap (`vir.text.Tokenizer`, `vir.Model`, dll).
- **F3.7**: tests/e2e + tests/python.
- **F4**: SFT training, verifier selection, RL loop, code eval, lm-eval adapter.
- **F5/F6/F7**: konteks panjang, multimodal, optimasi.
- Pecah file sesuai tree 6.2 (sebagian masih `lib.rs` tunggal).

## Catatan
- Crate Burn (`vir-vq`, `vir-nn`, `vir-model`, `vir-train`, `vir-eval`) standalone `[workspace]` karena Burn tidak compile di Termux (gemm-f16). Dibangun di CI x86_64.