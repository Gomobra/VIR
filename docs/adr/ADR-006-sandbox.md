# ADR-006: Sandbox (Landlock + seccomp)

## Konteks
Industri: Landlock + seccomp-bpf (Codex/Cursor), bubblewrap (Claude Code), Sandlock (ASPLOS 2026).

## Keputusan
- Landlock (filesystem) + seccomp-bpf allowlist + rlimit + timeout + kill process group.
- Deteksi ABI runtime; ABI 6 (kernel 6.12+) untuk IPC/signal; fallback eksplisit.
- Jaringan mati default.

## Status
Belum diimplementasikan (F3.1). Saat ini hanya subprocess + timeout.