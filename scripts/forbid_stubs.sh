#!/usr/bin/env bash
# Gerbang anti-kerangka (PRD F0.2). Menolak:
#   1. pola terlarang R7 (todo!/unimplemented!/NotImplementedError/TODO/FIXME/panic!);
#   2. badan fungsi kosong (stub);
#   3. crate di crates/* tanpa minimal satu tes (F0.2 tugas 3).
#
# Dijalankan di CI (ubuntu, `rg` tersedia) dan bisa di Termux (fallback `grep`).
set -euo pipefail
cd "$(dirname "$0")/.."

fail() {
    echo "forbid_stubs: $*" >&2
    exit 1
}

# `rg` bila ada, selain itu `grep -rEn` (pola ERE yang sama).
if command -v rg >/dev/null 2>&1; then
    search() { rg -n --no-heading "$@"; }
else
    search() { grep -rEn "$@"; }
fi

# 1. Pola terlarang (R7). Jangan cocokkan artefak build.
patterns='todo!|unimplemented!|NotImplementedError|TODO|FIXME|panic!\('
if search "$patterns" crates/ python/ 2>/dev/null | grep -v '/target/'; then
    fail "pola terlarang ditemukan (todo!/unimplemented!/TODO/FIXME/panic!)"
fi

# 2. Badan fungsi kosong: `fn nama(...) -> T {}` (termasuk lintas baris tipe).
#    Baris komentar dan berkas tes dikecualikan (heuristik, bukan parser).
empty_body='fn [A-Za-z0-9_]+\([^)]*\)[^{]*\{\s*\}'
if search "$empty_body" crates/ 2>/dev/null | grep -v '/target/' | grep -v '//'; then
    fail "badan fungsi kosong terdeteksi"
fi

# 3. Setiap crate di crates/* wajib punya minimal satu tes.
#    Allowlist = crate yang sah tanpa tes CPU, dengan alasan tertulis. Crate baru
#    tanpa tes dan tanpa alasan di sini akan menggagalkan CI.
declare -A ALLOWED=(
    [vir-agent]="F4.x: agen RLM; tes menyusul bersama agent-eng"
    [vir-cli]="CLI (bind-eng); jalur pakai diuji lewat tests/e2e dan vir-eval"
    [vir-modal]="Fase 6: multimodal; belum diaktifkan"
    [vir-model]="F1.5/F1.6: generator; gerbang F1.8 dijalankan lewat vir smoke"
    [vir-py]="binding PyO3; diuji lewat python/"
    [vir-tui]="F1.10: TUI; snapshot diuji oleh ui-eng"
)

count_tests() {
    # `|| true` menetralkan pipefail saat grep tak menemukan kecocokan.
    { grep -rhE '#\[(tokio::)?test\]' "$1" --include='*.rs' 2>/dev/null || true; } |
        wc -l | tr -d ' '
}

missing=0
for manifest in crates/*/Cargo.toml; do
    [ -f "$manifest" ] || continue
    crate_dir=$(dirname "$manifest")
    crate=$(basename "$crate_dir")
    n=$(count_tests "$crate_dir")
    if [ "$n" -eq 0 ]; then
        if [ -n "${ALLOWED[$crate]:-}" ]; then
            echo "forbid_stubs: $crate tanpa tes (allowlist: ${ALLOWED[$crate]})"
        else
            echo "forbid_stubs: $crate tidak punya tes dan tidak di-allowlist" >&2
            missing=1
        fi
    fi
done
[ "$missing" -eq 0 ] || fail "ada crate tanpa tes (F0.2 tugas 3)"

echo "forbid_stubs: OK"