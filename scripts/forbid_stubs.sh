#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

patterns='todo!|unimplemented!|NotImplementedError|TODO|FIXME|panic!\('
if rg -n "$patterns" crates/ python/ 2>/dev/null; then
    echo "forbid_stubs: pola terlarang ditemukan" >&2
    exit 1
fi

# badan fungsi kosong (heuristik sederhana)
if rg -n '\{\s*\}' crates/ 2>/dev/null | grep -v 'tests' | grep -v '//' ; then
    echo "forbid_stubs: badan fungsi kosong terdeteksi" >&2
    exit 1
fi

echo "forbid_stubs: OK"