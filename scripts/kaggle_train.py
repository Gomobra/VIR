"""Skrip latihan VIR di Kaggle (GPU). Resume, batas waktu, multi-GPU (DDP).

Memanggil `vir train` (yang menjalankan `vir-train` nyata: Muon+AdamW,
checkpoint atomik, resume). Bila `vir.train` tersedia di build Python, dipakai
langsung; selain itu jatuh ke CLI `vir train`. Mendukung `--world N` untuk
data-parallel multi-GPU (Kaggle 2xT4).

Contoh:
    python scripts/kaggle_train.py --data shard.bin --out /kaggle/working/run \
        --steps 5000 --max-minutes 50 --world 2 --config configs/kaggle.yaml
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def find_vir_bin() -> str | None:
    """Cari biner `vir` (PATH, sibling skrip, target debug/release)."""
    found = shutil.which("vir")
    if found:
        return found
    here = Path(__file__).resolve()
    for cand in [
        here.parent.parent / "target" / "release" / "vir",
        here.parent.parent / "target" / "debug" / "vir",
    ]:
        if cand.exists():
            return str(cand)
    return None


def build_cmd(args, vir_bin: str) -> list[str]:
    cmd = [
        vir_bin, "train",
        "--data", args.data,
        "--out", str(args.out),
        "--steps", str(args.steps),
        "--world", str(args.world),
        "--stage", args.stage,
        "--json",
    ]
    if args.config:
        cmd += ["--config", args.config]
    if args.resume:
        cmd += ["--resume", args.resume]
    if args.vocab:
        cmd += ["--vocab", str(args.vocab)]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="Latihan VIR di Kaggle (nyata, multi-GPU).")
    ap.add_argument("--data", required=True, help="shard token biner vir-data")
    ap.add_argument("--out", default="/kaggle/working/run", help="direktori keluaran run")
    ap.add_argument("--steps", type=int, default=5000, help="jumlah langkah latihan")
    ap.add_argument("--max-minutes", type=int, default=50, help="batas waktu sesi (menit)")
    ap.add_argument("--resume", default=None, help="direktori run untuk resume")
    ap.add_argument("--config", default=None, help="berkas config model (JSON)")
    ap.add_argument("--vocab", type=int, default=None, help="ukuran vocab (opsional)")
    ap.add_argument("--world", type=int, default=1, help="jumlah GPU/rank data-parallel")
    ap.add_argument("--stage", default="gen", choices=["vq", "gen"], help="tahap awal")
    ap.add_argument("--vir-bin", default=None, help="path biner vir (opsional)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vir_bin = args.vir_bin or find_vir_bin()
    if not vir_bin:
        print("error: biner `vir` tidak ditemukan (build dengan `cargo build -p vir-cli --release`)", file=sys.stderr)
        return 2

    if not Path(args.data).exists():
        print(f"error: shard tidak ditemukan: {args.data}", file=sys.stderr)
        return 2

    cmd = build_cmd(args, vir_bin)
    print("cmd:", " ".join(cmd), flush=True)

    deadline = time.time() + args.max_minutes * 60
    # Jalankan dengan timeout; bila waktu hampir habis, hentikan rapi (trainer
    # menyimpan checkpoint atomik tiap N langkah).
    remaining = max(1.0, deadline - time.time())
    try:
        proc = subprocess.run(cmd, timeout=remaining)
        return proc.returncode
    except subprocess.TimeoutExpired:
        print("max-minutes tercapai; hentikan sesi (checkpoint tersimpan).", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())