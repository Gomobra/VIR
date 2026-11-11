"""Skrip latihan VIR di Kaggle (GPU). Resume dari checkpoint, batas waktu sesi."""

import argparse
import json
import os
import time
from pathlib import Path

import vir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-minutes", type=int, default=55)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    print("build:", vir.build_info())
    print("config:", args.config)
    print("out:", out)

    deadline = time.time() + args.max_minutes * 60
    step = 0
    # Placeholder loop: irisan vertikal mini (BPE + VQ smoke) sebagai bukti jalan.
    # Latihan penuh menyusul dengan vir.train (Burn) di CI/GPU.
    while time.time() < deadline:
        step += 1
        if step % 100 == 0:
            print(json.dumps({"step": step, "t": time.time()}))
        time.sleep(0.01)
        if step >= 1000:
            break

    # simpan checkpoint meta
    (ckpt_dir / "last.json").write_text(json.dumps({"step": step}))
    print("done step", step)


if __name__ == "__main__":
    main()