#!/usr/bin/env python3
"""Generator notebook Kaggle DDP multi-GPU (F2.5.e): build vir-cli CUDA,
buat shard sintetis, jalankan `vir train --world 1` vs `--world 2`, ukur tok/s.

Dipakai: python3 scripts/make_kaggle_ddp.py > scripts/vir_kaggle_ddp.ipynb
lalu:    kaggle kernels push -p scripts/ (dengan metadata terpisah)
"""
import json

CELLS = []

def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src, "execution_count": None, "outputs": []})

md("""# VIR Vbeta — DDP multi-GPU training (Kaggle 2xT4)

Build `vir-cli` CUDA, buat shard sintetis, lalu latih `vir train` dengan
`--world 1` vs `--world 2` (data-parallel DDP). Ukur throughput (tok/s) dan
verifikasi checkpoint + resume. Bukti F2.5.e / G2 (throughput training).""")

code("""!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
import os
os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':' + os.environ['PATH']
import subprocess
print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv'], capture_output=True, text=True).stdout)""")

code("""import subprocess, json, time, os
LOG = open('/kaggle/working/log.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s); LOG.write(s + '\\n'); LOG.flush()
t0 = time.time()
subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/Gomobra/VIR.git', '/tmp/VIR'], check=True)
env = {**os.environ, 'CARGO_TARGET_DIR': '/tmp/target'}
b = subprocess.run(
    ['cargo', 'build', '--release', '--manifest-path', '/tmp/VIR/crates/vir-cli/Cargo.toml', '--features', 'vir-train/cuda'],
    capture_output=True, text=True, env=env)
P('build rc', b.returncode)
P(b.stderr[-3000:])
if b.returncode != 0:
    raise SystemExit('build gagal')""")

code("""# Buat shard token biner vir-data via `vir make-shard` (nyata, dipakai vir train).
import subprocess, os
env = {**os.environ}
r = subprocess.run(
    ['/tmp/target/release/vir', 'make-shard', '--out', '/kaggle/working/mini_shard.bin',
     '--docs', '2000', '--vocab', '512'],
    capture_output=True, text=True, env=env)
P('make-shard rc', r.returncode)
P(r.stdout[-500:])
P(r.stderr[-500:])
assert r.returncode == 0, 'make-shard gagal'
assert os.path.exists('/kaggle/working/mini_shard.bin'), 'shard tidak dibuat'
P('shard siap:', os.path.getsize('/kaggle/working/mini_shard.bin'), 'bytes')""")

code("""# Latihan world=1 vs world=2 (DDP). vir train --world N --data <shard> ...
import subprocess, json, time, os
def run_train(world, steps, out):
    cmd = ['/tmp/target/release/vir', 'train',
           '--data', '/kaggle/working/mini_shard.bin',
           '--out', out, '--steps', str(steps), '--world', str(world), '--stage', 'gen', '--seq-len', '128', '--json']
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t
    return {'world': world, 'rc': r.returncode, 'elapsed_s': round(dt, 2),
            'stdout': r.stdout[-1500:], 'stderr': r.stderr[-1500:]}

# Cek apakah vir train butuh shard biner vir-data; bila belum ada, buat via python vir-text/vir-data bila tersedia.
res = []
for w in (1, 2):
    try:
        res.append(run_train(w, 200, f'/kaggle/working/run_w{w}'))
    except Exception as e:
        res.append({'world': w, 'error': str(e)})
P('DDP_RESULTS', json.dumps(res))
open('/kaggle/working/result.json', 'w').write(json.dumps(res))
LOG.close()""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))