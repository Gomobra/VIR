#!/usr/bin/env python3
"""Generator notebook Kaggle: benchmark `vir train` roundtrip vs `--native`.

Build `vir-cli` CUDA SEKALI, buat shard sintetis, lalu jalankan dua run dengan
konfigurasi identik (mini_train.json, dim64 K5 L32):
  1) roundtrip  : `vir train`           (VirStepFn + Trainer, host<->GPU)
  2) native     : `vir train --native`  (NativeTrainer, param tetap di GPU)
Ukur wall-clock + turunkan tok/s nominal (steps*batch*L / dt). Bukti R1.

Dipakai: python3 scripts/make_kaggle_native_bench.py > scripts/vir_kaggle_native.ipynb
lalu:    kaggle kernels push -p scripts/ (metadata terpisah)
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# VIR Vbeta — benchmark latihan: roundtrip vs GPU-native

Build `vir-cli` CUDA sekali, buat shard sintetis, lalu jalankan `vir train`
(roundtrip host<->GPU) dan `vir train --native` (param tetap di GPU) dengan
konfigurasi mini identik. Ukur wall-clock & turunkan tok/s nominal (R1: ukur
dulu). Hasil: `result.json` + `log.txt`.""")

code("""!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
import os, subprocess
os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':' + os.environ['PATH']
print(subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv'], capture_output=True, text=True).stdout)""")

code("""import subprocess, time, os, json
LOG = open('/kaggle/working/log.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\\n'); LOG.flush()
t0 = time.time()
subprocess.run(['git','clone','--depth','1','https://github.com/Gomobra/VIR.git','/tmp/VIR'], check=True)
env = {**os.environ, 'CARGO_TARGET_DIR': '/tmp/target'}
b = subprocess.run(
    ['cargo','build','--release','--manifest-path','/tmp/VIR/crates/vir-cli/Cargo.toml',
     '--features','vir-train/cuda'],
    capture_output=True, text=True, env=env)
P('build rc', b.returncode); P(b.stderr[-2500:])
assert b.returncode == 0, 'build gagal'
VIR = '/tmp/target/release/vir'
P('build selesai', round(time.time()-t0,1), 's')""")

code("""# Shard token sintetis (L=32, vocab 512) — sama utk kedua run.
r = subprocess.run([VIR,'make-shard','--out','/kaggle/working/mini_shard.bin',
                    '--docs','1000','--vocab','512'], capture_output=True, text=True)
P('make-shard rc', r.returncode, r.stdout.strip(), r.stderr.strip())
assert r.returncode == 0""")

code("""# Konfigurasi identik; hanya --native yang beda.
CFG = '/tmp/VIR/configs/mini_train.json'
STEPS = 40
BATCH = 8
SEQLEN = 32

def run(native, tag, timeout=900):
    cmd = [VIR,'train','--data','/kaggle/working/mini_shard.bin',
           '--out',f'/kaggle/working/run_{tag}',
           '--steps',str(STEPS),'--stage','gen','--seq-len',str(SEQLEN),
           '--config',CFG,'--json']
    if native:
        cmd.append('--native')
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        dt = round(time.time()-t, 2)
        # tok/s nominal: steps * batch * seqlen / dt
        tok_s = round(STEPS*BATCH*SEQLEN/dt, 1) if dt > 0 else 0.0
        return {'mode':tag,'rc':r.returncode,'dt_s':dt,'tok_s':tok_s,
                'out':r.stdout[-400:],'err':r.stderr[-600:]}
    except subprocess.TimeoutExpired:
        dt = round(time.time()-t, 2)
        return {'mode':tag,'rc':'TIMEOUT','dt_s':dt,'tok_s':0.0,'out':'','err':'timeout'}

res = []
res.append(run(False, 'roundtrip')); P('ROUNDTRIP', json.dumps(res[-1]))
res.append(run(True, 'native'));     P('NATIVE', json.dumps(res[-1]))
open('/kaggle/working/result.json','w').write(json.dumps(res, indent=2))
if res[0]['tok_s'] > 0 and res[1]['tok_s'] > 0:
    P('speedup native/roundtrip', round(res[1]['tok_s']/res[0]['tok_s'], 3), 'x')
P('DONE')
LOG.close()""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))