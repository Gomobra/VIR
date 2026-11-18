#!/usr/bin/env python3
"""Generator notebook Kaggle: batch-scaling `vir train --native`.

Build `vir-cli` CUDA sekali, buat shard, lalu jalankan `vir train --native`
dengan batch 8/16/32/64 (langkah & seq-len tetap). Tujuan: tunjukkan apakah
throughput (tok/s) naik dengan batch — kalau ya berarti beban launch-bound pada
batch kecil (bottleneck berikutnya: fusi kernel / batch besar), bukan compute.

Dipakai: python3 scripts/make_kaggle_batch_scale.py > scripts/batch/vir_kaggle_batch.ipynb
lalu:    kaggle kernels push -p scripts/batch/
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# VIR Vbeta — batch-scaling native (R1: ukur)

Build `vir-cli` CUDA sekali, shard sintetis, lalu `vir train --native` pada
batch 8/16/32/64 (langkah & L tetap). Ukur wall-clock + tok/s nominal
(steps*batch*L / dt). Kalau tok/s naik ~linear dengan batch → beban launch-bound
pada batch kecil (peluang: fusi kernel / batch besar).""")

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
P('build rc', b.returncode); P(b.stderr[-2000:])
assert b.returncode == 0, 'build gagal'
VIR = '/tmp/target/release/vir'
P('build selesai', round(time.time()-t0,1), 's')""")

code("""r = subprocess.run([VIR,'make-shard','--out','/kaggle/working/mini_shard.bin',
                    '--docs','1000','--vocab','512'], capture_output=True, text=True)
P('make-shard rc', r.returncode, r.stdout.strip(), r.stderr.strip())
assert r.returncode == 0""")

code("""CFG = '/tmp/VIR/configs/mini_train.json'
STEPS = 30
SEQLEN = 32

def run(batch, timeout=900):
    cmd = [VIR,'train','--native','--data','/kaggle/working/mini_shard.bin',
           '--out',f'/kaggle/working/run_b{batch}',
           '--steps',str(STEPS),'--batch',str(batch),'--stage','gen',
           '--seq-len',str(SEQLEN),'--config',CFG,'--json']
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        dt = round(time.time()-t, 2)
        tok_s = round(STEPS*batch*SEQLEN/dt, 1) if dt > 0 else 0.0
        return {'batch':batch,'rc':r.returncode,'dt_s':dt,'tok_s':tok_s,
                'out':r.stdout[-300:],'err':r.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {'batch':batch,'rc':'TIMEOUT','dt_s':round(time.time()-t,2),'tok_s':0.0,'out':'','err':'timeout'}

res = []
for b in (8, 16, 32, 64):
    res.append(run(b)); P('BATCH', json.dumps(res[-1]))
open('/kaggle/working/result.json','w').write(json.dumps(res, indent=2))
base = next((x for x in res if x['batch']==8 and x['tok_s']>0), None)
if base:
    for x in res:
        if x['tok_s']>0:
            P(f"scale b{x['batch']}", round(x['tok_s']/base['tok_s'],2), 'x vs b8')
P('DONE')
LOG.close()""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))