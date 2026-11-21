#!/usr/bin/env python3
"""Notebook Kaggle: test vir-cuda-train (engine C++/CUDA) — compile + run + checkpoint.

Dipakai: python3 scripts/make_kaggle_vct.py > scripts/vct/vir_kaggle_vct.ipynb
lalu:    kaggle kernels push -p scripts/vct/
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# VIR Vbeta — `vir-cuda-train` (engine C++/CUDA) test
Compile, generate token sintetik, latih, verifikasi checkpoint (manifest.json + model.bin).""")

code("""!nvidia-smi --query-gpu=name,memory.total --format=csv""")

code("""import subprocess, time, os, json, numpy as np
LOG = open('/kaggle/working/log.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\\n'); LOG.flush()
print(subprocess.run(['nvcc','--version'], capture_output=True, text=True).stdout.split('\\n')[-2])""")

import base64
_cu = open('scripts/cuda_spike/vir_cuda_train.cu','rb').read()
_b64 = base64.b64encode(_cu).decode()
code("""import base64, subprocess, time, os, json, numpy as np
LOG = open('/kaggle/working/log.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\\n'); LOG.flush()
CU = base64.b64decode('%s')
with open('/tmp/vir_cuda_train.cu','wb') as f:
    f.write(CU); f.flush(); os.fsync(f.fileno())
# verify
with open('/tmp/vir_cuda_train.cu','rb') as f: raw=f.read()
P('wrote bytes', len(raw), 'line167:', repr(raw.split(b'\\n')[166]))
# compile
b = subprocess.run(['nvcc','-O3','-o','/tmp/vct','/tmp/vir_cuda_train.cu','-lcublas'], capture_output=True)
P('nvcc rc', b.returncode, b.stderr.decode()[-2000:])
assert b.returncode == 0, 'compile gagal'""" % _b64)

code("""# generate token sintetik (uint32, vocab 512)
n = 200000
tok = np.random.randint(0, 512, size=n, dtype=np.uint32)
tok.tofile('/tmp/tokens.bin')
P('tokens:', n)

os.makedirs('/tmp/out', exist_ok=True)
for f in ['/tmp/out/manifest.json','/tmp/out/model.bin']:
    if os.path.exists(f): os.remove(f)
r = subprocess.run(['/tmp/vct','--tokens','/tmp/tokens.bin','--out','/tmp/out',
                    '--vocab','512','--d-model','64','--n-heads','4','--n-blocks','6',
                    '--seq-len','32','--batch','64','--steps','100','--lr','1e-4'],
                   capture_output=True, text=True, timeout=600)
P('=== run ==='); P(r.stdout); P(r.stderr[-500:])
P('rc', r.returncode)""")

code("""# verifikasi checkpoint
import os, json, math
mp='/tmp/out/manifest.json'; bp='/tmp/out/model.bin'
P('manifest exists:', os.path.exists(mp), 'model.bin exists:', os.path.exists(bp))
if os.path.exists(mp):
    raw=open(mp).read()
    P('manifest raw:', raw[:500])
if os.path.exists(mp) and os.path.exists(bp):
    m=json.load(open(mp)); P('format:', m.get('format'), 'n_tensors:', len(m['tensors']))
    exp=sum(int(math.prod(t['shape'])) for t in m['tensors'])
    act=os.path.getsize(bp)//4
    P('expected floats:', exp, 'actual floats:', act, 'match:', exp==act)
    P('first:', m['tensors'][0]); P('last:', m['tensors'][-1])
open('/kaggle/working/result.json','w').write(json.dumps({'rc':r.returncode,'out':r.stdout[-500:]}))
P('DONE'); LOG.close()""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))