#!/usr/bin/env python3
"""Notebook Kaggle: MLP training — C++/CUDA murni vs PyTorch (bukti mesin sendiri).

Compile scripts/cuda_spike/mlp_train.cu (nvcc) → tok/s. PyTorch MLP sama → tok/s.
Tujuan: buktikan mesin C++/CUDA murni bisa > PyTorch (hilangkan dispatch Python).

Dipakai: python3 scripts/make_kaggle_mlp_bench.py > scripts/mlpb/vir_kaggle_mlpb.ipynb
lalu:    kaggle kernels push -p scripts/mlpb/
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# VIR Vbeta — MLP training: C++/CUDA murni vs PyTorch

Compile `scripts/cuda_spike/mlp_train.cu` (cuBLAS, forward+backward+SGD) → tok/s.
PyTorch MLP sama → tok/s. Buktikan mesin C++/CUDA murni > PyTorch (dispatch Python).""")

code("""!nvidia-smi --query-gpu=name,memory.total --format=csv""")

code("""import subprocess, time, os, json
LOG = open('/kaggle/working/log.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\\n'); LOG.flush()
subprocess.run(['git','clone','--depth','1','https://github.com/Gomobra/VIR.git','/tmp/VIR'], check=True)
src = '/tmp/VIR/scripts/cuda_spike/mlp_train.cu'
b = subprocess.run(['nvcc','-O3','-o','/tmp/mlp_train',src,'-lcublas'], capture_output=True, text=True)
P('nvcc rc', b.returncode, b.stderr[-800:])
assert b.returncode == 0, 'compile gagal'
r = subprocess.run(['/tmp/mlp_train'], capture_output=True, text=True)
P('=== C++/CUDA ==='); P(r.stdout); P(r.stderr[-300:])""")

code("""# --- PyTorch MLP (sama) ---
import torch, torch.nn as nn, time
torch.backends.cuda.matmul.allow_tf32 = True
DEV='cuda'; B,L,D,H=64,32,64,256; N=B*L; STEPS=200; WARMUP=20
torch.manual_seed(0)
model=nn.Sequential(nn.Linear(D,H), nn.GELU(), nn.Linear(H,D)).to(DEV)
opt=torch.optim.SGD(model.parameters(), lr=1e-3)
X=torch.randn(N,D,device=DEV); T=torch.randn(N,D,device=DEV)
def one():
    opt.zero_grad(set_to_none=True)
    y=model(X); loss=0.5*((y-T)**2).mean()
    loss.backward(); opt.step()
for _ in range(WARMUP): one()
torch.cuda.synchronize(); t0=time.time()
for _ in range(STEPS): one()
torch.cuda.synchronize(); dt=time.time()-t0
P('=== PyTorch ===')
P(f'{STEPS} steps, {dt*1000:.2f} ms, {dt*1000/STEPS*1000:.1f} us/step, {STEPS*N/dt:.1f} tok/s')""")

code("""open('/kaggle/working/result.json','w').write(json.dumps({'log':'see log.txt'}))
P('DONE'); LOG.close()""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))