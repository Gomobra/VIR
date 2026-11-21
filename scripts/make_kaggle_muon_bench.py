#!/usr/bin/env python3
"""Notebook Kaggle: 6 transformer blocks + Muon (Newton-Schulz) — C++/CUDA vs PyTorch.

Compile scripts/cuda_spike/transformer_muon.cu (6 blok + Muon optimizer manual) → tok/s.
PyTorch 6-block + Muon sama → tok/s. Tujuan: cover komponen "Muon" VIR di C++/CUDA.

Dipakai: python3 scripts/make_kaggle_muon_bench.py > scripts/mb/vir_kaggle_mb.ipynb
lalu:    kaggle kernels push -p scripts/mb/
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# VIR Vbeta — 6 Transformer Blocks + Muon (Newton-Schulz): C++/CUDA vs PyTorch

Compile `scripts/cuda_spike/transformer_muon.cu` (6 blok + Muon optimizer manual,
forward+backward+update) → tok/s. PyTorch 6-block + Muon sama → tok/s.
Buktikan mesin C++/CUDA menang dgn Muon (komponen optimizer VIR).""")

code("""!nvidia-smi --query-gpu=name,memory.total --format=csv""")

code("""import subprocess, time, os, json
LOG = open('/kaggle/working/log.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\\n'); LOG.flush()
subprocess.run(['git','clone','--depth','1','https://github.com/Gomobra/VIR.git','/tmp/VIR'], check=True)
src = '/tmp/VIR/scripts/cuda_spike/transformer_muon.cu'
b = subprocess.run(['nvcc','-O3','-o','/tmp/mb',src,'-lcublas'], capture_output=True, text=True)
P('nvcc rc', b.returncode, b.stderr[-1500:])
assert b.returncode == 0, 'compile gagal'
r = subprocess.run(['/tmp/mb'], capture_output=True, text=True)
P('=== C++/CUDA ==='); P(r.stdout); P(r.stderr[-400:])""")

code("""# --- PyTorch 6-block + Muon (sama) ---
import torch, torch.nn as nn, torch.nn.functional as F, time
torch.backends.cuda.matmul.allow_tf32 = True
DEV='cuda'; B,T,D,HH,HI,NBLK=64,32,64,4,256,6; N=B*T; STEPS=500; WARMUP=20

def newton_schulz(G, steps=5):
    X = G / (G.norm() + 1e-7)
    tr = X.size(0) > X.size(1)
    if tr: X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = 3.4445*X - 4.7750*(A@X) + 2.0315*(A@A@X)
    return X.T if tr else X

class Muon:
    def __init__(s, ps, lr=1e-4, mom=0.95):
        s.ps=list(ps); s.lr=lr; s.mom=mom; s.buf=[torch.zeros_like(p) for p in s.ps]
    @torch.no_grad()
    def step(s):
        for p,buf in zip(s.ps,s.buf):
            if p.grad is None: continue
            buf.mul_(s.mom).add_(p.grad); p.add_(newton_schulz(buf), alpha=-s.lr)

class RMSNorm(nn.Module):
    def __init__(s,d):
        super().__init__(); s.w=nn.Parameter(torch.ones(d))
    def forward(s,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)*s.w

class Blk(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1=RMSNorm(D); s.wqkv=nn.Linear(D,3*D,bias=False); s.wo=nn.Linear(D,D,bias=False)
        s.ln2=RMSNorm(D)
        s.wg=nn.Linear(D,HI,bias=False); s.wu=nn.Linear(D,HI,bias=False); s.wd=nn.Linear(HI,D,bias=False)
    def forward(s,x):
        b,t,_=x.shape
        xn=s.ln1(x); qkv=s.wqkv(xn); q,k,v=qkv.chunk(3,dim=-1)
        q=q.view(b,t,HH,-1).transpose(1,2); k=k.view(b,t,HH,-1).transpose(1,2); v=v.view(b,t,HH,-1).transpose(1,2)
        o=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(b,t,D)
        x=x+s.wo(o); xn2=s.ln2(x)
        return x+s.wd(F.silu(s.wg(xn2))*s.wu(xn2))

class Stack(nn.Module):
    def __init__(s):
        super().__init__(); s.blocks=nn.ModuleList([Blk() for _ in range(NBLK)])
    def forward(s,x):
        N=x.size(0); x=x.view(B,T,D)
        for blk in s.blocks: x=blk(x)
        return x.view(N,D)

def bench(compile_mode):
    torch.manual_seed(0); m=Stack().to(DEV)
    if compile_mode: m=torch.compile(m, mode=compile_mode)
    ps2=[p for p in m.parameters() if p.ndim==2]; ps1=[p for p in m.parameters() if p.ndim!=2]
    om=Muon(ps2); osgd=torch.optim.SGD(ps1, lr=1e-3)
    X=torch.randn(N,D,device=DEV); Tg=torch.randn(N,D,device=DEV)
    def one():
        osgd.zero_grad(set_to_none=True)
        for p in ps2: p.grad=None
        y=m(X); loss=0.5*((y-Tg)**2).mean(); loss.backward(); om.step(); osgd.step(); return loss.item()
    l0=None
    for _ in range(WARMUP): l0=one()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(STEPS): lf=one()
    torch.cuda.synchronize(); dt=time.time()-t0
    return {'dt_s':round(dt,3),'tok_s':round(STEPS*N/dt,1),'loss0':round(l0,3),'lossf':round(lf,3)}

pe=bench(None); P('=== PyTorch 6-block+Muon eager ==='); P(json.dumps(pe))
pc=bench('reduce-overhead'); P('=== PyTorch 6-block+Muon compile-ro ==='); P(json.dumps(pc))
open('/kaggle/working/result.json','w').write(json.dumps({'pt_eager':pe,'pt_compile_ro':pc},indent=2))
P('DONE'); LOG.close()""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))