#!/usr/bin/env python3
"""Notebook Kaggle: SATU TRANSFORMER BLOCK (blok VIR) — C++/CUDA murni vs PyTorch.

Compile scripts/cuda_spike/transformer_block.cu (RMSNorm + causal attention + SwiGLU,
forward+backward+SGD manual) → tok/s. PyTorch transformer block sama → tok/s.
Tujuan: buktikan mesin C++/CUDA murni bisa > PyTorch pada blok VIR (bukan MLP).

Dipakai: python3 scripts/make_kaggle_tb_bench.py > scripts/tb/vir_kaggle_tb.ipynb
lalu:    kaggle kernels push -p scripts/tb/
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# VIR Vbeta — Transformer Block (blok VIR): C++/CUDA murni vs PyTorch

Compile `scripts/cuda_spike/transformer_block.cu` (RMSNorm + causal attention +
SwiGLU, forward+backward+SGD manual) → tok/s. PyTorch transformer block sama → tok/s.
Buktikan mesin C++/CUDA murni > PyTorch pada primitif blok VIR (bukan MLP).""")

code("""!nvidia-smi --query-gpu=name,memory.total --format=csv""")

code("""import subprocess, time, os, json
LOG = open('/kaggle/working/log.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\\n'); LOG.flush()
subprocess.run(['git','clone','--depth','1','https://github.com/Gomobra/VIR.git','/tmp/VIR'], check=True)
src = '/tmp/VIR/scripts/cuda_spike/transformer_block.cu'
b = subprocess.run(['nvcc','-O3','-o','/tmp/tb',src,'-lcublas'], capture_output=True, text=True)
P('nvcc rc', b.returncode, b.stderr[-1200:])
assert b.returncode == 0, 'compile gagal'
r = subprocess.run(['/tmp/tb'], capture_output=True, text=True)
P('=== C++/CUDA ==='); P(r.stdout); P(r.stderr[-400:])""")

code("""# --- PyTorch transformer block (sama) ---
import torch, torch.nn as nn, torch.nn.functional as F, time
torch.backends.cuda.matmul.allow_tf32 = True
DEV='cuda'; B,T,D,HH,HI=64,32,64,4,256; N=B*T; STEPS=500; WARMUP=20

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
        N=x.size(0); x=x.view(B,T,D)
        b,t,_=x.shape
        xn=s.ln1(x); qkv=s.wqkv(xn); q,k,v=qkv.chunk(3,dim=-1)
        q=q.view(b,t,HH,-1).transpose(1,2); k=k.view(b,t,HH,-1).transpose(1,2); v=v.view(b,t,HH,-1).transpose(1,2)
        o=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(b,t,D)
        x=x+s.wo(o); xn2=s.ln2(x)
        return (x+s.wd(F.silu(s.wg(xn2))*s.wu(xn2))).view(N,D)

def bench(compile_mode):
    torch.manual_seed(0); m=Blk().to(DEV)
    if compile_mode: m=torch.compile(m, mode=compile_mode)
    opt=torch.optim.SGD(m.parameters(), lr=1e-3)
    X=torch.randn(N,D,device=DEV); Tg=torch.randn(N,D,device=DEV)
    def one():
        opt.zero_grad(set_to_none=True)
        y=m(X); loss=0.5*((y-Tg)**2).mean(); loss.backward(); opt.step(); return loss.item()
    l0=None
    for i in range(WARMUP):
        l0=one()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(STEPS): one()
    torch.cuda.synchronize(); dt=time.time()-t0
    return {'dt_s':round(dt,3),'tok_s':round(STEPS*N/dt,1),'loss0':round(l0,4)}

pe=bench(None); P('=== PyTorch eager ==='); P(json.dumps(pe))
pc=bench('reduce-overhead'); P('=== PyTorch compile-ro ==='); P(json.dumps(pc))
open('/kaggle/working/result.json','w').write(json.dumps({'pt_eager':pe,'pt_compile_ro':pc},indent=2))
P('DONE'); LOG.close()""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))