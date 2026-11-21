#!/usr/bin/env python3
"""Notebook Kaggle: 6 TRANSFORMER BLOCKS bertumpuk (recurrent depth VIR) — C++/CUDA vs PyTorch.

Compile scripts/cuda_spike/transformer_stack.cu (6 blok: RMSNorm+attn+SwiGLU,
forward+backward+SGD) → tok/s. PyTorch 6-block sama → tok/s.
Tujuan: buktikan mesin C++/CUDA menang pada KEDALAMAN VIR (6 blok), bukan 1 blok.

Dipakai: python3 scripts/make_kaggle_stack_bench.py > scripts/st/vir_kaggle_st.ipynb
lalu:    kaggle kernels push -p scripts/st/
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# VIR Vbeta — 6 Transformer Blocks bertumpuk (recurrent depth VIR): C++/CUDA vs PyTorch

Compile `scripts/cuda_spike/transformer_stack.cu` (6 blok: RMSNorm + causal attention +
SwiGLU, forward+backward+SGD manual) → tok/s. PyTorch 6-block sama → tok/s.
Buktikan mesin C++/CUDA menang pada KEDALAMAN VIR (6 blok).""")

code("""!nvidia-smi --query-gpu=name,memory.total --format=csv""")

code("""import subprocess, time, os, json
LOG = open('/kaggle/working/log.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\\n'); LOG.flush()
subprocess.run(['git','clone','--depth','1','https://github.com/Gomobra/VIR.git','/tmp/VIR'], check=True)
src = '/tmp/VIR/scripts/cuda_spike/transformer_stack.cu'
b = subprocess.run(['nvcc','-O3','-o','/tmp/st',src,'-lcublas'], capture_output=True, text=True)
P('nvcc rc', b.returncode, b.stderr[-1500:])
assert b.returncode == 0, 'compile gagal'
r = subprocess.run(['/tmp/st'], capture_output=True, text=True)
P('=== C++/CUDA ==='); P(r.stdout); P(r.stderr[-400:])""")

code("""# --- PyTorch 6-block (sama) ---
import torch, torch.nn as nn, torch.nn.functional as F, time
torch.backends.cuda.matmul.allow_tf32 = True
DEV='cuda'; B,T,D,HH,HI,NBLK=64,32,64,4,256,6; N=B*T; STEPS=500; WARMUP=20

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
    opt=torch.optim.SGD(m.parameters(), lr=1e-3)
    X=torch.randn(N,D,device=DEV); Tg=torch.randn(N,D,device=DEV)
    def one():
        opt.zero_grad(set_to_none=True)
        y=m(X); loss=0.5*((y-Tg)**2).mean(); loss.backward(); opt.step()
    for _ in range(WARMUP): one()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(STEPS): one()
    torch.cuda.synchronize(); dt=time.time()-t0
    return {'dt_s':round(dt,3),'tok_s':round(STEPS*N/dt,1)}

pe=bench(None); P('=== PyTorch 6-block eager ==='); P(json.dumps(pe))
pc=bench('reduce-overhead'); P('=== PyTorch 6-block compile-ro ==='); P(json.dumps(pc))
open('/kaggle/working/result.json','w').write(json.dumps({'pt_eager':pe,'pt_compile_ro':pc},indent=2))
P('DONE'); LOG.close()""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))