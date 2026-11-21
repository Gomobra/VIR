#!/usr/bin/env python3
"""Notebook Kaggle: FULL VIR (VQ encode + 6-block + 5-level multiscale heads + Muon)
— C++/CUDA vs PyTorch. Tujuan: full VIR di C++/CUDA, ukur vs PyTorch VIR.

Dipakai: python3 scripts/make_kaggle_vir_bench.py > scripts/vb/vir_kaggle_vb.ipynb
lalu:    kaggle kernels push -p scripts/vb/
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# VIR Vbeta — FULL VIR: VQ encode + 6-block + 5-level multiscale heads + Muon
C++/CUDA vs PyTorch. Buktikan full VIR (VQ + multiscale + recurrent + Muon) > PyTorch.""")

code("""!nvidia-smi --query-gpu=name,memory.total --format=csv""")

code("""import subprocess, time, os, json
LOG = open('/kaggle/working/log.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\\n'); LOG.flush()
subprocess.run(['git','clone','--depth','1','https://github.com/Gomobra/VIR.git','/tmp/VIR'], check=True)
src = '/tmp/VIR/scripts/cuda_spike/transformer_vir.cu'
b = subprocess.run(['nvcc','-O3','-o','/tmp/vb',src,'-lcublas'], capture_output=True, text=True)
P('nvcc rc', b.returncode, b.stderr[-2000:])
assert b.returncode == 0, 'compile gagal'
r = subprocess.run(['/tmp/vb'], capture_output=True, text=True)
P('=== C++/CUDA ==='); P(r.stdout); P(r.stderr[-400:])""")

code("""# --- PyTorch VIR (sama: VQ + 6-block + 5-head + Muon) ---
import torch, torch.nn as nn, torch.nn.functional as F, time
torch.backends.cuda.matmul.allow_tf32 = True
DEV='cuda'; B,T,D,HH,HI,NBLK=64,32,64,4,256,6; N=B*T
K=5; LEV=[2,4,8,16,32]; V=512; CB=256
STEPS=500; WARMUP=20

def newton_schulz(G, steps=5):
    X=G/(G.norm()+1e-7); tr=X.size(0)>X.size(1)
    if tr: X=X.T
    for _ in range(steps):
        A=X@X.T; X=3.4445*X-4.7750*(A@X)+2.0315*(A@A@X)
    return X.T if tr else X

class Muon:
    def __init__(s,ps,lr=1e-4,mom=0.95):
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

class VIR(nn.Module):
    def __init__(s):
        super().__init__()
        s.blocks=nn.ModuleList([Blk() for _ in range(NBLK)])
        s.cbs=nn.ParameterList([nn.Parameter(torch.randn(CB,D)*0.05) for _ in range(K)])
        s.heads=nn.ModuleList([nn.Linear(D,V,bias=False) for _ in range(K)])
    def forward(s, z, tg):
        res=z
        for k in range(K):
            d=torch.cdist(res.reshape(-1,D), s.cbs[k]); idx=d.argmin(-1)
            res=res-s.cbs[k][idx].view_as(res)
        x=res.view(B,T,D)
        for blk in s.blocks: x=blk(x)
        loss=0.0
        for k in range(K):
            lk=LEV[k]; lg=s.heads[k](x[:,:lk,:])
            loss=loss+F.cross_entropy(lg.reshape(-1,V), tg[:,:lk].reshape(-1))
        return loss/K

def bench(compile_mode):
    torch.manual_seed(0); m=VIR().to(DEV)
    if compile_mode: m=torch.compile(m, mode=compile_mode)
    ps2=[p for p in m.parameters() if p.ndim==2]; ps1=[p for p in m.parameters() if p.ndim!=2]
    om=Muon(ps2); osgd=torch.optim.SGD(ps1,lr=1e-3)
    z=torch.randn(B,T,D,device=DEV)
    tg=torch.randint(0,V,(B,T),device=DEV)
    def one():
        osgd.zero_grad(set_to_none=True)
        for p in ps2: p.grad=None
        loss=m(z,tg); loss.backward(); om.step(); osgd.step(); return loss.item()
    l0=None
    for _ in range(WARMUP): l0=one()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(STEPS): lf=one()
    torch.cuda.synchronize(); dt=time.time()-t0
    return {'dt_s':round(dt,3),'tok_s':round(STEPS*N/dt,1),'loss0':round(l0,3),'lossf':round(lf,3)}

pe=bench(None); P('=== PyTorch VIR eager ==='); P(json.dumps(pe))
pc=bench('reduce-overhead'); P('=== PyTorch VIR compile-ro ==='); P(json.dumps(pc))
open('/kaggle/working/result.json','w').write(json.dumps({'pt_eager':pe,'pt_compile_ro':pc},indent=2))
P('DONE'); LOG.close()""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))