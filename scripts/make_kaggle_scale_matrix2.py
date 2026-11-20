#!/usr/bin/env python3
"""Notebook Kaggle: head-to-head vir vs PyTorch VIR pada beberapa batch.

Build `vir` sekali, lalu untuk batch ∈ {64, 256, 1024}: `vir train --native
--warmup 3` (parse tok_s) vs PyTorch VIR (model sama, 5-level CE, Muon) pada
batch sama. Cari apakah vir melampaui PyTorch pada batch besar (amortisasi
luncuran). Tujuan: 2–5× di atas PyTorch.

Dipakai: python3 scripts/make_kaggle_scale_matrix2.py > scripts/scale2/vir_kaggle_scale2.ipynb
lalu:    kaggle kernels push -p scripts/scale2/
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# VIR Vbeta — head-to-head vir vs PyTorch VIR per batch (target 2–5×)

Build `vir` sekali. Untuk batch 64/256/1024: `vir train --native --warmup 3`
(parse tok_s) vs PyTorch VIR (model sama). Cari batch di mana rasio > 1 (ideal
2–5×). Bukti objektif menuju tujuan.""")

code("""!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
import os, subprocess
os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':' + os.environ['PATH']
print(subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv'], capture_output=True, text=True).stdout)
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())""")

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
P('build rc', b.returncode); P(b.stderr[-1200:])
assert b.returncode == 0, 'build gagal'
VIR = '/tmp/target/release/vir'
P('vir build', round(time.time()-t0,1), 's')""")

code("""r = subprocess.run([VIR,'make-shard','--out','/kaggle/working/mini_shard.bin',
                    '--docs','3000','--vocab','512'], capture_output=True, text=True)
P('make-shard rc', r.returncode, r.stdout.strip(), r.stderr.strip())
assert r.returncode == 0
CFG = '/tmp/VIR/configs/mini_train.json'
SEQLEN = 32
BATCHES = [64, 256, 1024]
STEPS = 33
WARMUP = 3""")

code("""def run_vir(batch, timeout=1200):
    cmd = [VIR,'train','--native','--warmup',str(WARMUP),
           '--data','/kaggle/working/mini_shard.bin','--out',f'/kaggle/working/run_vir_{batch}',
           '--steps',str(STEPS),'--batch',str(batch),'--stage','gen',
           '--seq-len',str(SEQLEN),'--config',CFG,'--json']
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        dt = round(time.time()-t,2); tok=0.0; ph={}
        try:
            vj = json.loads(r.stdout.strip().splitlines()[-1]); tok=float(vj.get('tok_s',0.0))
            ph = {k: round(float(vj.get(k,0.0)),1) for k in ('fwd_ms','bwd_ms','opt_ms')}
        except Exception as e:
            P('parse vir gagal', batch, e)
        return {'batch':batch,'rc':r.returncode,'dt_s':dt,'tok_s':tok,'phase':ph,'err':r.stderr[-300:]}
    except subprocess.TimeoutExpired:
        return {'batch':batch,'rc':'TIMEOUT','dt_s':round(time.time()-t,2),'tok_s':0.0}

res = {}
for b in BATCHES:
    res[f'vir_{b}'] = run_vir(b); P('VIR', json.dumps(res[f'vir_{b}']))""")

code("""import torch, torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
DEV='cuda'
D=64; HH=4; LEVELS=[2,4,8,16,32]; K=len(LEVELS); CB=256; V=512
PRELUDE=1; CORE=2; CODA=1; LOOPS=2

def rope_tables(t, hd, device):
    inv=1.0/(10000.0**(torch.arange(0,hd,2,device=device).float()/hd))
    ang=torch.outer(torch.arange(t,device=device).float(),inv)
    return torch.cat([ang.cos(),ang.cos()],-1), torch.cat([ang.sin(),ang.sin()],-1)

def apply_rope(x,cos,sin):
    hd=x.size(-1); x1=x[...,:hd//2]; x2=x[...,hd//2:]
    return x*cos[None,None] + torch.cat([-x2,x1],-1)*sin[None,None]

class RMSNorm(nn.Module):
    def __init__(s,d,eps=1e-6):
        super().__init__(); s.w=nn.Parameter(torch.ones(d)); s.eps=eps
    def forward(s,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+s.eps)*s.w

class Blk(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1=RMSNorm(D)
        s.wq=nn.Linear(D,D,bias=False); s.wk=nn.Linear(D,D,bias=False)
        s.wv=nn.Linear(D,D,bias=False); s.wo=nn.Linear(D,D,bias=False)
        s.ln2=RMSNorm(D)
        hid=(D*4//3*2//8)*8
        s.wg=nn.Linear(D,hid,bias=False); s.wu=nn.Linear(D,hid,bias=False); s.wd=nn.Linear(hid,D,bias=False)
    def forward(s,x):
        b,t,_=x.shape; hd=D//HH
        xn=s.ln1(x)
        q=s.wq(xn).view(b,t,HH,hd).transpose(1,2)
        k=s.wk(xn).view(b,t,HH,hd).transpose(1,2)
        v=s.wv(xn).view(b,t,HH,hd).transpose(1,2)
        cos,sin=rope_tables(t,hd,x.device)
        q=apply_rope(q,cos,sin); k=apply_rope(k,cos,sin)
        o=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        o=o.transpose(1,2).reshape(b,t,D)
        x=x+s.wo(o)
        xn2=s.ln2(x)
        x=x+s.wd(F.silu(s.wg(xn2))*s.wu(xn2))
        return x

class VIR(nn.Module):
    def __init__(s):
        super().__init__()
        s.embed=nn.Embedding(V,D)
        s.prelude=nn.ModuleList([Blk() for _ in range(PRELUDE)])
        s.core=nn.ModuleList([Blk() for _ in range(CORE)])
        s.coda=nn.ModuleList([Blk() for _ in range(CODA)])
        s.adapter=nn.Linear(2*D,D,bias=False)
        s.level_in=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(K)])
        s.codes=nn.ParameterList([nn.Parameter(torch.randn(CB,D)*0.02) for _ in range(K)])
        s.heads=nn.ModuleList([nn.Linear(D,V,bias=False) for _ in range(K)])
    def encode(s,z):
        res=z
        for k in range(K):
            idx=torch.cdist(res.reshape(-1,D), s.codes[k]).argmin(-1)
            res=res-s.codes[k][idx].view_as(res)
    def recurrent(s,e):
        x=e
        for blk in s.prelude: x=blk(x)
        for _ in range(max(1,LOOPS)):
            inj=s.adapter(torch.cat([x,e],-1)); x=x+inj
            for blk in s.core: x=blk(x)
        for blk in s.coda: x=blk(x)
        return x
    def forward(s,idx):
        z=s.embed(idx).detach(); s.encode(z)
        out=[]
        for k in range(K):
            L=LEVELS[k]
            feat=s.recurrent(s.level_in[k](z[:, :L, :]))
            out.append(s.heads[k](feat))
        return out

def newton_schulz(G, steps=5):
    X=G/(G.norm()+1e-7); tr=X.size(0)>X.size(1)
    if tr: X=X.T
    for _ in range(steps):
        A=X@X.T; X=3.4445*X-4.7750*(A@X)+2.0315*(A@A@X)
    return X.T if tr else X

class Muon:
    def __init__(s,ps,lr=1e-3,mom=0.95):
        s.ps=list(ps); s.lr=lr; s.mom=mom; s.buf=[torch.zeros_like(p) for p in s.ps]
    @torch.no_grad()
    def step(s):
        for p,buf in zip(s.ps,s.buf):
            if p.grad is None: continue
            buf.mul_(s.mom).add_(p.grad); p.add_(newton_schulz(buf), alpha=-s.lr)

def bench_pt(batch, timeout=1200):
    torch.manual_seed(0); m=VIR().to(DEV)
    ps2=[p for p in m.parameters() if p.ndim==2]; ps1=[p for p in m.parameters() if p.ndim!=2]
    om=Muon(ps2); oa=torch.optim.AdamW(ps1,lr=1e-3)
    def one():
        idx=torch.randint(0,V,(batch,SEQLEN),device=DEV); tgt=torch.randint(0,V,(batch,SEQLEN),device=DEV)
        oa.zero_grad(set_to_none=True)
        for p in ps2: p.grad=None
        logits=m(idx)
        loss=sum(nn.functional.cross_entropy(lg.reshape(-1,V), tgt[:, :LEVELS[k]].reshape(-1)) for k,lg in enumerate(logits))/K
        loss.backward(); om.step(); oa.step()
    for _ in range(3): one()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(STEPS-WARMUP): one()
    torch.cuda.synchronize(); dt=round(time.time()-t0,2)
    return {'batch':batch,'dt_s':dt,'tok_s':round((STEPS-WARMUP)*batch*SEQLEN/dt,1) if dt>0 else 0.0}

for b in BATCHES:
    res[f'pt_{b}'] = bench_pt(b); P('PT', json.dumps(res[f'pt_{b}']))
open('/kaggle/working/result.json','w').write(json.dumps(res, indent=2))
for b in BATCHES:
    v=res.get(f'vir_{b}',{}).get('tok_s',0); p=res.get(f'pt_{b}',{}).get('tok_s',0)
    if v>0 and p>0: P(f'ratio vir/pt b{b}', round(v/p,3), 'x')
P('DONE'); LOG.close()""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))