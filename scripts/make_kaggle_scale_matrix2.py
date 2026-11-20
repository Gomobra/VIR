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
DEV='cuda'; D, HH, LAYERS, V = 64, 4, 6, 512
LEVELS = [2, 4, 8, 16, 32]

class Blk(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1=nn.LayerNorm(D); s.attn=nn.MultiheadAttention(D,HH,batch_first=True); s.ln2=nn.LayerNorm(D)
        s.mlp=nn.Sequential(nn.Linear(D,4*D), nn.GELU(), nn.Linear(4*D,D))
    def forward(s,x,mask):
        a,_=s.attn(x,x,x,attn_mask=mask,need_weights=False); x=s.ln1(x+a); return s.ln2(x+s.mlp(x))

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok=nn.Embedding(V,D); s.pos=nn.Embedding(SEQLEN,D)
        s.blocks=nn.ModuleList([Blk() for _ in range(LAYERS)]); s.ln=nn.LayerNorm(D)
        s.heads=nn.ModuleList([nn.Linear(D,V,bias=False) for _ in range(len(LEVELS))])
    def forward(s,idx):
        T=idx.size(1); x=s.tok(idx)+s.pos(torch.arange(T,device=idx.device))[None]
        mask=torch.triu(torch.full((T,T),float('-inf'),device=idx.device),1)
        for b in s.blocks: x=b(x,mask)
        x=s.ln(x); return [h(x) for h in s.heads]

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
    torch.manual_seed(0); m=Net().to(DEV)
    ps2=[p for p in m.parameters() if p.ndim==2]; ps1=[p for p in m.parameters() if p.ndim!=2]
    om=Muon(ps2); oa=torch.optim.AdamW(ps1,lr=1e-3)
    def one():
        idx=torch.randint(0,V,(batch,SEQLEN),device=DEV); tgt=torch.randint(0,V,(batch,SEQLEN),device=DEV)
        oa.zero_grad(set_to_none=True)
        for p in ps2: p.grad=None
        logits=m(idx)
        loss=sum(nn.functional.cross_entropy(lg.reshape(-1,V), tgt[:, :LEVELS[k]].reshape(-1)) for k,lg in enumerate(logits))/LEVELS
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