#!/usr/bin/env python3
"""Notebook Kaggle: inferensi — vir generate (NCM paralel) vs PyTorch AR.

Build `vir` sekali, latih model mini (vir train --native), lalu:
  1) `vir generate --ckpt <run>` (NCM: 1 lintasan paralel utk L token)
  2) PyTorch AR transformer (dekoder-only, KV-cache) generate L token
Ukur tok/s (warmup). Tujuan: keunggulan non-AR (di atas PyTorch) di inferensi.

Dipakai: python3 scripts/make_kaggle_gen_bench.py > scripts/gen/vir_kaggle_gen.ipynb
lalu:    kaggle kernels push -p scripts/gen/
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# VIR Vbeta — inferensi: vir generate (NCM paralel) vs PyTorch AR

Build `vir` sekali, latih model mini, lalu `vir generate` (NCM: seluruh L token
dalam 1 lintasan paralel) vs PyTorch AR (KV-cache, L lintasan). Ukur tok/s.""")

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
b = subprocess.run(['cargo','build','--release','--manifest-path','/tmp/VIR/crates/vir-cli/Cargo.toml',
     '--features','vir-train/cuda'], capture_output=True, text=True, env=env)
P('build rc', b.returncode); P(b.stderr[-1000:])
assert b.returncode == 0, 'build gagal'
VIR = '/tmp/target/release/vir'
P('vir build', round(time.time()-t0,1), 's')""")

code("""# Latih model mini (untuk generate).
r = subprocess.run([VIR,'make-shard','--out','/kaggle/working/mini_shard.bin','--docs','1000','--vocab','512'],
                   capture_output=True, text=True)
P('make-shard rc', r.returncode, r.stdout.strip()[:120])
r = subprocess.run([VIR,'train','--native','--data','/kaggle/working/mini_shard.bin',
                    '--out','/kaggle/working/run','--steps','20','--stage','gen','--seq-len','32',
                    '--config','/tmp/VIR/configs/mini_train.json','--json'],
                   capture_output=True, text=True, timeout=600)
P('train rc', r.returncode, r.stdout.strip()[-150:])
assert r.returncode == 0, 'train gagal'""")

code("""# --- 1) vir generate (NCM paralel) ---
def run_vir_gen():
    t=time.time()
    r=subprocess.run([VIR,'generate','--ckpt','/kaggle/working/run','--batch','1','--json'],
                     capture_output=True, text=True, timeout=300)
    dt=round(time.time()-t,3)
    try:
        vj=json.loads(r.stdout.strip().splitlines()[-1])
        return {'rc':r.returncode,'dt_s':dt,'tok_s':vj.get('tok_s',0.0),'tokens':vj.get('tokens',0),
                'out':r.stdout[-200:],'err':r.stderr[-300:]}
    except Exception as e:
        return {'rc':r.returncode,'dt_s':dt,'tok_s':0.0,'err':r.stderr[-300:],'parse':str(e)}

res = {'vir_gen': run_vir_gen()}; P('VIR_GEN', json.dumps(res['vir_gen']))""")

code("""# --- 2) PyTorch AR (KV-cache) ---
import torch, torch.nn as nn, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
DEV='cuda'; D=64; HH=4; V=512; LAYERS=6; SEQLEN=32

class Blk(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1=nn.LayerNorm(D); s.wq=nn.Linear(D,D,bias=False); s.wk=nn.Linear(D,D,bias=False)
        s.wv=nn.Linear(D,D,bias=False); s.wo=nn.Linear(D,D,bias=False)
        s.ln2=nn.LayerNorm(D)
        hid=(D*4//3*2//8)*8
        s.wg=nn.Linear(D,hid,bias=False); s.wu=nn.Linear(D,hid,bias=False); s.wd=nn.Linear(hid,D,bias=False)
    def forward(s,x):
        b,t,_=x.shape
        xn=s.ln1(x)
        q=s.wq(xn).view(b,t,HH,-1).transpose(1,2)
        k=s.wk(xn).view(b,t,HH,-1).transpose(1,2)
        v=s.wv(xn).view(b,t,HH,-1).transpose(1,2)
        o=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        o=o.transpose(1,2).reshape(b,t,D)
        x=x+s.wo(o); xn2=s.ln2(x)
        return x+s.wd(F.silu(s.wg(xn2))*s.wu(xn2))

class AR(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok=nn.Embedding(V,D); s.pos=nn.Embedding(SEQLEN,D)
        s.blocks=nn.ModuleList([Blk() for _ in range(LAYERS)])
        s.ln=nn.LayerNorm(D); s.head=nn.Linear(D,V,bias=False)
    def forward(s,idx):
        T=idx.size(1); x=s.tok(idx)+s.pos(torch.arange(T,device=idx.device))[None]
        for b in s.blocks: x=b(x)
        return s.head(s.ln(x))

@torch.no_grad()
def gen_ar(model, L):
    idx=torch.zeros(1,1,dtype=torch.long,device=DEV)
    for _ in range(L-1):
        logits=model(idx)[:, -1, :]
        idx=torch.cat([idx, logits.argmax(-1,keepdim=True)], 1)
    return idx

def bench_ar():
    torch.manual_seed(0); m=AR().to(DEV).eval()
    for _ in range(2): gen_ar(m, SEQLEN)   # warmup
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(10): gen_ar(m, SEQLEN)
    torch.cuda.synchronize(); dt=round(time.time()-t0,3)
    return {'dt_s':dt,'tok_s':round(10*SEQLEN/dt,1) if dt>0 else 0.0}

res['pt_ar'] = bench_ar(); P('PT_AR', json.dumps(res['pt_ar']))
open('/kaggle/working/result.json','w').write(json.dumps(res, indent=2))
v=res['vir_gen']['tok_s']; p=res['pt_ar']['tok_s']
if v>0 and p>0: P('ratio vir/pt_ar', round(v/p,3), 'x')
P('DONE'); LOG.close()""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))