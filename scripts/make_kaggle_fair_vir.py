#!/usr/bin/env python3
"""Notebook Kaggle: perbandingan ADIL apple-to-apple — vir (Burn) vs PyTorch VIR.

Model SAMA di kedua framework: multiscale VQ (residual, codebook 256), recurrent
depth (prelude1 + 2×(core2+inject) + coda1), 5 level (ℓ=[2,4,8,16,32]), RMSNorm,
RoPE, attention (causal), SwiGLU MLP, head per level, Muon (Newton-Schulz 5 iter)
+ AdamW. Ukur tok/s steady-state (warmup).

Dipakai: python3 scripts/make_kaggle_fair_vir.py > scripts/fair/vir_kaggle_fair.ipynb
lalu:    kaggle kernels push -p scripts/fair/
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# VIR Vbeta — adil apple-to-apple: vir (Burn) vs PyTorch VIR

Model identik: multiscale VQ (codebook 256) + recurrent depth (prelude1 +
2×(core2+inject) + coda1) + 5 level (ℓ=[2,4,8,16,32]) + RMSNorm + RoPE +
attention causal + SwiGLU + Muon (Newton-Schulz) + AdamW. Ukur tok/s
steady-state (warmup). Tujuan: apakah mesin kita (Burn) mengalahkan PyTorch
pada model & kerja yang SAMA.""")

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
                    '--docs','1000','--vocab','512'], capture_output=True, text=True)
P('make-shard rc', r.returncode, r.stdout.strip(), r.stderr.strip())
assert r.returncode == 0
CFG = '/tmp/VIR/configs/mini_train.json'
BATCH = 64; SEQLEN = 32; STEPS = 103; WARMUP = 5""")

code("""# --- 1) vir train --native: fp32 vs bf16 (Device::configure tensor core) ---
CFG = '/tmp/VIR/configs/mini_train.json'
STEPS = 33
WARMUP = 5

def run_vir(dtype):
    t = time.time()
    r = subprocess.run([VIR,'train','--native','--warmup',str(WARMUP),'--dtype',dtype,
                        '--data','/kaggle/working/mini_shard.bin',
                        '--out',f'/kaggle/working/run_vir_{dtype}',
                        '--steps',str(STEPS),'--batch',str(BATCH),'--stage','gen',
                        '--seq-len',str(SEQLEN),'--config',CFG,'--json'],
                       capture_output=True, text=True, timeout=1200)
    dt = round(time.time()-t,2); tok=0.0; ph={}
    try:
        vj = json.loads(r.stdout.strip().splitlines()[-1]); tok=float(vj.get('tok_s',0.0))
        ph = {k: round(float(vj.get(k,0.0)),1) for k in ('fwd_ms','bwd_ms','opt_ms')}
    except Exception as e:
        P('parse vir gagal', dtype, e)
    return {'rc': r.returncode, 'tok_s': tok, 'dt_s': dt, 'phase': ph,
            'err': r.stderr[-300:]}

res = {}
res['vir_fp32'] = run_vir('fp32'); P('VIR_FP32', json.dumps(res['vir_fp32']))
res['vir_fp16'] = run_vir('fp16'); P('VIR_FP16', json.dumps(res['vir_fp16']))""")

code("""# --- 2) PyTorch VIR: model SAMA, RMSNorm+RoPE+SDPA+SwiGLU+Muon+AdamW ---
import torch, torch.nn as nn, torch.nn.functional as F
DEV='cuda'
D=64; H=4; LEVELS=[2,4,8,16,32]; K=len(LEVELS); CB=256; VOCAB=512
PRELUDE=1; CORE=2; CODA=1; LOOPS=2

def rope_tables(t, hd, device):
    inv = 1.0/(10000.0**(torch.arange(0,hd,2,device=device).float()/hd))
    ang = torch.outer(torch.arange(t,device=device).float(), inv)   # [t, hd/2]
    return torch.cat([ang.cos(),ang.cos()],-1), torch.cat([ang.sin(),ang.sin()],-1)

def apply_rope(x, cos, sin):            # x [b,H,t,hd]
    hd=x.size(-1); x1=x[...,:hd//2]; x2=x[...,hd//2:]
    rot=torch.cat([-x2,x1],-1)
    return x*cos[None,None] + rot*sin[None,None]

class RMSNorm(nn.Module):
    def __init__(s,d,eps=1e-6):
        super().__init__(); s.w=nn.Parameter(torch.ones(d)); s.eps=eps
    def forward(s,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+s.eps)*s.w

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1=RMSNorm(D)
        s.wq=nn.Linear(D,D,bias=False); s.wk=nn.Linear(D,D,bias=False)
        s.wv=nn.Linear(D,D,bias=False); s.wo=nn.Linear(D,D,bias=False)
        s.ln2=RMSNorm(D)
        hid=(D*4//3*2//8)*8
        s.wg=nn.Linear(D,hid,bias=False); s.wu=nn.Linear(D,hid,bias=False); s.wd=nn.Linear(hid,D,bias=False)
    def forward(s,x):
        b,t,_=x.shape; hd=D//H
        xn=s.ln1(x)
        q=s.wq(xn).view(b,t,H,hd).transpose(1,2)
        k=s.wk(xn).view(b,t,H,hd).transpose(1,2)
        v=s.wv(xn).view(b,t,H,hd).transpose(1,2)
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
        s.embed=nn.Embedding(VOCAB,D)
        s.prelude=nn.ModuleList([Block() for _ in range(PRELUDE)])
        s.core=nn.ModuleList([Block() for _ in range(CORE)])
        s.coda=nn.ModuleList([Block() for _ in range(CODA)])
        s.adapter=nn.Linear(2*D,D,bias=False)
        s.level_in=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(K)])
        s.codes=nn.ParameterList([nn.Parameter(torch.randn(CB,D)*0.02) for _ in range(K)])
        s.heads=nn.ModuleList([nn.Linear(D,VOCAB,bias=False) for _ in range(K)])
    def encode(s, z):
        res=z
        for k in range(K):
            cb=s.codes[k]
            idx=torch.cdist(res.reshape(-1,D), cb).argmin(-1)
            res=res-cb[idx].view_as(res)
    def recurrent(s, e):
        x=e
        for blk in s.prelude: x=blk(x)
        for _ in range(max(1,LOOPS)):
            inj=s.adapter(torch.cat([x,e],-1)); x=x+inj
            for blk in s.core: x=blk(x)
        for blk in s.coda: x=blk(x)
        return x
    def forward(s, idx):
        z=s.embed(idx).detach()
        s.encode(z)
        out=[]
        for k in range(K):
            L=LEVELS[k]
            feat=s.recurrent(s.level_in[k](z[:, :L, :]))
            out.append(s.heads[k](feat))
        return out

def newton_schulz(G, steps=5):
    X=G/(G.norm()+1e-7)
    tr = X.size(0)>X.size(1)
    if tr: X=X.T
    for _ in range(steps):
        A=X@X.T
        X=3.4445*X - 4.7750*(A@X) + 2.0315*(A@A@X)
    return X.T if tr else X

class Muon:
    def __init__(s, ps, lr=1e-3, mom=0.95):
        s.ps=list(ps); s.lr=lr; s.mom=mom; s.buf=[torch.zeros_like(p) for p in s.ps]
    @torch.no_grad()
    def step(s):
        for p,buf in zip(s.ps,s.buf):
            if p.grad is None: continue
            buf.mul_(s.mom).add_(p.grad)
            p.add_(newton_schulz(buf), alpha=-s.lr)

def bench_pt(use_fp16=False):
    torch.manual_seed(0)
    m=VIR().to(DEV)
    ps2=[p for p in m.parameters() if p.ndim==2]
    ps1=[p for p in m.parameters() if p.ndim!=2]
    opt_m=Muon(ps2); opt_a=torch.optim.AdamW(ps1, lr=1e-3)
    tgt=torch.randint(0,VOCAB,(BATCH,SEQLEN),device=DEV)
    def one():
        idx=torch.randint(0,VOCAB,(BATCH,SEQLEN),device=DEV)
        opt_a.zero_grad(set_to_none=True)
        for p in ps2: p.grad=None
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_fp16):
            logits=m(idx)
            loss=sum(F.cross_entropy(lg.reshape(-1,VOCAB), tgt[:, :LEVELS[k]].reshape(-1))
                     for k,lg in enumerate(logits))/K
        loss.backward(); opt_m.step(); opt_a.step()
        return float(loss)
    for _ in range(WARMUP): one()
    torch.cuda.synchronize(); t0=time.time(); last=0.0
    for _ in range(STEPS-WARMUP): last=one()
    torch.cuda.synchronize(); dt=round(time.time()-t0,2)
    return {'tok_s': round((STEPS-WARMUP)*BATCH*SEQLEN/dt,1) if dt>0 else 0.0,
            'dt_s': dt, 'precision': 'fp16' if use_fp16 else 'fp32', 'loss': round(last,4)}

res['pt_vir'] = bench_pt(False); P('PT_VIR', json.dumps(res['pt_vir']))
res['pt_vir_fp16'] = bench_pt(True); P('PT_VIR_FP16', json.dumps(res['pt_vir_fp16']))
open('/kaggle/working/result.json','w').write(json.dumps(res, indent=2))
for vk, pk in (('vir_fp32','pt_vir'), ('vir_fp16','pt_vir_fp16'), ('vir_fp16','pt_vir')):
    v=res[vk]['tok_s']; p=res[pk]['tok_s']
    if v>0 and p>0: P(f'ratio {vk}/{pk}', round(v/p,3), 'x')
P('DONE'); LOG.close()""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))