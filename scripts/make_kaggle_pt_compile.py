#!/usr/bin/env python3
"""Notebook Kaggle: PyTorch VIR — eager vs torch.compile vs bf16 (target PyTorch terbaik).

Tidak perlu build vir. Ukur tok/s model VIR (level-aware, 5-level, Muon) pada:
  1) eager fp32  2) torch.compile fp32  3) torch.compile + bf16 autocast
Tujuan: tahu performa PyTorch TERBAIK (compile) sebagai target "di atas PyTorch",
dan seberapa besar gap vs eager (yang jadi baseline sebelumnya).

Dipakai: python3 scripts/make_kaggle_pt_compile.py > scripts/ptc/vir_kaggle_ptc.ipynb
lalu:    kaggle kernels push -p scripts/ptc/
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# VIR Vbeta — PyTorch VIR: eager vs torch.compile vs bf16

Model VIR (5-level multiscale + recurrent + Muon). Ukur tok/s untuk eager fp32,
torch.compile fp32, torch.compile+bf16. Target: performa PyTorch TERBAIK.""")

code("""import torch, torch.nn as nn, torch.nn.functional as F, time, json
torch.backends.cuda.matmul.allow_tf32 = True
DEV='cuda'
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print(torch.cuda.get_device_name(0))""")

code("""D=64; HH=4; LEVELS=[2,4,8,16,32]; K=len(LEVELS); CB=256; VOCAB=512
PRELUDE=1; CORE=2; CODA=1; LOOPS=2
BATCH=64; SEQLEN=32; STEPS=33; WARMUP=5

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
    def forward(s,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+s.eps)*s.w

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
        cos,sin=rope_tables(t,hd,x.device); q=apply_rope(q,cos,sin); k=apply_rope(k,cos,sin)
        o=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(b,t,D)
        x=x+s.wo(o); xn2=s.ln2(x)
        return x+s.wd(F.silu(s.wg(xn2))*s.wu(xn2))

class VIR(nn.Module):
    def __init__(s):
        super().__init__()
        s.embed=nn.Embedding(VOCAB,D)
        s.prelude=nn.ModuleList([Blk() for _ in range(PRELUDE)])
        s.core=nn.ModuleList([Blk() for _ in range(CORE)])
        s.coda=nn.ModuleList([Blk() for _ in range(CODA)])
        s.adapter=nn.Linear(2*D,D,bias=False)
        s.level_in=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(K)])
        s.codes=nn.ParameterList([nn.Parameter(torch.randn(CB,D)*0.02) for _ in range(K)])
        s.heads=nn.ModuleList([nn.Linear(D,VOCAB,bias=False) for _ in range(K)])
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
        return [s.heads[k](s.recurrent(s.level_in[k](z[:, :LEVELS[k], :]))) for k in range(K)]

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

def bench(use_bf16, compile_mode, tag):
    torch.manual_seed(0); m=VIR().to(DEV)
    if compile_mode: m=torch.compile(m, mode=compile_mode)
    ps2=[p for p in m.parameters() if p.ndim==2]; ps1=[p for p in m.parameters() if p.ndim!=2]
    om=Muon(ps2); oa=torch.optim.AdamW(ps1,lr=1e-3)
    tgt=torch.randint(0,VOCAB,(BATCH,SEQLEN),device=DEV)
    def one():
        idx=torch.randint(0,VOCAB,(BATCH,SEQLEN),device=DEV)
        oa.zero_grad(set_to_none=True)
        for p in ps2: p.grad=None
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            logits=m(idx)
            loss=sum(F.cross_entropy(lg.reshape(-1,VOCAB), tgt[:,:LEVELS[k]].reshape(-1)) for k,lg in enumerate(logits))/K
        loss.backward(); om.step(); oa.step()
    for _ in range(3): one()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(STEPS-WARMUP): one()
    torch.cuda.synchronize(); dt=round(time.time()-t0,2)
    return {'tag':tag,'dt_s':dt,'tok_s':round((STEPS-WARMUP)*BATCH*SEQLEN/dt,1) if dt>0 else 0.0}

res = {}
res['pt_eager']=bench(False,None,'eager_fp32'); print('EAGER', res['pt_eager'])
res['pt_compile']=bench(False,"default",'compile_fp32'); print('COMPILE', res['pt_compile'])
res['pt_compile_ro']=bench(False,"reduce-overhead",'compile_ro'); print('COMPILE_RO', res['pt_compile_ro'])
res['pt_compile_bf16']=bench(True,"default",'compile_bf16'); print('COMPILE_BF16', res['pt_compile_bf16'])
open('/kaggle/working/result.json','w').write(json.dumps(res, indent=2))
print('DONE')""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))