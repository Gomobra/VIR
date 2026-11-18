#!/usr/bin/env python3
"""Generator notebook Kaggle: bandingkan `vir train --native` vs PyTorch nyata.

Build `vir-cli` CUDA sekali, buat shard, lalu:
  1) `vir train --native` (mesin kita, Burn/CubeCL, fp32)
  2) PyTorch transformer setara (d_model/heads/layers ~ generator VIR, 5-level
     CE seperti tugas NCM), fp32 & bf16-autocast, warmup lalu ukur tok/s.
Konfigurasi sama: batch 64, L=32, vocab 512, 30 langkah, T4 yang sama.

Dipakai: python3 scripts/make_kaggle_vs_pytorch.py > scripts/vsp/vir_kaggle_vs_pytorch.ipynb
lalu:    kaggle kernels push -p scripts/vsp/
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# VIR Vbeta — `vir train --native` vs PyTorch (nyata, T4 yang sama)

Build `vir-cli` CUDA sekali → `vir train --native` (Burn/CubeCL, fp32) dan
**PyTorch** transformer setara (d_model 64, 4 head, ~6 blok, 5-level CE seperti
tugas NCM VIR) fp32 & bf16-autocast. Konfigurasi identik: batch 64, L=32,
vocab 512, 30 langkah. Ukur tok/s (warmup lalu timed).""")

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
P('build rc', b.returncode); P(b.stderr[-1500:])
assert b.returncode == 0, 'build gagal'
VIR = '/tmp/target/release/vir'
P('vir build', round(time.time()-t0,1), 's')""")

code("""r = subprocess.run([VIR,'make-shard','--out','/kaggle/working/mini_shard.bin',
                    '--docs','1000','--vocab','512'], capture_output=True, text=True)
P('make-shard rc', r.returncode, r.stdout.strip(), r.stderr.strip())
assert r.returncode == 0""")

code("""# --- 1) Mesin kita: vir train --native (fp32), batch 64, 30 langkah ---
CFG = '/tmp/VIR/configs/mini_train.json'
STEPS = 30
BATCH = 64
SEQLEN = 32
t = time.time()
r = subprocess.run([VIR,'train','--native','--data','/kaggle/working/mini_shard.bin',
                    '--out','/kaggle/working/run_vir','--steps',str(STEPS),
                    '--batch',str(BATCH),'--stage','gen','--seq-len',str(SEQLEN),
                    '--config',CFG,'--json'], capture_output=True, text=True, timeout=900)
vir_dt = round(time.time()-t, 2)
vir_tok = round(STEPS*BATCH*SEQLEN/vir_dt, 1) if vir_dt>0 else 0.0
res = {'vir_native': {'rc': r.returncode, 'dt_s': vir_dt, 'tok_s': vir_tok,
                      'precision': 'fp32', 'out': r.stdout[-200:], 'err': r.stderr[-300:]}}
P('VIR_NATIVE', json.dumps(res['vir_native']))""")

code("""# --- 2) PyTorch: transformer setara, 5-level CE (seperti NCM VIR) ---
import torch, torch.nn as nn, math
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEV = 'cuda'
D, H, LAYERS, V, LEVELS = 64, 4, 6, 512, 5

class Blk(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1 = nn.LayerNorm(D); s.attn = nn.MultiheadAttention(D, H, batch_first=True)
        s.ln2 = nn.LayerNorm(D)
        s.mlp = nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))
    def forward(s, x, mask):
        a, _ = s.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = s.ln1(x + a); x = s.ln2(x + s.mlp(x)); return x

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, D); s.pos = nn.Embedding(SEQLEN, D)
        s.blocks = nn.ModuleList([Blk() for _ in range(LAYERS)])
        s.ln = nn.LayerNorm(D)
        s.heads = nn.ModuleList([nn.Linear(D, V, bias=False) for _ in range(LEVELS)])
    def forward(s, idx):
        T = idx.size(1)
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))[None]
        mask = torch.triu(torch.full((T, T), float('-inf'), device=idx.device), 1)
        for b in s.blocks: x = b(x, mask)
        x = s.ln(x)
        return [h(x) for h in s.heads]

def bench(use_bf16, tag):
    torch.manual_seed(0)
    m = Net().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    def one():
        idx = torch.randint(0, V, (BATCH, SEQLEN), device=DEV)
        tgt = torch.randint(0, V, (BATCH, SEQLEN), device=DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            logits = m(idx)
            loss = sum(nn.functional.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1))
                       for lg in logits) / LEVELS
        loss.backward(); opt.step()
        return float(loss)
    for _ in range(3): one()          # warmup
    torch.cuda.synchronize(); t0 = time.time()
    last = 0.0
    for _ in range(STEPS): last = one()
    torch.cuda.synchronize(); dt = round(time.time()-t0, 2)
    tok = round(STEPS*BATCH*SEQLEN/dt, 1) if dt>0 else 0.0
    return {'rc': 0, 'dt_s': dt, 'tok_s': tok,
            'precision': 'bf16' if use_bf16 else 'fp32', 'loss': round(last, 4)}

res['pytorch_fp32'] = bench(False, 'pt_fp32'); P('PYTORCH_FP32', json.dumps(res['pytorch_fp32']))
res['pytorch_bf16'] = bench(True, 'pt_bf16');  P('PYTORCH_BF16', json.dumps(res['pytorch_bf16']))
open('/kaggle/working/result.json','w').write(json.dumps(res, indent=2))

v = res['vir_native']['tok_s']
for k in ('pytorch_fp32', 'pytorch_bf16'):
    p = res[k]['tok_s']
    if v>0 and p>0:
        P('ratio vir/pt_'+k.split('_')[1], round(v/p, 3), 'x')
P('DONE'); LOG.close()""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))