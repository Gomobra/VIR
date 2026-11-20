#!/usr/bin/env python3
"""Notebook Kaggle: bukti konsep CUDA graph vs eager (C++/CUDA murni) + PyTorch.

Compile scripts/cuda_spike/graph_bench.cu (nvcc) → ukur eager vs CUDA graph.
Lalu PyTorch matmul (eager) utk pembanding. Tujuan: buktikan CUDA graph
hilangkan overhead luncuran (dinding Burn ~0.7x PyTorch).

Dipakai: python3 scripts/make_kaggle_cuda_graph.py > scripts/cg/vir_kaggle_cg.ipynb
lalu:    kaggle kernels push -p scripts/cg/
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src,
                  "execution_count": None, "outputs": []})


md("""# VIR Vbeta — bukti konsep: CUDA graph vs eager (C++/CUDA murni)

Compile `scripts/cuda_spike/graph_bench.cu` (nvcc) → ukur eager vs CUDA graph.
PyTorch matmul (eager) sbagai pembanding. Buktikan CUDA graph hilangkan overhead
luncuran (dinding Burn).""")

code("""!nvidia-smi --query-gpu=name,memory.total --format=csv
import subprocess, os
print(subprocess.run(['nvcc','--version'], capture_output=True, text=True).stdout.split('\\n')[-2] if True else '')""")

code("""import subprocess, time, os, json
LOG=open('/kaggle/working/log.txt','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); LOG.write(s+'\\n'); LOG.flush()
subprocess.run(['git','clone','--depth','1','https://github.com/Gomobra/VIR.git','/tmp/VIR'], check=True)
src='/tmp/VIR/scripts/cuda_spike/graph_bench.cu'
b=subprocess.run(['nvcc','-O3','-o','/tmp/graph_bench',src,'-lcublas'], capture_output=True, text=True)
P('nvcc rc', b.returncode, b.stderr[-800:])
assert b.returncode==0, 'compile gagal'
r=subprocess.run(['/tmp/graph_bench'], capture_output=True, text=True)
P('=== C++/CUDA ==='); P(r.stdout); P(r.stderr[-300:])""")

code("""# --- PyTorch matmul (eager) pembanding ---
import torch, time
torch.backends.cuda.matmul.allow_tf32 = True
DEV='cuda'; M=N=K=64; ITERS=500
A=torch.randn(M,K,device=DEV); B=torch.randn(K,N,device=DEV)
for _ in range(20): C=A@B
torch.cuda.synchronize(); t0=time.time()
for _ in range(ITERS): C=A@B
torch.cuda.synchronize(); dt=time.time()-t0
P('=== PyTorch eager ===')
P(f'{ITERS} matmul, {dt*1000:.3f} ms ({dt*1000/ITERS*1000:.1f} us/call)')""")

code("""open('/kaggle/working/result.json','w').write(json.dumps({'log':'see log.txt'}))
P('DONE'); LOG.close()""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
print(json.dumps(nb))