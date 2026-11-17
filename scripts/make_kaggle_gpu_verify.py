#!/usr/bin/env python3
"""Notebook komprehensif verifikasi GPU (sekali build 18 mnt):
1. vir smoke --json di CUDA (G1 smoke pada GPU)
2. vir train world=1 & world=2 (DDP 2xT4), 200 langkah mini -> tok/s (G2)
3. cache vir-bin.tar.gz (dataset)
Output: /kaggle/working/result.json"""
import json

CELLS=[]
def md(s): CELLS.append({"cell_type":"markdown","metadata":{},"source":s})
def code(s): CELLS.append({"cell_type":"code","metadata":{},"source":s,"execution_count":None,"outputs":[]})

md("""# VIR — verifikasi GPU komprehensif (build sekali)

Build `vir-cli` CUDA sekali → uji `vir smoke` (G1 pada GPU) → `vir train`
world=1 & world=2 (DDP 2×T4, 200 langkah mini) → ukur **tok/s** (G2) → cache
binary. Semua dalam satu sesi, hemat waktu.""")

code("""!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
import os, subprocess
os.environ['PATH']=os.path.expanduser('~/.cargo/bin')+':'+os.environ['PATH']
print(subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv'],capture_output=True,text=True).stdout)""")

code("""import subprocess, time, os, json
LOG=open('/kaggle/working/log.txt','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); LOG.write(s+'\\n'); LOG.flush()
t0=time.time()
subprocess.run(['git','clone','--depth','1','https://github.com/Gomobra/VIR.git','/tmp/VIR'],check=True)
env={**os.environ,'CARGO_TARGET_DIR':'/tmp/target'}
b=subprocess.run(['cargo','build','--release','--manifest-path','/tmp/VIR/crates/vir-cli/Cargo.toml','--features','vir-train/cuda'],capture_output=True,text=True,env=env)
P('build rc',b.returncode); P(b.stderr[-1200:]); assert b.returncode==0
VIR='/tmp/target/release/vir'; P('build selesai',round(time.time()-t0,1),'s')""")

code("""# 1. vir smoke --json di CUDA (G1 smoke pada GPU)
r=subprocess.run([VIR,'smoke','--json'],capture_output=True,text=True,timeout=300)
P('smoke rc',r.returncode); P(r.stdout[-800:]); P(r.stderr[-400:])""")

code("""# make-shard (L=32 utk config mini)
r=subprocess.run([VIR,'make-shard','--out','/kaggle/working/mini_shard.bin','--docs','1000','--vocab','512'],capture_output=True,text=True)
P('make-shard rc',r.returncode,r.stdout.strip())
assert r.returncode==0""")

code("""# 2. vir train world=1 & world=2 (mini, 200 langkah) -> tok/s
def run(world,steps,timeout=300):
    cmd=[VIR,'train','--data','/kaggle/working/mini_shard.bin','--out',f'/kaggle/working/run_w{world}',
         '--steps',str(steps),'--world',str(world),'--stage','gen','--seq-len','32',
         '--config','/tmp/VIR/configs/mini_train.json','--json']
    t=time.time()
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
        dt=time.time()-t
        # tok/s = steps*batch(8)*L(32) / dt
        toks=steps*8*32
        return {'world':world,'rc':r.returncode,'dt':round(dt,1),
                'tok_s':round(toks/dt,1) if dt>0 else 0,
                'out':r.stdout[-300:],'err':r.stderr[-300:]}
    except subprocess.TimeoutExpired:
        return {'world':world,'rc':'TIMEOUT'}

res=[run(1,200), run(2,200)]
open('/kaggle/working/result.json','w').write(json.dumps(res))
for x in res: P('RESULT', json.dumps(x))
# cache
import tarfile
with tarfile.open('/kaggle/working/vir-bin.tar.gz','w:gz') as t: t.add(VIR, arcname='vir')
P('DONE')
LOG.close()""")

nb={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
print(json.dumps(nb))