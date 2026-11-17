#!/usr/bin/env python3
"""Notebook debug DDP minimal: isolasi apakah `vir train` world=1 atau world=2
yang hang. Pakai timeout subprocess + langkah kecil. Build vir-cli CUDA sekali."""
import json

CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src})
def code(src): CELLS.append({"cell_type":"code","metadata":{},"source":src,"execution_count":None,"outputs":[]})

md("""# VIR — debug `vir train` (isolasi world=1 vs world=2)

Tujuan: cari apakah `vir train --world 1` atau `--world 2` yang hang di GPU.
Tiap run pakai `timeout` subprocess (bukan hang selamanya) + langkah kecil.
Hasil: log.txt + result.json (JSONL progres).""")

code("""!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
import os
os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':' + os.environ['PATH']
import subprocess
print(subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv'], capture_output=True, text=True).stdout)""")

code("""import subprocess, json, time, os
LOG = open('/kaggle/working/log.txt','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); LOG.write(s+'\\n'); LOG.flush()
t0=time.time()
subprocess.run(['git','clone','--depth','1','https://github.com/Gomobra/VIR.git','/tmp/VIR'], check=True)
env={**os.environ,'CARGO_TARGET_DIR':'/tmp/target'}
b=subprocess.run(['cargo','build','--release','--manifest-path','/tmp/VIR/crates/vir-cli/Cargo.toml','--features','vir-train/cuda'], capture_output=True, text=True, env=env)
P('build rc', b.returncode); P(b.stderr[-2000:])
assert b.returncode==0, 'build gagal'
P('build selesai', round(time.time()-t0,1),'s')""")

code("""# make-shard (shard sintetis kecil, L=128)
r=subprocess.run(['/tmp/target/release/vir','make-shard','--out','/kaggle/working/mini_shard.bin','--docs','500','--vocab','256'], capture_output=True, text=True)
P('make-shard rc', r.returncode, r.stdout.strip())
assert r.returncode==0""")

code("""# world=1 dulu, steps kecil, timeout 120s
def run(world, steps, timeout=120):
    cmd=['/tmp/target/release/vir','train','--data','/kaggle/working/mini_shard.bin',
         '--out',f'/kaggle/working/run_w{world}','--steps',str(steps),'--world',str(world),
         '--stage','gen','--seq-len','128','--json']
    t=time.time()
    try:
        r=subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {'world':world,'rc':r.returncode,'dt':round(time.time()-t,1),
                'out':r.stdout[-500:],'err':r.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {'world':world,'rc':'TIMEOUT','dt':round(time.time()-t,1),'out':'','err':'timeout'}
res=[]
for w in (1,2):
    res.append(run(w, 20))
    P('RESULT', w, res[-1])
open('/kaggle/working/result.json','w').write(json.dumps(res))
P('DONE', json.dumps(res))
LOG.close()""")

nb={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
print(json.dumps(nb))