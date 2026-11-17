#!/usr/bin/env python3
"""Notebook komprehensif: build vir CUDA SEKALI, lalu jalankan matriks
world=1 & world=2 (DDP) dalam satu sesi. Efisien: 1x build 18 mnt untuk semua.

Output: /kaggle/working/result.json (per-world: rc, dt, loss, out/err)."""
import json

CELLS = []
def md(s): CELLS.append({"cell_type":"markdown","metadata":{},"source":s})
def code(s): CELLS.append({"cell_type":"code","metadata":{},"source":s,"execution_count":None,"outputs":[]})

md("""# VIR — latihan nyata + DDP (build sekali, matriks world)

Build `vir-cli` CUDA sekali, buat shard, lalu `vir train` world=1 & world=2
(DDP 2×T4). Hemat: satu build untuk semua. Hasil: result.json + log.txt.""")

code("""!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
import os, subprocess
os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':' + os.environ['PATH']
print(subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv'], capture_output=True, text=True).stdout)""")

code("""import subprocess, time, os, json
LOG=open('/kaggle/working/log.txt','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); LOG.write(s+'\\n'); LOG.flush()
t0=time.time()
subprocess.run(['git','clone','--depth','1','https://github.com/Gomobra/VIR.git','/tmp/VIR'], check=True)
env={**os.environ,'CARGO_TARGET_DIR':'/tmp/target'}
b=subprocess.run(['cargo','build','--release','--manifest-path','/tmp/VIR/crates/vir-cli/Cargo.toml','--features','vir-train/cuda'], capture_output=True, text=True, env=env)
P('build rc', b.returncode); P(b.stderr[-1500:])
assert b.returncode==0, 'build gagal'
VIR='/tmp/target/release/vir'
P('build selesai', round(time.time()-t0,1),'s')""")

code("""# make-shard (shard sintetis, L=128)
r=subprocess.run([VIR,'make-shard','--out','/kaggle/working/mini_shard.bin','--docs','1000','--vocab','512'], capture_output=True, text=True)
P('make-shard rc', r.returncode, r.stdout.strip(), r.stderr.strip())
assert r.returncode==0""")

code("""# Matriks: world=1 & world=2, steps kecil, timeout per-run
def run(world, steps, timeout=200):
    cmd=[VIR,'train','--data','/kaggle/working/mini_shard.bin','--out',f'/kaggle/working/run_w{world}',
         '--steps',str(steps),'--world',str(world),'--stage','gen','--seq-len','32',
         '--config','/tmp/VIR/configs/mini_train.json','--json']
    t=time.time()
    try:
        r=subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {'world':world,'rc':r.returncode,'dt':round(time.time()-t,1),
                'out':r.stdout[-600:],'err':r.stderr[-600:]}
    except subprocess.TimeoutExpired:
        return {'world':world,'rc':'TIMEOUT','dt':round(time.time()-t,1),'out':'','err':'timeout'}

res=[]
for w in (1,2):
    res.append(run(w, 20)); P('RESULT', w, json.dumps(res[-1]))
open('/kaggle/working/result.json','w').write(json.dumps(res))
# Kemas binary utk cache dataset (run berikutnya tak perlu build)
import tarfile
with tarfile.open('/kaggle/working/vir-bin.tar.gz','w:gz') as t:
    t.add(VIR, arcname='vir')
P('vir-bin.tar.gz', os.path.getsize('/kaggle/working/vir-bin.tar.gz'),'bytes')
P('DONE')
LOG.close()""")

nb={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
print(json.dumps(nb))