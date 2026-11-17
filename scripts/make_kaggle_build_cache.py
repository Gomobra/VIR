#!/usr/bin/env python3
"""Builder cache: build `vir` (CUDA) sekali, simpan sebagai Kaggle Dataset
`spesialassme/vir-bin-cuda`, supaya run training berikutnya TIDAK build ulang.

Output: /kaggle/working/vir-bin.tar.gz (binary + libs). Publish manual:
    kaggle datasets create -p <dir> --dir-mode tar
atau via notebook `kaggle datasets create`.
"""
import json

CELLS = []
def md(s): CELLS.append({"cell_type":"markdown","metadata":{},"source":s})
def code(s): CELLS.append({"cell_type":"code","metadata":{},"source":s,"execution_count":None,"outputs":[]})

md("""# VIR — build cache `vir` CUDA (sekali, untuk dipakai ulang)

Build `vir-cli` dengan backend CUDA, lalu simpan binary ke `/kaggle/working/`.
Setelah selesai, output (`vir-bin.tar.gz`) dipublikasikan sebagai dataset
`spesialassme/vir-bin-cuda` agar notebook latihan tidak perlu build 18 menit lagi.""")

code("""!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
import os, subprocess
os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':' + os.environ['PATH']
print(subprocess.run(['nvidia-smi','--query-gpu=index,name','--format=csv'], capture_output=True, text=True).stdout)""")

code("""import subprocess, time, os
t0=time.time()
subprocess.run(['git','clone','--depth','1','https://github.com/Gomobra/VIR.git','/tmp/VIR'], check=True)
env={**os.environ,'CARGO_TARGET_DIR':'/tmp/target'}
b=subprocess.run(['cargo','build','--release','--manifest-path','/tmp/VIR/crates/vir-cli/Cargo.toml','--features','vir-train/cuda'], capture_output=True, text=True, env=env)
print('build rc', b.returncode); print(b.stderr[-1500:])
assert b.returncode==0
print('build selesai', round(time.time()-t0,1),'s')""")

code("""# Kemas binary + cek
import subprocess, os, tarfile
binp='/tmp/target/release/vir'
assert os.path.exists(binp), 'binary tidak ada'
# statis-ish: cek dynamic deps
print(subprocess.run(['file', binp], capture_output=True, text=True).stdout)
print(subprocess.run(['ldd', binp], capture_output=True, text=True).stdout[:1000])
# tar.gz
os.makedirs('/kaggle/working', exist_ok=True)
with tarfile.open('/kaggle/working/vir-bin.tar.gz','w:gz') as t:
    t.add(binp, arcname='vir')
print('vir-bin.tar.gz', os.path.getsize('/kaggle/working/vir-bin.tar.gz'),'bytes')
print('OK')""")

nb={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
print(json.dumps(nb))