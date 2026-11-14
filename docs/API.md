# API VIR Vbeta

## Python
```python
import vir

vir.build_info()          # -> str
vir.version()             # -> "Vbeta"

vir.text.encode("halo")   # -> list[int]
vir.text.decode([104,97]) # -> "ha"

mem = vir.mfs.Memory("memori.db")
vid = mem.write("/a", "isi")
mem.read(vid)
mem.find("kata", 5)

vir.agent.rlm_run(["print('FINAL(ok)')"], "ctx", 2, 2)
```

## CLI
```
vir smoke [--json]
vir version
vir tui [--follow <run>]
vir mem --path <db> [--write <s>] [--read <id>] [--find <q>]
vir rlm --context <s>
```

## JSON schema
- `vir smoke --json`: `{"status","vocab_size","roundtrip_ok","build"}`
- `vir mem --json`: per-line JSON (skema berversi di `docs/`).