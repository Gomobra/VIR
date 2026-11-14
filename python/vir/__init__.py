"""VIR Vbeta - Variable-depth Iterative Reasoner (Rust core)."""

from vir._core import build_info, version, text_encode, text_decode, PyMemory, rlm_run
from types import SimpleNamespace

__version__ = version()

def info() -> str:
    return build_info()

# Namespace tipis: vir.text, vir.mfs, vir.agent
text = SimpleNamespace(
    encode=text_encode,
    decode=text_decode,
)

mfs = SimpleNamespace(
    Memory=PyMemory,
)

agent = SimpleNamespace(
    rlm_run=rlm_run,
)

Memory = PyMemory