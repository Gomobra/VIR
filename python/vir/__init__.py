"""VIR Vbeta - Variable-depth Iterative Reasoner (Rust core)."""

from vir._core import build_info, version, text_encode, text_decode, PyMemory, rlm_run

__version__ = version()

def info() -> str:
    return build_info()

Memory = PyMemory