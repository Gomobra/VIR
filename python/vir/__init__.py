"""VIR Vbeta - Variable-depth Iterative Reasoner (Rust core).

Pembungkus tipis: seluruh logika ada di Rust (``vir._core``); modul ini hanya
menyusun namespace yang rapi (PRD F3.6)::

    import vir

    tok = vir.text.Tokenizer.train(corpus, vocab_size=32000)
    cfg = vir.model.ModelConfig()
    mem = vir.mfs.Memory("memori.db")
    out = vir.agent.RLM(lambda prompt: model.generate(prompt)).run(tugas)

Namespace ``model`` dan ``train`` hanya memuat konfigurasi pada build inti;
backend Burn (latihan, checkpoint, VQ penuh) tidak disertakan di sana.
"""

from types import SimpleNamespace

from ._core import (
    Memory,
    ModelConfig,
    RLM,
    Tokenizer,
    backend_status,
    build_info,
    fsq_bound,
    fsq_decode,
    fsq_index,
    fsq_num_codes,
    model_available,
    model_backend,
    model_load,
    pass_at_k,
    recon_accuracy,
    require_backend,
    text_decode,
    text_encode,
    text_special_tokens,
    train_available,
    train_backend,
    train_run,
    unigram_entropy,
    version,
)

__version__ = version()

__all__ = [
    "Memory",
    "ModelConfig",
    "RLM",
    "Tokenizer",
    "__version__",
    "agent",
    "backend",
    "eval",
    "info",
    "mfs",
    "model",
    "text",
    "train",
    "version",
    "vq",
]


def info() -> str:
    """Info build dari Rust."""
    return build_info()


# --- Namespace per lapisan (PRD F3.6) ---
text = SimpleNamespace(
    Tokenizer=Tokenizer,
    train=Tokenizer.train,
    encode=text_encode,
    decode=text_decode,
    special_tokens=text_special_tokens,
)

vq = SimpleNamespace(
    fsq_bound=fsq_bound,
    fsq_index=fsq_index,
    fsq_decode=fsq_decode,
    fsq_num_codes=fsq_num_codes,
    note="VQ penuh (codebook EMA, multiscale, decoder) butuh backend Burn",
)

model = SimpleNamespace(
    ModelConfig=ModelConfig,
    load=model_load,
    available=model_available,
    backend=model_backend,
)

train = SimpleNamespace(
    run=train_run,
    available=train_available,
    backend=train_backend,
)

mfs = SimpleNamespace(Memory=Memory)

agent = SimpleNamespace(
    RLM=RLM,
)

eval = SimpleNamespace(
    pass_at_k=pass_at_k,
    recon_accuracy=recon_accuracy,
    unigram_entropy=unigram_entropy,
)

backend = SimpleNamespace(status=backend_status, require=require_backend)