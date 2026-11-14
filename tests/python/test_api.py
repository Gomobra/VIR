"""Tes API Python VIR (PRD F3.6).

Dijalankan terhadap modul hasil build (``maturin develop`` atau wheel dari
``maturin build``)::

    pytest tests/python
"""

from __future__ import annotations

import pytest

import vir


def test_info_build():
    assert "VIR" in vir.info()
    assert vir.__version__ == vir.version()


def test_text_roundtrip_byte_level():
    tokens = vir.text.encode("1 + 2 = 3")
    assert tokens
    assert vir.text.decode(tokens) == "1 + 2 = 3"
    assert isinstance(vir.text.special_tokens()[0], str)


def test_tokenizer_train_roundtrip():
    corpus = [f"{a} + {b} = {a + b}" for a in range(6) for b in range(6)]
    tok = vir.text.Tokenizer.train(corpus, vocab_size=300)
    assert tok.vocab_size() >= 260
    assert tok.roundtrip("3 + 4 = 7")
    assert tok.decode(tok.encode("3 + 4 = 7")) == "3 + 4 = 7"


def test_vq_fsq_roundtrip():
    levels = [4, 4, 4]
    codes = [-2, 0, 1]
    index = vir.vq.fsq_index(codes, levels)
    assert vir.vq.fsq_decode(index, levels) == codes
    assert vir.vq.fsq_num_codes(levels) == 64
    assert vir.vq.fsq_bound([0.0, 0.0, 0.0], levels) == [0, 0, 0]
    with pytest.raises(ValueError):
        vir.vq.fsq_index([9], [4])


def test_model_config():
    cfg = vir.model.ModelConfig()
    assert cfg.levels()[-1] == cfg.target_length
    assert cfg.validate() is None
    assert "target_length" in cfg.as_json()
    assert cfg.schedule == "linear"
    assert vir.model.ModelConfig.with_defaults().num_levels == 12
    assert vir.model.available() is False
    with pytest.raises(RuntimeError, match="belum tersedia di build ini"):
        vir.model.load("/tidak/ada/ckpt.safetensors")
    with pytest.raises(FileNotFoundError):
        vir.model.load("/tidak/ada/ckpt.safetensors")


def test_model_config_menolak_target_kecil():
    with pytest.raises(ValueError):
        vir.model.ModelConfig(target_length=0)


def test_train_backend_tidak_tersedia(tmp_path):
    data = tmp_path / "tokens.bin"
    data.write_bytes(b"1 2 3")
    out = tmp_path / "run"
    out.mkdir()

    assert vir.train.available() is False
    assert "Burn" in vir.train.backend()
    with pytest.raises(RuntimeError, match="belum tersedia di build ini"):
        vir.train.run(str(data), str(out))
    with pytest.raises(FileNotFoundError):
        vir.train.run(str(tmp_path / "hilang.bin"), str(out))
    with pytest.raises(ValueError):
        vir.train.run(str(data), str(out), 0)


def test_backend_status():
    status = vir.backend.status()
    assert status["text"] is True
    assert status["mfs"] is True
    assert status["agent"] is True
    assert status["eval"] is True
    assert status["model"] is False
    assert status["train"] is False
    assert "vq" in status
    assert vir.backend.require("text") is None
    with pytest.raises(RuntimeError):
        vir.backend.require("model")


def test_memory_write_read_find():
    mem = vir.mfs.Memory(":memory:")
    version_id = mem.write("/catatan/a", "rust memori vektor")
    assert version_id > 0
    assert mem.read(version_id) == "rust memori vektor"
    assert mem.read_path("/catatan/a") == "rust memori vektor"
    assert mem.current_version("/catatan/a") == version_id
    assert mem.verify(version_id) is True
    assert len(mem.hash_of(version_id)) == 64
    assert mem.history("/catatan/a")[0] == version_id

    hits = mem.find("memori", 5)
    assert hits and hits[0][0] == version_id

    mem.forget(version_id)
    mem.restore(version_id)
    assert mem.read(version_id) == "rust memori vektor"


def test_memory_in_memory_statis():
    mem = vir.mfs.Memory.in_memory()
    version_id = mem.write("/catatan/b", "tanpa basis data")
    assert mem.read(version_id) == "tanpa basis data"


def test_eval_pass_at_k():
    assert vir.eval.pass_at_k(10, 3, 5) == pytest.approx(1.0 - 21.0 / 252.0)
    assert vir.eval.pass_at_k(10, 0, 1) == pytest.approx(0.0)
    assert vir.eval.pass_at_k(10, 10, 1) == pytest.approx(1.0)
    assert vir.eval.pass_at_k(10, 3, 0) == pytest.approx(1.0)


def test_eval_recon_dan_entropi():
    assert vir.eval.recon_accuracy([1, 2, 3], [1, 2, 4]) == pytest.approx(2.0 / 3.0)
    assert vir.eval.recon_accuracy([], []) == 0.0
    assert vir.eval.unigram_entropy([]) == 0.0
    assert vir.eval.unigram_entropy([7, 7, 7]) == pytest.approx(0.0)
    assert vir.eval.unigram_entropy([1, 2]) == pytest.approx(1.0)


def _rlm_bisa_jalan() -> bool:
    try:
        vir.agent.rlm_run(["print('FINAL(ok)')"], "probe", 1, 1)
    except RuntimeError as exc:
        pesan = str(exc)
        return "REPL" not in pesan and "spawn" not in pesan
    return True


RLM_SKIP = pytest.mark.skipif(not _rlm_bisa_jalan(), reason="REPL CPython tidak tersedia")


@RLM_SKIP
def test_rlm_run_skrip():
    out = vir.agent.rlm_run(["print('FINAL(ok)')"], "tugas uji", 2, 2)
    assert out is not None
    assert "FINAL(ok)" in out


@RLM_SKIP
def test_rlm_run_info_dict():
    out = vir.agent.rlm_run_info(["print('FINAL(ok)')"], "tugas uji", 2, 2)
    assert out["final_answer"] is not None
    assert out["final_answer"].startswith("FINAL(ok)")
    assert out["steps"] >= 1
    assert out["calls_used"] >= 1


@RLM_SKIP
def test_rlm_kelas_dengan_mem():
    rlm = vir.agent.RLM(["print('FINAL(selesai)')"], lambda op, payload: "null", 2, 2)
    assert rlm.max_depth == 2
    assert rlm.max_calls == 2
    out = rlm.run("tugas uji")
    assert out["final_answer"] is not None
    assert out["final_answer"].startswith("FINAL(selesai)")