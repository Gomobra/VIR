"""Stub ``_core.pyi`` harus sinkron dengan modul (PRD F3.6)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "crates" / "vir-py" / "check_stub_sync.py"


def load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_stub_sync", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_stub_sync"] = module
    spec.loader.exec_module(module)
    return module


def test_stub_modul_tanpa_beda():
    problems = load_checker().compare_module(ROOT)
    assert problems == []


def test_skrip_cli_sukses():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(ROOT)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr