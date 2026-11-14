#!/usr/bin/env python3
"""Pastikan ``python/vir/_core.pyi`` sinkron dengan modul ``vir._core``.

PRD F3.6: stub tipe harus selalu cocok dengan modul hasil build. Skrip ini
membandingkan nama tingkat atas dan anggota kelas antara stub (dibaca dengan
``ast``) dan modul yang benar-benar diimpor, lalu keluar dengan kode 1 bila
berbeda. Dipakai di CI dan oleh ``tests/python/test_stub_sync.py``.

Pakai::

    python crates/vir-py/check_stub_sync.py [--root DIR] [--module vir._core]
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODULE = "vir._core"
STUB_PATH = Path("python") / "vir" / "_core.pyi"


def stub_names(path: Path) -> set[str]:
    """Nama tingkat atas plus anggota kelas yang dideklarasikan stub."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(f"{node.name}.{sub.name}")
    return names


def module_names(module: ModuleType) -> set[str]:
    """Nama publik modul (``dir`` tanpa prefiks garis bawah)."""
    return {name for name in dir(module) if not name.startswith("_")}


def compare_names(declared: set[str], actual: set[str]) -> list[str]:
    """Beda dua himpunan nama dalam bentuk pesan."""
    problems: list[str] = []
    missing = sorted(declared - actual)
    extra = sorted(actual - declared)
    if missing:
        problems.append(f"modul tidak punya nama yang dideklarasikan stub: {missing}")
    if extra:
        problems.append(f"modul punya nama yang tidak ada di stub: {extra}")
    return problems


def compare_members(module: ModuleType, declared: set[str]) -> list[str]:
    """Bandingkan anggota kelas stub dengan kelas sebenarnya."""
    problems: list[str] = []
    for name in sorted(declared):
        if "." not in name:
            continue
        cls_name, member = name.split(".", 1)
        cls = getattr(module, cls_name, None)
        if cls is None:
            problems.append(f"kelas '{cls_name}' ada di stub tetapi tidak di modul")
            continue
        if not hasattr(cls, member):
            problems.append(f"{cls_name} tidak punya anggota '{member}' yang ada di stub")
    return problems


def compare_signatures(module: ModuleType, stub: Path) -> list[str]:
    """Bandingkan nama parameter fungsi tingkat atas bila bisa diintrospeksi."""
    problems: list[str] = []
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        func = getattr(module, node.name, None)
        if not callable(func):
            continue
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            continue
        want = [arg.arg for arg in node.args.args]
        got = list(signature.parameters)
        if want != got:
            problems.append(f"tanda tangan '{node.name}' berbeda: stub={want} modul={got}")
    return problems


def load_module(root: Path, name: str) -> ModuleType:
    """Impor modul, dengan ``root/python`` sebagai sumber bila perlu."""
    try:
        return importlib.import_module(name)
    except ImportError:
        source = str(root / "python")
        if source not in sys.path:
            sys.path.insert(0, source)
        spec = importlib.util.find_spec(name)
        if spec is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


def compare_module(root: Path, name: str = DEFAULT_MODULE) -> list[str]:
    """Semua ketidaksinkronan antara stub dan modul `name`."""
    stub = root / STUB_PATH
    module = load_module(root, name)
    declared = stub_names(stub)
    problems = compare_names(declared, module_names(module))
    problems += compare_members(module, declared)
    problems += compare_signatures(module, stub)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="akar repo VIR")
    parser.add_argument("--module", default=DEFAULT_MODULE, help="nama modul inti")
    args = parser.parse_args(argv)

    problems = compare_module(Path(args.root), args.module)
    if problems:
        for problem in problems:
            print(f"stub tidak sinkron: {problem}", file=sys.stderr)
        return 1
    print(f"stub sinkron: {args.module}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())