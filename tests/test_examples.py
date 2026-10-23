"""Run every example on the CPU: each checks its own results against NumPy, and
its in-kernel `if TYPE_CHECKING:` blocks must not affect Warp. (The static
half, `pyrefly check`, covers examples/ through pyrefly.toml.)"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"example_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
