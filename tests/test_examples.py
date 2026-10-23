"""Run the examples on the CPU: their in-kernel `if TYPE_CHECKING:` shape
assertions must not affect Warp, and their results must match NumPy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_tile_mlp() -> None:
    spec = importlib.util.spec_from_file_location("tile_mlp", EXAMPLES / "tile_mlp.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
