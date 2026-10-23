"""The lint for runtime rules (skills/.../verify_port.py) reports exactly the
lines marked `LINT: <LEVEL>` in its fixture, and nothing in the real examples."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "lint_fixtures"

spec = importlib.util.spec_from_file_location(
    "verify_port", ROOT / "skills" / "add-shape-types-to-warp-project" / "verify_port.py"
)
assert spec is not None and spec.loader is not None
verify_port = importlib.util.module_from_spec(spec)
sys.modules["verify_port"] = verify_port  # its dataclasses look the module up
spec.loader.exec_module(verify_port)


def test_fixture_findings() -> None:
    path = FIXTURES / "violations.py"
    expected = {
        (number, match.group(1))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if (match := re.search(r"LINT: (ERROR|WARN|INFO)", line))
    }
    actual = {(f.line, f.level) for f in verify_port.check(path)}
    assert actual == expected


def test_clean_files() -> None:
    for path in [ROOT / "examples" / "tile_mlp.py", ROOT / "tests" / "test_tiles.py"]:
        errors = [f for f in verify_port.check(path) if f.level == "ERROR"]
        assert errors == [], errors
