"""The static half of the suite: Pyrefly over the stubs, the tests and the examples."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYREFLY = ROOT / ".venv" / "bin" / "pyrefly"


def run(*command: str | Path) -> None:
    result = subprocess.run([str(c) for c in command], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_generated_stubs_are_fresh() -> None:
    run(sys.executable, "tools/generate_stubs.py", "--check")


def test_stale_stubs_are_reported(tmp_path: Path) -> None:
    stubs = tmp_path / "warp-stubs"
    shutil.copytree(ROOT / "warp-stubs", stubs)
    with (stubs / "_src" / "types.pyi").open("a") as f:
        f.write("# drift\n")
    result = subprocess.run(
        [sys.executable, "tools/generate_stubs.py", "--check", "--output", str(stubs)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "stale generated stubs: _src/types.pyi" in result.stderr


def test_shape_rules_validate() -> None:
    # Pyrefly validates `@type_shape_dsl_function` bodies; an invalid rule would
    # otherwise only surface as an `Unknown` shape at its call sites.
    run(
        PYREFLY,
        "check",
        "--config",
        "/dev/null",
        "--search-path",
        ".",
        "--python-interpreter-path",
        sys.executable,
        "warp-stubs/_shapes.pyi",
    )


def test_expectations() -> None:
    # Every `assert_type` must hold, and errors must occur exactly on the lines
    # carrying an `E:` comment. Uses pyrefly.toml (tests/ and examples/).
    run(PYREFLY, "check", "--expectations")
