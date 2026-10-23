#!/usr/bin/env python3
"""Compare Pyrefly errors on a corpus with and without the local warp-stubs.

The stubs should only ever *remove* errors from correct code. Errors that appear
only with the stubs are printed: each one is either a real bug in the corpus or
a false positive to fix in overrides/.

    python tools/compare_examples.py [PATH ...]   # default: warp/examples
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(paths: list[str], *, stubs: Path | None) -> list[dict]:
    command = [
        str(ROOT / ".venv/bin/pyrefly"),
        "check",
        "--config",
        "/dev/null",
        "--python-interpreter-path",
        str(ROOT / ".venv/bin/python"),
        "--output-format",
        "json",
        *(["--search-path", str(stubs)] if stubs else []),
        *paths,
    ]
    out = subprocess.run(command, capture_output=True, text=True).stdout
    data = json.loads(out)
    errors = data["errors"] if isinstance(data, dict) else data
    return [e for e in errors if e.get("severity", "error") == "error"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*")
    parser.add_argument(
        "--stubs",
        type=Path,
        default=ROOT,
        help="directory containing the warp-stubs package to evaluate (default: this repo)",
    )
    args = parser.parse_args()
    paths = args.paths
    if not paths:
        import warp  # noqa: PLC0415

        paths = [str(Path(warp.__file__).parent / "examples")]

    base = run(paths, stubs=None)
    with_stubs = run(paths, stubs=args.stubs)
    key = lambda e: (e["path"], e["line"], e["name"])  # noqa: E731
    base_keys = {key(e) for e in base}
    new = [e for e in with_stubs if key(e) not in base_keys]

    print(f"errors without stubs: {len(base)}, with stubs: {len(with_stubs)}, new: {len(new)}")
    for name, count in collections.Counter(e["name"] for e in new).most_common():
        print(f"  {count:4d} {name}")
    for e in new:
        source = Path(e["path"]).read_text().splitlines()[e["line"] - 1].strip()
        print(f"{e['path']}:{e['line']} [{e['name']}] {e['description'].splitlines()[0][:200]}")
        print(f"    | {source[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
