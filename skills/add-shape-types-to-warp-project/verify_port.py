#!/usr/bin/env python3
"""Lint a Warp file for shape-annotation mistakes that Pyrefly cannot see.

Pyrefly checks the types; this script checks the *runtime* rules that shape
annotations must respect in Warp code, all of which otherwise only fail when a
kernel is built or a module is imported:

  ERROR  `reveal_type`/`assert_type`/`cast` inside a `@wp.kernel`/`@wp.func`,
         unless under `if TYPE_CHECKING:` (Warp drops that block when it builds
         the kernel; anywhere else its code generator rejects the call)
  ERROR  annotated assignment (`x: T = ...`) inside a kernel or function
         (not supported by Warp's code generator)
  ERROR  shape syntax in a `@wp.struct` field (Warp evaluates field annotations)
  ERROR  runtime import or use of a stub-only name (`from warp import Tile`,
         `Vec3 = wp.Vector[...]`): `Tile`, `Array`, `Vector`, ... exist for type
         checkers only (WARN inside functions, which only fail when called)
  WARN   `assert_type`/`cast` with an unquoted shape type outside kernels
         (evaluated at runtime when executed; quote it: `"wp.array[Any, [N, 3]]"`)
  ERROR  shape syntax (`wp.array[float, [N, 3]]`, `Tile[...]`, `IntVar`, ...)
         in a kernel or function signature (Warp evaluates those annotations)
  ERROR  shape syntax in annotations without `from __future__ import annotations`
         (Python evaluates them at import time and Warp rejects them)
  WARN   `shape_extensions` imported outside `if TYPE_CHECKING:`
         (makes pyrefly-shape-extensions a runtime dependency)
  WARN   `# type: ignore` / `# pyrefly: ignore` suppressions (audit each one)
  INFO   kernel parameters typed `wp.array[dtype]` (unknown rank to Pyrefly;
         `wp.array1d[dtype]` means the same to Warp and enables rank checks)

Usage: verify_port.py FILE [FILE ...]      exit status 1 if any ERROR
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

WARP_DECORATORS = {"kernel", "func", "func_grad", "func_replay", "func_native"}
PROBES = {"reveal_type", "assert_type", "cast", "expect_shape"}
SHAPE_NAMES = {"Tile", "IntVar", "IntTuple", "Int", "Elements", "Shaped"}
RANKLESS = {"array", "indexedarray", "fabricarray"}
# Subscripts whose list argument is a shape (unlike `Callable[[int], float]`).
SHAPED_BASES = {"array", "Array", "indexedarray", "fabricarray", "Tile", "Shaped"}
# Declared in warp's stubs but absent from the runtime `warp` module.
STUB_ONLY = {
    "Array", "Tile", "TileStack", "Vector", "Matrix", "Quaternion", "Transformation",
    "FabricArray", "IndexedFabricArray", "Length", "Rows", "Cols", "DType", "NDim",
    "Shape", "Capacity", "DTypeFloat", "DTypeScalar",
}
SUPPRESSION = re.compile(r"#\s*(type:\s*ignore|pyrefly:\s*ignore)")


@dataclass
class Finding:
    level: str
    line: int
    message: str


def decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def is_warp_function(node: ast.AST, wrapped: set[str] = frozenset()) -> bool:
    """Decorated with `@wp.kernel`/`@wp.func`, or passed to `wp.kernel(fn)` (`wrapped`)."""
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
        node.name in wrapped or any(decorator_name(d) in WARP_DECORATORS for d in node.decorator_list)
    )


def wrapped_function_names(tree: ast.AST) -> set[str]:
    """Names of functions turned into kernels by a call, e.g. `k = wp.kernel(body)`."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and decorator_name(node.func) in WARP_DECORATORS and node.args:
            if isinstance(node.args[0], ast.Name):
                names.add(node.args[0].id)
    return names


def has_shape_syntax(annotation: ast.expr) -> bool:
    """`array[..., [dims]]`, or a shape-extensions name such as `Tile[...]` / `IntVar`."""
    for node in ast.walk(annotation):
        if isinstance(node, ast.Subscript) and decorator_name(node.value) in SHAPED_BASES:
            elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if any(isinstance(e, ast.List) for e in elements):
                return True
        name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
        if name in SHAPE_NAMES:
            return True
    # Quoted annotations hide their structure from the AST.
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return bool(re.search(r"\b(array|Array|indexedarray|fabricarray)\[[^\]]*\[|\bTile\[|\bIntVar\b", annotation.value))
    return False


def annotations_of(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.expr]:
    args = fn.args
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
        if arg is not None and arg.annotation is not None:
            yield arg.annotation
    if fn.returns is not None:
        yield fn.returns


def type_checking_nodes(root: ast.AST) -> set[int]:
    """ids of every node nested in an `if TYPE_CHECKING:` body under `root`."""
    inside: set[int] = set()
    for node in ast.walk(root):
        if isinstance(node, ast.If) and is_type_checking_guard(node):
            for stmt in node.body:
                inside.update(id(n) for n in ast.walk(stmt))
    return inside


def is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def runtime_shape_extension_imports(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    for stmt in body:
        if isinstance(stmt, ast.If) and is_type_checking_guard(stmt):
            continue
        if isinstance(stmt, ast.ImportFrom) and (stmt.module or "").split(".")[0] == "shape_extensions":
            yield stmt
        elif isinstance(stmt, ast.Import) and any(a.name.split(".")[0] == "shape_extensions" for a in stmt.names):
            yield stmt
        for child in ("body", "orelse", "finalbody"):
            nested = getattr(stmt, child, None)
            if isinstance(nested, list) and not isinstance(stmt, (ast.FunctionDef, ast.ClassDef)):
                yield from runtime_shape_extension_imports(nested)


def check(path: Path) -> list[Finding]:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    findings: list[Finding] = []

    future = any(
        isinstance(s, ast.ImportFrom) and s.module == "__future__" and any(a.name == "annotations" for a in s.names)
        for s in tree.body
    )

    wrapped = wrapped_function_names(tree)
    for fn in (n for n in ast.walk(tree) if is_warp_function(n, wrapped)):
        assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        kind = (
            "kernel"
            if fn.name in wrapped or any(decorator_name(d) == "kernel" for d in fn.decorator_list)
            else "function"
        )
        for annotation in annotations_of(fn):
            # `wp.tile[float, M, N]` is Warp's own syntax for tile parameters.
            if has_shape_syntax(annotation):
                findings.append(
                    Finding(
                        "ERROR",
                        annotation.lineno,
                        f"shape annotation in @wp.{kind} `{fn.name}` signature: Warp evaluates it at "
                        "build time; keep `wp.array2d[dtype]` and let Pyrefly infer shapes in the body",
                    )
                )
            elif kind == "kernel" and isinstance(annotation, ast.Subscript):
                base = annotation.value
                name = base.attr if isinstance(base, ast.Attribute) else base.id if isinstance(base, ast.Name) else ""
                single = not isinstance(annotation.slice, ast.Tuple)
                if name in RANKLESS and single:
                    findings.append(
                        Finding(
                            "INFO",
                            annotation.lineno,
                            f"`{ast.unparse(annotation)}` in kernel `{fn.name}` has unknown rank for Pyrefly; "
                            f"`{ast.unparse(annotation).replace(name + '[', name + '1d[', 1)}` is identical to Warp "
                            "and enables rank checks",
                        )
                    )
        guarded = type_checking_nodes(fn)
        for node in ast.walk(fn):
            if node is fn or id(node) in guarded:
                continue
            if isinstance(node, ast.Call):
                callee = decorator_name(node.func)
                if callee in PROBES:
                    findings.append(
                        Finding(
                            "ERROR",
                            node.lineno,
                            f"`{callee}` inside @wp.{kind} `{fn.name}` breaks Warp codegen; remove it or "
                            "move it under `if TYPE_CHECKING:`",
                        )
                    )
            elif isinstance(node, ast.AnnAssign):
                findings.append(
                    Finding("ERROR", node.lineno, f"annotated assignment inside @wp.{kind} `{fn.name}` is not supported by Warp")
                )

    if not future:
        for node in ast.walk(tree):
            annotation = None
            if isinstance(node, ast.AnnAssign):
                annotation = node.annotation
            elif isinstance(node, ast.arg):
                annotation = node.annotation
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                annotation = node.returns
            if annotation is not None and has_shape_syntax(annotation):
                findings.append(
                    Finding(
                        "ERROR",
                        annotation.lineno,
                        "shape annotation without `from __future__ import annotations`: it is evaluated "
                        "at import time and Warp rejects it",
                    )
                )

    warp_functions = {id(n) for fn in ast.walk(tree) if is_warp_function(fn, wrapped) for n in ast.walk(fn)}
    in_functions = {
        id(n)
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        for n in ast.walk(fn)
        if n is not fn
    }
    module_guarded = type_checking_nodes(tree)
    for node in ast.walk(tree):
        if id(node) in module_guarded:
            continue
        # @wp.struct fields are evaluated by Warp, like kernel signatures.
        if isinstance(node, ast.ClassDef) and any(decorator_name(d) == "struct" for d in node.decorator_list):
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and has_shape_syntax(stmt.annotation):
                    findings.append(
                        Finding("ERROR", stmt.lineno, f"shape annotation on @wp.struct `{node.name}` field: Warp evaluates it")
                    )
        # Stub-only modules and names fail at import time.
        if isinstance(node, ast.ImportFrom) and node.module == "warp.shape_checks":
            findings.append(
                Finding("ERROR", node.lineno, "`warp.shape_checks` only exists in the stubs; import it under `if TYPE_CHECKING:`")
            )
        if (
            isinstance(node, ast.Call)
            and decorator_name(node.func) == "expect_shape"
            and id(node) not in warp_functions
        ):
            findings.append(
                Finding("ERROR", node.lineno, "`expect_shape` only exists for the type checker; call it under `if TYPE_CHECKING:`")
            )
        if isinstance(node, ast.ImportFrom) and node.module == "warp":
            for alias in node.names:
                if alias.name in STUB_ONLY:
                    findings.append(
                        Finding(
                            "ERROR",
                            node.lineno,
                            f"`from warp import {alias.name}` fails at runtime ({alias.name} only exists in the "
                            "stubs); import it under `if TYPE_CHECKING:`",
                        )
                    )
        # Legacy type aliases are ordinary assignments: Python evaluates them at
        # import time. PEP 695 `type X = ...` is lazy and fine.
        alias_value = None
        if isinstance(node, ast.Assign):
            alias_value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None and decorator_name(node.annotation) == "TypeAlias":
            alias_value = node.value
        # Only type expressions (`X[...]`, `X[...] | Y`), not calls such as `cast(...)`.
        if isinstance(alias_value, (ast.Subscript, ast.BinOp)) and has_shape_syntax(alias_value):
            findings.append(
                Finding(
                    "ERROR",
                    node.lineno,
                    "shaped type alias is evaluated when this line runs and Warp rejects it; "
                    "use a PEP 695 alias (`type Points = wp.array[Any, [int, 3]]`)",
                )
            )
        # `assert_type(x, T)` / `cast(T, x)` evaluate `T` whenever they run.
        if isinstance(node, ast.Call) and id(node) not in warp_functions:
            callee = decorator_name(node.func)
            type_arg = None
            if callee == "assert_type" and len(node.args) == 2:
                type_arg = node.args[1]
            elif callee == "cast" and node.args:
                type_arg = node.args[0]
            if type_arg is not None and not isinstance(type_arg, ast.Constant) and has_shape_syntax(type_arg):
                # At module level the call always runs; in a function, only if called.
                level = "WARN" if id(node) in in_functions else "ERROR"
                findings.append(
                    Finding(
                        level,
                        node.lineno,
                        f"`{callee}` evaluates its shape type when executed, which crashes; quote it "
                        f"(`\"{ast.unparse(type_arg)}\"`) unless this code never runs",
                    )
                )

    # `wp.Vector[...]`, `wp.Tile[...]` evaluated at runtime. Deferred annotations,
    # PEP 695 aliases, string types and `if TYPE_CHECKING:` blocks never run.
    deferred: set[int] = set(module_guarded)
    for node in ast.walk(tree):
        parts: list[ast.AST] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and future:
            parts = [*annotations_of(node)]
        elif isinstance(node, ast.AnnAssign) and future:
            parts = [node.annotation]
        elif isinstance(node, ast.TypeAlias):
            parts = [node.value]
        for part in parts:
            deferred.update(id(n) for n in ast.walk(part))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in STUB_ONLY
            and isinstance(node.value, ast.Name)
            and node.value.id in {"wp", "warp"}
            and id(node) not in deferred
        ):
            level = "WARN" if id(node) in in_functions and id(node) not in warp_functions else "ERROR"
            findings.append(
                Finding(
                    level,
                    node.lineno,
                    f"`{node.value.id}.{node.attr}` only exists in the stubs and fails when this runs; use it in "
                    "annotations, a PEP 695 `type` alias, a quoted type, or under `if TYPE_CHECKING:`",
                )
            )

    for stmt in runtime_shape_extension_imports(tree.body):
        findings.append(
            Finding(
                "WARN",
                stmt.lineno,
                "`shape_extensions` imported at runtime: move it under `if TYPE_CHECKING:` unless "
                "pyrefly-shape-extensions is meant to be a runtime dependency",
            )
        )

    for number, line in enumerate(source.splitlines(), start=1):
        if SUPPRESSION.search(line):
            findings.append(Finding("WARN", number, f"suppression, justify it: {line.strip()[:100]}"))

    return sorted(findings, key=lambda f: f.line)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    errors = 0
    for name in sys.argv[1:]:
        path = Path(name)
        findings = check(path)
        print(f"=== {path}: {sum(f.level == 'ERROR' for f in findings)} error(s), {len(findings)} finding(s)")
        for f in findings:
            print(f"{path}:{f.line}: {f.level}: {f.message}")
        errors += sum(f.level == "ERROR" for f in findings)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
