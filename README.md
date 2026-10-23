# pyrefly-warp-stubs

[NVIDIA Warp](https://github.com/NVIDIA/warp) type stubs with array and tile
shape tracking for [Pyrefly](https://pyrefly.org/en/docs/tensor-shapes/),
in the spirit of Pyrefly's `pyrefly-numpy-stubs` and `pyrefly-torch-stubs`.

```python
@wp.kernel
def mlp(x: wp.array2d[float], w: wp.array2d[float], out: wp.array2d[float]):
    inputs = wp.tile_load(x, shape=(16, 32))      # Tile[float, [16, 32]]
    weights = wp.tile_load(w, shape=(64, 16))     # Tile[float, [64, 16]]
    hidden = wp.tile_matmul(weights, inputs)      # Tile[float, [64, 32]]
    wp.tile_matmul(inputs, weights)               # error: expected Int[32], got Int[64]
    bias = wp.tile_load(w, shape=(64, 1))
    hidden + bias                                 # error: tiles never broadcast implicitly
    wp.tile_sum(hidden, axis=2)                   # error: reduction axis out of bounds
```

Tile shapes in Warp are compile-time constants, so a shape bug normally only
surfaces when the kernel is built at launch. With these stubs Pyrefly reports it
in the editor, along with Warp's scalar and vector type rules
(`wp.max(f64, 0.0)`, `vec3 * vec3`, `mat44 * vec3` are errors, as in Warp).

## Install

Add it as a **development** dependency of a project that uses Warp:

```bash
uv add --dev pyrefly-warp-stubs        # or: pip install pyrefly-warp-stubs
# latest main: uv add --dev "pyrefly-warp-stubs @ git+https://github.com/julien-blanchon/pyrefly-warp-stubs"
```

This installs `warp-stubs` (a PEP 561 partial stub package) and
`pyrefly-shape-extensions`. Pyrefly picks the stubs up automatically. Requires
Python ≥ 3.12 and Pyrefly ≥ 1.4, in `default` or `strict` mode: add a
`pyrefly.toml` (or `[tool.pyrefly]` in `pyproject.toml`), since the `basic`
mode used without one hides shape errors.

The stubs are generated from **warp-lang 1.18**. Newer Warp releases install
fine, but the stubs replace Warp's own `__init__.pyi`, so functions added after
1.18 are reported as missing until the stubs are regenerated (see
[Development](#development)).

## Using it in your project

The stubs only exist for the type checker: Python never imports `.pyi` files,
so nothing changes at runtime, and people who install *your* package never
download these stubs or need Pyrefly. That holds as long as shape annotations
stay type-checking-only:

```python
from __future__ import annotations  # required: see below

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:  # never imported at runtime
    from shape_extensions import IntVar


def normalize[N: IntVar](points: wp.array[wp.vec3, [N]]) -> wp.array[wp.vec3, [N]]:
    ...

normalize(wp.zeros(10, dtype=wp.vec3))           # ok: array[vec3f, [10]]
normalize(wp.zeros((10, 3), dtype=float))         # error: array[float, [10, 3]]
```

- **Keep `from __future__ import annotations`** in every file with shaped
  annotations. Without it Python evaluates `wp.array[wp.vec3, [N]]` at import
  time, and Warp rejects anything but an integer `ndim` there.
- **Import `shape_extensions` only under `TYPE_CHECKING`.** Using
  `assert_shape`, `shape_vars` or a top-level `from shape_extensions import ...`
  in shipped code makes `pyrefly-shape-extensions` a runtime dependency.
- **No shapes in `@wp.kernel`/`@wp.func` signatures or `@wp.struct` fields.**
  Warp evaluates those annotations itself, even with the future import; keep
  `wp.array2d[float]` there. Shapes are still inferred inside the kernel body.
- **Shape assertions inside kernels go under `if TYPE_CHECKING:`.** Warp
  constant-folds the condition and never compiles the block, while Pyrefly
  checks it. A bare `reveal_type`/`assert_type` in a kernel breaks the build.
  `expect_shape` (from the stub-only `warp.shape_checks`) checks against your
  tile-size constants instead of literals:
  ```python
  t = wp.tile_load(a, shape=(TILE_M, TILE_N))
  if TYPE_CHECKING:
      expect_shape(t, (TILE_M, TILE_N))
      assert_type(t, Tile[float, [8, 4]])
  ```
- **Quote shaped types in executed `assert_type`/`cast`**:
  `assert_type(x, "wp.array[float, [10, 3]]")`. Their type argument is evaluated
  when the line runs, even with the future import. Shaped type aliases must use
  PEP 695 (`type Points = wp.array[wp.vec3, [int]]`).
- **`Tile`, `Array`, `Vector`, `warp.shape_checks`, ... exist only in the
  stubs**: import and use them under `if TYPE_CHECKING:` or in annotations.
- **Public API.** Other type checkers (Pyright, mypy, ty) cannot read
  `wp.array[wp.vec3, [N]]`, so a downstream user without these stubs sees an
  untyped parameter. If that matters, use the portable
  `Shaped[wp.array[wp.vec3], "[N]"]` inside a `@shape_vars("N")` function, which
  other checkers read as plain `wp.array[wp.vec3]`. That decorator runs at
  import time, so it does make the small, pure-Python `pyrefly-shape-extensions`
  a runtime dependency of your package.
- **Contributors using other checkers.** Pyright or ty in an environment with
  your dev dependencies will also pick up `warp-stubs` and report its shape
  syntax as errors. This only affects people developing your project.

`skills/add-shape-types-to-warp-project/` contains an agent skill that applies
all of this step by step, and `verify_port.py`, a lint for these runtime rules
(Pyrefly cannot see them).

### Shapes on kernel inputs: typed launchers

Kernel signatures must stay plain Warp, and `wp.launch(inputs=[...])` is not
checked against them. Put the shapes on the host function that launches the
kernel; Pyrefly then checks every call, and the dimensions flow to the result:

```python
def matmul[M: IntVar, K: IntVar, N: IntVar](
    a: wp.array[float, [M, K]], b: wp.array[float, [K, N]]
) -> wp.array[float, [M, N]]:
    c = wp.zeros((a.shape[0], b.shape[1]), dtype=float)   # array[float, [M, N]]
    wp.launch_tiled(gemm_kernel, dim=..., inputs=[a, b, c], block_dim=64)
    return c

matmul(wp.empty((64, 32)), wp.empty((32, 48)))   # array[float, [64, 48]]
matmul(wp.empty((32, 48)), wp.empty((64, 32)))   # error: expected [48, @_]
```

Inside the kernel, `a`, `b` and `c` are only `array[float, [int, int]]`: a
kernel cannot state that two parameters share a shape (a generic kernel
signature fails when Warp evaluates it), and `inputs=[...]` is not checked
against the kernel. Allocating outputs from the inputs in the launcher
(`wp.zeros((a.shape[0], b.shape[1]))`, `wp.empty_like(x)`) enforces the
relationship by construction and leaves that one line as the only unchecked
link.

### Named dimensions (`B C H W`)

Named dimensions are type parameters. Three spellings check the same way:

| Spelling | Needs at runtime |
|---|---|
| `def f[B: IntVar, C: IntVar, H: IntVar, W: IntVar](x: wp.array[float, [B, C, H, W]])` | nothing (`IntVar` imported under `TYPE_CHECKING`) |
| `@shape_vars("B, C, H, W")` and `Shaped[wp.array[float], "[B, C, H, W]"]` | `pyrefly-shape-extensions` |
| `@static_jaxtyping("B C H W")` and `Float[wp.array, "B C H W"]` | `jaxtyping` and `pyrefly-shape-extensions` (no dtype) |

Arithmetic on them is tracked: `x.reshape((x.shape[0], x.shape[1], -1))` on a
`[B, C, H, W]` array is `[B, C, H * W]` (see `examples/image_batch.py`).

### Libraries: a `typing` extra

If your library uses Warp and wants its users to get the shape types too, offer
the stubs as an **extra**, not only a dev dependency: dependency groups
(PEP 735) are never published, extras are.

```toml
[project.optional-dependencies]
typing = ["pyrefly-warp-stubs>=0.1"]       # users opt in: pip install mylib[typing]

[dependency-groups]
dev = ["mylib[typing]", "pyrefly>=1.4", "pytest"]   # contributors get it by default
```

A plain `pip install mylib` still installs only Warp. Keep the published
requirement a plain name (`pyrefly-warp-stubs>=0.1`): build backends refuse git
URLs in `optional-dependencies` and PyPI rejects them. To develop against the
latest `main` instead, add a source that is not published:

```toml
[tool.uv.sources]
pyrefly-warp-stubs = { git = "https://github.com/julien-blanchon/pyrefly-warp-stubs" }
```

## What is tracked

| Area | Examples |
|---|---|
| Array creation | `wp.zeros((128, 3), dtype=wp.vec3)` → `array[vec3f, [128, 3]]`; `zeros`/`ones`/`empty`/`full` (dtype from `value`), `wp.array(shape=...)`, list literals, `from_numpy`, `*_like`, `clone` |
| Array views | `reshape` (with `-1`), `flatten`, `transpose`, slicing `a[1:5, ::2]`, `contiguous`, `to`, `grad` |
| Rank aliases | `wp.array2d[float]` is `array[float, [int, int]]`; passing a 2D array where `array3d` is expected is an error |
| Kernel indexing | `i, j = wp.tid()`; `a[i, j]` → element type; `a[i]` → row view; too many indices is an error |
| Scalars | `float` *is* `wp.float32`, `int` is `wp.int32`, `bool` is `wp.bool`, as in Warp. No implicit conversions: `wp.max(f32, 0.0)` is fine, `wp.max(f64, 0.0)` and `f32 + f64` are errors |
| Vectors, matrices | `wp.vec3`, `wp.vec3f` and `Vector[float, Literal[3]]` are one type. `v + v`, `v * s`, `m * v` (`[R, C] × [C] → [R]`), `m @ m`, `q * q`, `t.p`, `v.x` are typed; `v * v`, `v + 1.0`, `v * f64`, `mat44 * vec3`, `q * v` are errors, as in Warp |
| Tile creation | `tile_zeros`/`ones`/`empty`/`full`/`randf`/`randi`, `tile_arange(start, stop, step)` |
| Global memory | `tile_load`/`tile_store`/`tile_atomic_add` (tile rank must match array rank), `*_indexed` |
| Tile layout | `tile_transpose`, `tile_reshape`, `tile_squeeze`, `tile_broadcast` (NumPy `broadcast_to` rules), `tile_view`, `tile_astype`, subscripts `t[i]`, `t[1:3, :]`, `t[i, j]` |
| Tile operators | `+ - * / @` between identical shapes, `* /` with scalars, unary `-`, `& \| ^` on integer tiles. `t + 1.0`, `+t`, `t % u` are errors, as in Warp |
| Tile math | `tile_map`, `tile_matmul` (incl. `out=`), `tile_dot`, `tile_sum`/`min`/`max`/`argmin`/`argmax`/`reduce` (with `axis`), scans |
| Linear algebra | `tile_matmul`, `tile_cholesky`, `*_solve`, `*_solve_inplace` (float tiles, square / matching extents), `tile_diag_add` |
| `warp.fem` | Fields, domains, operators and integrands are callable inside integrands (`u(s)`, `fem.inner(u, s)`); `@wp.func` overloads such as `fem.make_free_sample` resolve per argument type |

Literal sizes, module-level constants (`TILE = 64`) and `wp.constant(64)` are
tracked; sizes only known at runtime become `int` dimensions, which keep the rank.

Every operator and rule above was checked against what Warp 1.18 actually
builds, by compiling kernels on the CPU; the tests keep it that way.

## Examples

Each example runs on the CPU (`python examples/<name>.py`), checks its result
against NumPy, and type-checks cleanly; its docstring lists errors Pyrefly
reports for typical mistakes (all verified by the test suite or by hand).

| Example | Shows |
|---|---|
| `tile_mlp.py` | tile matmul/broadcast/map, `expect_shape` in a kernel, a typed launcher generic in the batch size |
| `blocked_matmul.py` | tile offsets, `tile_matmul(a, b, out)` accumulation, `[M, K] @ [K, N] -> [M, N]` |
| `row_softmax.py` | axis reductions, `tile_reshape`, explicit `tile_broadcast`, element-wise tile math |
| `image_batch.py` | named dimensions `B, C, H, W`, symbolic `H * W`, views, a 4-D launch |
| `particles.py` | `vec3`/`mat33` arithmetic, scalar rules, class-level array declarations, a shared `N` |

## How it works

Pyrefly treats any class with a type parameter bound to `IntTuple` as a shaped
array. Warp already declares `Array(Generic[DType, NDim])` and
`Tile(Generic[DType, Shape])`. These stubs re-declare them as
`Array[DType, Shape: IntTuple]` and `Tile[DType, Shape: IntTuple]`, keeping the
shape second so that `wp.array[float]` still means "dtype `float`".

Warp's `__init__.pyi` is ~11k autogenerated lines, and the classes live in
`warp/_src`, so the stubs are **generated** rather than hand-written:

```
overrides/<path>.pyi    ─┐                          ┌─> warp-stubs/<path>.pyi
installed warp package  ─┴─ tools/generate_stubs.py ┘   (_shapes.pyi, shape_checks.pyi are hand-written)
```

Each file under `overrides/` stubs the Warp module at the same path (12 modules:
`__init__`, `_src/types`, renderers, `sparse`, several `fem` modules).
`tools/generate_stubs.py` copies the module and merges in the overrides:

- An override `def` replaces every upstream overload of that name; the upstream
  docstring is carried over (also for functions upstream only re-exports).
- An override `class` merges into the upstream class: it supplies the header and
  the members it redefines, and keeps the rest.
- `rewrite_params` widens upstream parameter annotations that are narrower than
  what the code accepts (e.g. renderer `pos: tuple` also takes a `wp.vec3`).
- Warp's `@wp.func` overloading by redefinition becomes `typing.overload`
  groups; otherwise the last definition would shadow the others.
- An override import may not silently rebind a name upstream imports from
  elsewhere (an early version replaced `wp.overload` with `typing.overload`).
- Function bodies are kept: Pyrefly infers attributes and return types from
  them even in stubs.

Shape computations that need logic (reshape, axis reductions, broadcasting,
`tile_arange`) are Pyrefly [shape DSL](https://pyrefly.org/en/docs/tensor-shapes-library-authors/)
rules in `warp-stubs/_shapes.pyi`.

### Fixes to upstream typing

Precise element types expose gaps that were hidden while `a[i]` was untyped, so
the overrides also fix these:

- Scalars: `float`/`int`/`bool` and `wp.float32`/`wp.int32`/`wp.bool` were
  distinct types, so `wp.max(a[i], 0.0)` and passing a `dtype=float` array to a
  `wp.float32` parameter were errors. Scalar operators are typed.
- Vectors: upstream has two unrelated `vec3f` classes (`warp/__init__.pyi` and
  `warp._src.types`), neither a `Vector`, with constructors that reject valid
  calls (`wp.vec3(np_array)`). They are now one type with Warp's operators;
  `wp.types.vector(...)` and `wp.spatial_vector` are vectors too.
- `wp.tid()` is `int | tuple[int, int] | ...` upstream, which cannot be unpacked
  as `i, j = wp.tid()`. It is now an `int` that can also be unpacked.
- `warp.fem`: fields, domains, operators and integrands were not callable (360
  errors on Warp's FEM examples), and `@integrand` was typed as a union.
- `for i in wp.hash_grid_query(...)` is iterable; `wp.where`/`wp.select` keep
  the value type; `tile_*_solve_inplace` accept 2D right-hand sides;
  `tile_fft` and the `*_indexed` functions accept any element type;
  `wp.indexedarray1d[...]`–`4d[...]` work in annotations; `fem.Grid2D`/`Grid3D`
  and the renderers accept the vector values Warp's examples pass them.

On Warp's own 92 example scripts, Pyrefly reports **415 errors with these stubs
versus 2145 without**. Nine errors are new: one real bug in the examples
(`range(num_bodies / TILE_SIZE)` passes a float), four where
`example_diffusion_3d.py` passes 3D bounds to a 2D helper annotated `wp.vec2`,
and four from a Pyrefly bug (below). See `tools/compare_examples.py`.

## Known limitations

- `wp.array[dtype, Literal[ndim]]` cannot be expressed: the second parameter is
  now the shape. Use `wp.array2d[dtype]` etc.
- Tile shapes cannot be written in kernel signatures or kernel-local
  annotations: Warp evaluates kernel annotations at runtime and does not support
  annotated assignments in kernels. Shapes are inferred inside kernel bodies.
- `wp.array[float]` (no rank) is an array of unknown rank; use `wp.array1d[...]`
  to get rank checks.
- `wp.launch(kernel, inputs=[...])` arguments are not checked against the
  kernel signature.
- `wp.tile[float, M, N]` annotations on `@wp.func` parameters are valid Warp,
  but Pyrefly 1.4 cannot express a name that is both the `wp.tile(x)` builtin
  and a variadic shape type, so those parameters are `Unknown` (as upstream).
- Module constants cannot be used as dimensions in annotations (`M = 4`,
  `Tile[float, [M, 8]]`); no spelling works in Pyrefly 1.4. Use literals,
  `IntVar` parameters, or `expect_shape(t, (M, 8))`. Constants *are* tracked in
  calls (`shape=(M, 8)`).
- Typing lets `int` pass where `float` is expected, so `wp.int32` (which is
  `int`) tiles reach float-only tile operations, and `wp.max(f32, 1)` passes;
  Warp rejects both. Other integer types (`wp.int64`, `wp.uint32`, ...) are caught.
- A list literal passed inline as an argument (`f(wp.array([0.5, 0.5]))`) is
  typed from the expected parameter and not checked against it; assigned to a
  variable first, it is.
- `.w` on a 3-vector is not caught; the length from
  `wp.types.vector(length=5, ...)` is widened to `int`.
- `@wp.func` overloading by redefinition in *your* code: the checker sees the
  last definition only (the generator fixes this for Warp's own modules).
- Results of `tile_map`, of field evaluations in `warp.fem`, and of builtins
  that upstream declares as `Vector[Scalar, Any]` (`wp.normalize`) are not
  length- or dtype-tracked.
- `wp.spatial_vector` is a function in upstream's stubs (the kernel
  constructor), so it cannot be used as a type; use `wp.spatial_vectorf`.
- **Pyrefly 1.4 issues found while building this:**
  - A shape list spelled with `builtins.int` (`[builtins.int, builtins.int]`)
    silently becomes `Unknown`. Bare `int` works.
  - Calling an unspecialised generic alias (`self.x = wp.array2d(data)`)
    reports the alias's type variable as leaking (`invalid-type-var`), even with
    a `default=Any`.

## Development

```bash
uv venv -p 3.13 && uv pip install pyrefly==1.4.0 pyrefly-shape-extensions==1.4.0 \
    "warp-lang==1.18.*" numpy pytest

.venv/bin/python tools/generate_stubs.py    # after editing overrides/
.venv/bin/python -m pytest tests/           # static + runtime suites
.venv/bin/python tools/compare_examples.py  # errors added/removed on Warp's examples
```

Each test file pairs **static** `check_*` functions (`assert_type`, with an
`# E: <message>` comment on lines that must error, checked by
`pyrefly check --expectations`) with **runtime** `test_*` functions. The runtime
functions launch the same operations on Warp's CPU backend: well-typed kernels
must match NumPy, and every kernel Pyrefly rejects must also fail to build in
Warp. `tests/test_static.py` runs the Pyrefly half, checks that the generated
stubs are up to date, and validates the DSL rules; `tests/test_verify_port.py`
pins the lint.

To target another Warp release: install it, regenerate, and run the tests.
Overrides of functions whose upstream signature changed will show up as test
failures or new errors in `compare_examples.py`.

## License

The generated stubs contain portions of Warp's source, stubs and docstrings
(Apache-2.0, © NVIDIA). `warp-stubs/_shapes.pyi` adapts rules from
`pyrefly-numpy-stubs` (MIT, © Meta Platforms).
