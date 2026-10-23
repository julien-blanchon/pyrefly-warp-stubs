# Warp shape typing reference

What `pyrefly-warp-stubs` tracks, the Warp-specific rules annotations must
follow, and how to read Pyrefly's shape errors. Facts here were verified against
warp-lang 1.18 and Pyrefly 1.4. When in doubt, a `reveal_type` probe is the
authority, not this page.

## Core types

| Type | Meaning |
|---|---|
| `wp.array[DType, [D0, D1]]` | Array with shape `(D0, D1)`. Dims are literals (`3`), `IntVar` params (`N`), arithmetic (`N * 3`), or `int` (known rank, unknown extent). |
| `wp.array[DType]` | Array of **unknown rank**: compatible with any shape. |
| `wp.array1d[DType]` … `wp.array4d[DType]` | `array[DType, [int]]` … `array[DType, [int, int, int, int]]`: known rank, unknown extents. |
| `Tile[DType, [M, N]]` | Kernel-scope tile. `Tile` exists only for the type checker: import it with `if TYPE_CHECKING: from warp import Tile`. |
| `Int[N]` (shape_extensions) | An `int` whose value is the type-level `N`. `Int[N] * 2` is `Int[N * 2]`. |
| `IntVar` / `IntTuple` (shape_extensions) | Bounds for dimension (`N: IntVar`) and whole-shape (`S: IntTuple`) type parameters. |

`a.shape` is `IntTuple[D0, D1]`; `a.shape[0]` is `Int[D0]`.

## Where the shapes come from

Shape-aware signatures live in the installed `warp-stubs` package (find it with
`pyrefly dump-config` or `python -c "import importlib.util as u; print(u.find_spec('warp-stubs'))"`;
it sits next to `warp` in site-packages):

- `warp-stubs/__init__.pyi`: functions and builtins (`wp.zeros`, `wp.tile_*`).
- `warp-stubs/_src/types.pyi`: the `Array`, `array` and `Tile` classes and their methods.
- `warp-stubs/_shapes.pyi`: the rules computing derived shapes (reshape, reductions, broadcast).

Search for the function there. If its signature mentions `Tile[D, [M, N]]`,
`IntTupleOrList[S]` or calls a `*_shape(...)` rule, it is tracked. If it says
`Tile[Any, tuple[int, ...]]`, `Array[Any]` or bare `array`, it is untracked
(upstream signature) and its result is gradual.

## What is tracked

### Host (Python scope)

| API | Result |
|---|---|
| `wp.zeros/ones/empty/full(shape, dtype=...)` | `array[D, S]` from a literal, constant or `Int[N]` shape; `D` is the dtype (default `float`; `wp.full` takes it from `value`). A dtype built at runtime gives a vector with unknown length. |
| `wp.array(shape=...)`, `wp.array([[...]])` | Shape from `shape=` or from a regular nested list literal. |
| `wp.array(np_array)`, `wp.from_numpy(np_array)` | Only what NumPy's stubs know: `np.zeros((3, 4))` gives rank 2 (`[int, int]`), `rng.random(...)` gives no rank. Extents are unknown unless NumPy is shape-typed. |
| `zeros_like/ones_like/empty_like/full_like/clone(a)` | Same dtype and shape as `a`. |
| `a.reshape(...)` (incl. `-1`), `a.flatten()`, `a.transpose(axes)` | Computed; impossible reshapes are errors. |
| `a[1:5, ::2]`, `a.contiguous()`, `a.to(device)`, `a.grad` | Computed / preserved. |
| `a.view(dtype)`, `a.numpy()`, `wp.copy`, interop (`to_torch`, …) | Not tracked. |

### Kernel scope

| API | Result |
|---|---|
| `i, j = wp.tid()` / `i = wp.tid()` | `int` values (unpacking works for any launch rank). |
| `a[i, j]` (one index per dim) | Element type (the annotated dtype). Too many indices is an error. |
| `a[i]` on a 2D array | Row view `array[DType, [int]]`. |
| `wp.tile_load(a, shape=(M, N))` | `Tile[DType, [M, N]]`; the tile rank must equal the array rank (unknown-rank arrays accept any). |
| `wp.tile_store`, `wp.tile_atomic_add` | Rank checked. |
| `wp.tile_zeros/ones/empty/full/randf/randi(shape=...)` | Shape from argument. |
| `wp.tile_arange(n)`, `(start, stop)`, `(start, stop, step)` | `[n]`, `[stop - start]`, `ceil((stop - start) / step)`. |
| `t + u`, `t - u`, `t * u`, `t / u`, `t @ u` | **Identical shapes required**; tiles never broadcast implicitly. `@` is element-wise, like `*`. |
| `t * 2.0`, `2.0 * t`, `t / 2.0`, `2.0 / t`, `-t` | Same shape. **`t + 1.0`, `t - 1.0`, unary `+t` and `t % u` are errors**: Warp has no such tile overloads. |
| `i & j`, `i \| j`, `i ^ j` (integer tiles), `len(t)` | Same shape / `int`. `~i`, `<<`, `//` are not supported on tiles. |
| `wp.tile_broadcast(t, shape=...)` | NumPy `broadcast_to` rules, checked. |
| `wp.tile_matmul(a, b)` | `[M, K] @ [K, N] → [M, N]`; `out=` variant checks `[M, N]`. |
| `wp.tile_transpose`, `tile_reshape` (incl. `-1`), `tile_squeeze`, `tile_view(shape=)`, `tile_astype` | Computed. |
| `t[i, j]`, `t[i]`, `t[1:3, :]`, `wp.tile_extract` | Element / sub-tile / slice. |
| `tile_sum/min/max/argmin/argmax/reduce` | `[1]`, or the axis removed with `axis=`; out-of-range axis is an error. |
| `tile_scan_*`, `tile_map(op, t[, u])` | Same shape (map result dtype is `Any`). |
| `tile_matmul`, `tile_cholesky`, `*_solve`, `*_solve_inplace` | Floating-point tiles only (`wp.int32` tiles are an error; a Python `int` dtype slips through because typing treats `int` as a `float`). |
| `tile_dot`, `tile_diag_add` | Matching / square extents checked. |
| Scalars | `float` **is** `wp.float32`, `int` is `wp.int32`, `bool` is `wp.bool` (Warp canonicalises them), so `a[i]` on a `wp.array[wp.float32]` is `float`. No implicit conversions, as in Warp: `wp.max(a[i], 0.0)` is fine; `wp.max(f64, 0.0)`, `f32 + f64` are errors. `wp.where(c, x, y)` keeps the value type. |
| Vectors, matrices, quaternions, transforms | `wp.vec3` = `wp.vec3f` = `Vector[float, Literal[3]]`. `v + v`, `v * s`, `s * v`, `v / s`, `m * v` / `m @ v` (`[R, C] × [C] → [R]`), `v * m`, `m @ m`, `q * q`, `t * t`, `v.x`, `v[0]`, `m[i]` (row), `t.p`, `t.q` are typed. `v * v`, `v + 1.0`, `v * f64`, `mat44 * vec3`, `q * v`, `-t` are errors, as in Warp. `wp.dot`/`wp.length` give the scalar; `wp.normalize` & co. give `Vector[float, Any]`. |

## Warp rules annotations must respect

These are runtime rules. Pyrefly does not report them; `verify_port.py` does.
Each was verified by building and running kernels on Warp 1.18.

1. **No shapes in `@wp.kernel` / `@wp.func` signatures.** Warp evaluates kernel
   annotations itself (with `eval_str=True`) and only accepts
   `wp.array[dtype]`, `wp.array[dtype, Literal[ndim]]`, `wp.arrayNd[dtype]`,
   `wp.tile[dtype, *dims]`, and value types. Keep those; Pyrefly infers shapes
   inside the body from `tile_load(..., shape=...)` etc.
2. **`reveal_type` / `assert_type` / `cast` inside kernels or `wp.func`s only
   under `if TYPE_CHECKING:`.** Warp constant-folds `if` conditions and never
   compiles that block, while Pyrefly checks it. Anywhere else in a kernel the
   code generator rejects the call (`WarpCodegenError`). Import `Tile` under
   `if TYPE_CHECKING:` too.
3. **No annotated assignments inside kernels** (`x: float = ...`):
   ``Construct `ast.AnnAssign` not supported in kernels``.
4. **`from __future__ import annotations` in every module with shape
   annotations** on host code. Otherwise Python evaluates
   `wp.array[Any, [N, 3]]` at import time and Warp raises (`ndim must be an integer`).
5. **Import `shape_extensions` under `if TYPE_CHECKING:`** unless the project
   accepts `pyrefly-shape-extensions` as a runtime dependency. PEP 695 bounds
   (`def f[N: IntVar]`) are lazy, so a type-checking-only import works.
6. **Quote shaped types in executed host-side `assert_type` / `cast`**:
   `assert_type(x, "wp.array[Any, [10, 3]]")`. Their type argument is an
   ordinary expression, evaluated when the line runs even with the future
   import, and Warp raises `TypeError: ndim must be an integer`.
7. **No shapes on `@wp.struct` fields**: Warp evaluates field annotations
   (same `TypeError`). Keep `wp.array2d[dtype]` there.
8. **Stub-only names don't exist at runtime**: `Tile`, `Array`, `Vector`,
   `Matrix`, `Quaternion`, `Transformation`, `TileStack`. `from warp import Tile`
   passes Pyrefly but raises `ImportError`; import them under `if TYPE_CHECKING:`.
9. **Tile shapes must be compile-time constants for Warp**: literals,
   module-level ints, `wp.constant(...)`, or closure variables of a kernel
   factory. Pyrefly tracks literals, module-level ints and `wp.constant(...)`
   (including arithmetic such as `wp.constant(4 * NUM_FREQ)`). Closure and
   function-argument values become `int` dimensions: the rank is kept, extents are not.

## Checking shapes against constants: `expect_shape`

`assert_type(t, Tile[float, [8, 4]])` needs literal dimensions, which go stale
when `TILE_M` changes. The stubs ship a type-checking-only helper that takes the
shape as values:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from warp.shape_checks import expect_shape   # not a runtime module

@wp.kernel
def k(a: wp.array2d[float]):
    t = wp.tile_load(a, shape=(TILE_M, TILE_N))
    if TYPE_CHECKING:
        expect_shape(t, (TILE_M, TILE_N))   # (TILE_N, TILE_M) or (TILE_M,) is an error
```

Works for arrays too (`expect_shape(x, (n, 3))`); runtime-sized dimensions
match anything. Both the import and the call must stay under `TYPE_CHECKING`.

## Arrays stored on objects, and type aliases

- Declare array attributes at class level; that fixes their rank and dtype for
  every use and catches a wrong-rank reassignment:
  ```python
  class Simulation:
      x: wp.array1d[wp.vec3]
      rho: wp.array1d[float]
  ```
  It does **not** help with `self.x = wp.array2d(...)`: calling the
  `array1d`–`array4d` aliases still hits the Pyrefly `_ArrayDType` bug (see the
  error decoder). Assigning `wp.zeros(...)`, `wp.empty(...)` or `wp.array(...)`
  to a declared attribute works.
- Shaped type aliases must use PEP 695: `type Points = wp.array[Any, [int, 3]]`
  is lazy. `Points = wp.array[...]` and `Points: TypeAlias = ...` are evaluated
  at import and crash.

## Safe, behaviour-preserving improvements

- `wp.array[dtype]` → `wp.array1d[dtype]` in a **kernel signature**. Warp builds
  the identical annotation (1-D, same canonical dtype), and Pyrefly gains a rank.
  Do not apply this to host signatures that accept arrays of any rank.
- `TILE = 64` / `TILE = wp.constant(64)` for tile sizes that are already constant.
- Host helpers: `def f[N: IntVar](x: wp.array[Any, [N, 3]]) -> wp.array[Any, [N]]`.
  Bind sizes from integer parameters with `count: Int[N]`.

Anything that changes runtime structure (inlining, reordering, changing dtypes)
is not annotation work: report it, don't do it.

## Error decoder

Pyrefly prints a headline and, for shapes, a detail line (`Shape dimension
mismatch: ...`, `Tensor rank mismatch: ...`). When several overloads exist and
none fits, the headline is `No matching overload found for function ...` and the
per-overload reasons follow.

| Pyrefly says | Meaning / fix |
|---|---|
| `Argument Tile[.., [K2, N]] is not assignable to parameter b with type Tile[.., [K, @_]] in function warp.tile_matmul` (detail: `Shape dimension mismatch: expected Int[K], got Int[K2]`) | Inner dimensions differ: wrong operand order or a missing `tile_transpose`. |
| `` `+` is not supported between `Tile[.., [M, N]]` and `Tile[.., [M, 1]]` `` | Tiles don't broadcast: wrap the smaller one in `wp.tile_broadcast(t, shape=(M, N))`. |
| `No matching overload found for function warp.tile_store` / `tile_load` | Tile rank ≠ array rank (e.g. 2D tile into `array1d`). |
| `Cannot evaluate type-level shape DSL call: <reason>` | A shape rule rejected the call (bad reshape count, axis out of bounds, impossible broadcast). The reason is the real error. |
| `Too many indices for tensor: got 3, expected at most 2` | Indexing an array with more indices than dimensions. |
| `Argument Tile[.., [2, 2, 2]] is not assignable to parameter a with type Tile[.., [@_, @_]]` (detail: `Tensor rank mismatch: expected 2 dimensions, got 3 dimensions`) | Rank-specific API (e.g. `tile_transpose`) given another rank. |
| `X is not assignable to array[float, [int, int, int]]` | Passing a 2D array where `array3d` is expected. |
| `` `M` is not a valid type alias: Number literal cannot be used in annotations `` / `Tensor shape dimensions must be integer literals or type variables` | A module constant used as a dimension in an annotation. No spelling works in Pyrefly 1.4 (`Final`, `Literal` aliases and `wp.constant` all fail). To check against the constants, use `expect_shape(t, (TILE_M, TILE_N))` (see below); otherwise write literals. |
| ``reveal_type must be imported from typing for runtime usage`` | Add `from typing import reveal_type, assert_type` (Python ≥ 3.11). |
| `` `(x: Any, ...) -> Tile[...]` is not subscriptable `` | `wp.tile[float, M, N]` in a `@wp.func` signature. Correct Warp, but Pyrefly 1.4 cannot express a name that is both the `wp.tile(x)` builtin and a variadic shape type; the function's tile parameters become `Unknown`. Upstream stubs have the same problem. Leave the Warp syntax as is. |
| `` `+` is not supported between `Tile[...]` and `float` `` / `` Unary `+` is not supported `` | A real Warp error: tiles have no scalar add/sub and no unary plus. Use `t + wp.tile_full(...)` / drop the `+`. |
| `` `int32` is not assignable to any of constraints `float`, ... `` | Integer tiles passed to `tile_matmul` / Cholesky / solves: Warp rejects them too. |
| ``Attribute `x` cannot depend on type variable `_ArrayDType` `` | `self.x = wp.array2d(...)`: a Pyrefly 1.4 bug with calling generic aliases. Suppress narrowly: `# pyrefly: ignore[invalid-type-var]`. |
| `No matching overload found for function warp.max called with arguments: (float64, float)` / `` `+` is not supported between `float` and `float64` `` | A real Warp error: no implicit conversion between scalar types; a literal like `0.0` is `float32`. Use `wp.float64(0.0)` for double precision. |
| `` `*` is not supported between `Vector[...]` and `Vector[...]` `` / `... and float64` / `Matrix[.., Literal[4], ..] and Vector[.., Literal[3]]` | A real Warp error: no element-wise `*` for vectors (`wp.cw_mul`), scalar types must match, matrix/vector sizes must agree. |
| `array[float, [int]] is not assignable to array[float64, [int]]` | Different dtypes (Warp rejects this at launch too). `float` and `wp.float32` are interchangeable. |

## What remains gradual (and is fine)

- Arrays from untyped sources (`wp.array(ptr=...)`, data loaded from files,
  `np.ndarray` without shape stubs), and `wp.array[dtype]` parameters.
- `wp.launch(kernel, inputs=[...])`: arguments are not checked against the
  kernel signature.
- Data-dependent extents (stream compaction counts, query results).

A gradual value satisfies any precise annotation, so **0 errors does not prove
shapes were inferred**. Confirm important flows with a `reveal_type` probe.
