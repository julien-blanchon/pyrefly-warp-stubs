---
name: add-shape-types-to-warp-project
description: >
  Add Pyrefly tensor-shape typing to an NVIDIA Warp project using
  pyrefly-warp-stubs: shaped host-side array signatures (wp.array[Any, [N, 3]]),
  checked tile shapes inside @wp.kernel bodies (tile_load, tile_matmul,
  broadcasts, reductions), and in-kernel shape assertions that Warp ignores. Use
  whenever the user wants to shape-check, shape-annotate or "port to shape
  typing" Warp kernels or Warp-using code, sees Pyrefly shape errors in Warp
  code, or asks why a Warp shape is Unknown. Invoke BEFORE editing: Warp
  evaluates some annotations at runtime, and the gated workflow prevents edits
  that type-check but break kernels.
---

You are adding Pyrefly shape typing to code that uses NVIDIA Warp.

**Shape typing must never change runtime behaviour.** Warp reads kernel
signatures, `@wp.func` signatures and `@wp.struct` fields at runtime, and builds
kernels from their source. An annotation that Pyrefly accepts can still break a
kernel build or a module import. Every rule below that differs from ordinary
Python typing exists for that reason, and the final gate is running the code.

Read `warp_shape_reference.md` (this directory) first: what is tracked, the
runtime rules, and an error decoder. `verify_port.py` lints the runtime rules
that Pyrefly cannot see.

Pick the depth from the request:

- **Own project / application code** (common): type host-side boundaries, make
  kernels shape-checked, add assertions at shape-changing steps. Hand back the
  code and a short report of what stayed untracked.
- **Library with a public API**: as above, plus decide whether public
  signatures use shaped annotations (readable only by Pyrefly) or the portable
  `Shaped[...]` spelling (adds a runtime dependency), see Gate 2.
- **Large codebase migration**: keep the existing structure and tests; precise
  contracts at module boundaries first, internal precision second. Remove
  temporary probes before handing off.

Use subagents only for bounded read-only work (inventorying kernels, clustering
errors by cause). One agent owns the edits.

# Gate 0: Environment and baseline

1. Check the project's Python: the stubs and the PEP 695 syntax used here
   (`def f[N: IntVar](...)`) need **Python ≥ 3.12**. On 3.10/3.11, stop and
   tell the user: shape typing isn't supported there.
2. Install the stubs as a **dev** dependency (they are type-check-only):
   `uv add --dev "pyrefly-warp-stubs @ git+https://github.com/julien-blanchon/pyrefly-warp-subs"`.
   Pyrefly ≥ 1.4 is required.
3. Make sure a `pyrefly.toml` or `[tool.pyrefly]` exists. Pyrefly's `basic` mode,
   used when there is no config, **hides shape errors**.
4. Prove the stubs are active with a throwaway probe file:
   ```python
   from typing import reveal_type
   import warp as wp
   reveal_type(wp.zeros((2, 3), dtype=float))  # must be array[Any, [2, 3]]
   ```
   `array[Unknown]` or bare `array` means the stubs are not on the path:
   check `pyrefly dump-config` and the interpreter Pyrefly uses. Fix the
   environment; never weaken annotations to get a clean check.
5. Record the baseline: `pyrefly check` error count for the target files
   **before** editing, and whether the project's tests or example run passes.
   Only the baseline tells new errors from pre-existing ones.
6. Note the installed `warp-lang` version. The stubs are generated from 1.18;
   on a newer Warp, functions added since are reported as missing. Report it;
   don't work around it.

# Gate 1: Inventory and op audit

List, for the selected scope:

- every `@wp.kernel`, `@wp.func` and `@wp.struct`, with its array/tile parameters;
- host functions that create, reshape or pass Warp arrays;
- tile sizes and where they come from (literal, module int, `wp.constant`,
  closure of a kernel factory, function argument).

For each shape-relevant operation, look it up in the reference tables (or in
`warp-stubs/__init__.pyi`) and classify it: **tracked** (shaped signature or a
`*_shape(...)` rule), **gradual** (upstream `Tile[Any, tuple[int, ...]]`,
`Array[Any]`, bare `array`), or **boundary** (`wp.launch` inputs, interop,
data-dependent sizes). This is diagnostic: gradual operations are fine; they
tell you where shapes will stop flowing.

**Stop early when there is little to gain.** If the scope has no tiles and
every size comes from a runtime value (particle counts, mesh sizes), the useful
work is small: rank the kernel parameters (Step 1), declare array attributes at
class level (Step 4), and say so in the report rather than adding annotations
that only restate `int`.

# Gate 2: Decide the annotation surface

- **Kernel and `@wp.func` signatures stay as Warp syntax**: `wp.array2d[dtype]`,
  `wp.tile[dtype, M, N]`, value types. Never put `[N, 3]` shapes, `Tile[...]`
  or `IntVar` there.
- **Host code** gets shaped annotations, PEP 695 type parameters, and
  `from __future__ import annotations`. Import `shape_extensions` names under
  `if TYPE_CHECKING:` so it is not a runtime dependency.
- **Public library API**: `wp.array[Any, [N, 3]]` is only meaningful to Pyrefly;
  other checkers see an untyped parameter. If that matters, use
  `Shaped[wp.array[Any], "[N, 3]"]` inside a `@shape_vars("N")` function. That
  decorator runs at import, so `pyrefly-shape-extensions` becomes a runtime
  dependency. Ask the user if the request doesn't settle it.

# Module loop (one kernel or host function at a time)

## Step 1: Make shape sources visible

- Tile sizes that are already constant should be module-level ints or
  `wp.constant(...)` (arithmetic like `wp.constant(4 * NUM_FREQ)` is tracked).
  **Do not change their values or move runtime computation.** Sizes from a
  kernel factory's closure or a function argument become `int` dimensions: the
  rank is kept, which is acceptable.
- In kernel signatures, `wp.array[dtype]` → `wp.array1d[dtype]` is safe: Warp
  builds the identical annotation, and Pyrefly gains the rank. Apply it only
  where the parameter really is 1-D (in a kernel signature it always is).
  `verify_port.py` lists these as INFO.

## Step 2: Probe

Inside the kernel, add a **guarded** probe block and run `pyrefly check`:

```python
from typing import TYPE_CHECKING, assert_type, reveal_type
...
t = wp.tile_load(a, shape=(M, N))
if TYPE_CHECKING:  # Warp drops this block when building the kernel
    reveal_type(t)
```

`TYPE_CHECKING` is a compile-time constant, so Warp never compiles the block.
A bare `reveal_type`/`assert_type` in a kernel body breaks the kernel build
(`WarpCodegenError`), and because Warp builds all kernels of a module together,
**one unguarded probe breaks every kernel in that module**, whichever is
launched. `if not TYPE_CHECKING: ... else: <probe>` is not dropped either; use
the plain `if TYPE_CHECKING:` form.

For each shape-relevant local, record: shaped (`Tile[float, [4, 8]]`), partly
known (`[int, 8]`), or gradual (`Tile[float]`, `Unknown`).

## Step 3: Diagnose losses

Trace a gradual value upstream to the first operation that lost the shape.
Typical causes, in order of likelihood:

1. An untracked operation (gradual in the Gate 1 audit). Keep it; shapes
   resume at the next tracked op that takes an explicit `shape=`.
2. A size that is a runtime value: `int` dims are expected; don't refactor.
3. A rank-less source (`wp.array[dtype]` parameter, untyped host data): fix
   with Step 1 or a host-side annotation.
4. Real stub gap: report it with the operation and a probe. Change the stubs
   only if the user asked for stub work.

Errors: read them with the reference's error decoder. A shape error in
existing, working code is usually either a **real bug** (report it, with the
reason) or a stub limitation (also in the reference). Do not silence real
errors with `cast`. Do not "fix" working runtime code to satisfy the checker
without the user's approval.

## Step 4: Write permanent checks

- **Kernels**: keep a few checkpoints at shape-changing steps (after
  `tile_matmul`, reshapes, reductions, broadcasts), inside `if TYPE_CHECKING:`.
  Prefer `expect_shape(t, (TILE_M, TILE_N))` (from `warp.shape_checks`, imported
  under `TYPE_CHECKING`): it names the constants, so it stays correct when they
  change. `assert_type(t, Tile[float, [8, 4]])` needs literal dimensions. Import
  `Tile` under `if TYPE_CHECKING:` too.
- **Arrays held on objects**: declare them at class level
  (`x: wp.array1d[wp.vec3]`), rather than relying on what `__init__` assigns.
- **Type aliases** for shaped types: PEP 695 only (`type Points = wp.array[Any, [int, 3]]`).
- **Host functions**: type the signature (shapes bind at call sites, which is
  where most value lies). `assert_type`/`cast` in **executed** host code must
  quote the type (`assert_type(x, "wp.array[Any, [N, 3]]")`) or sit under
  `if TYPE_CHECKING:`. Unquoted, Python evaluates `wp.array[Any, [N, 3]]` when
  the line runs and Warp raises `TypeError: ndim must be an integer`.
- **Typed launchers carry the kernel's input shapes.** Kernel signatures stay
  plain Warp and `wp.launch(inputs=[...])` is unchecked, so give the host
  function that launches the kernel the shapes, with named dimensions as type
  parameters: `def matmul[M: IntVar, K: IntVar, N: IntVar](a: wp.array[float, [M, K]], b: wp.array[float, [K, N]]) -> wp.array[float, [M, N]]`.
  Build outputs from the inputs' shapes (`wp.zeros((a.shape[0], b.shape[1]))`)
  so the result type follows. Annotations need literal sizes or type
  parameters (not module constants). See `examples/` in the stubs repository.
- Bind sizes from integer parameters with `count: Int[N]`, for example
  `def particles[N: IntVar](count: Int[N]) -> wp.array[float, [N, 3]]`.
- Allocate with a shape and fill (`a = wp.empty((64, 16), dtype=float); a.assign(np_data)`)
  rather than `wp.array(np_data)` when the NumPy data's shape is unknown to the
  checker: the shape then stays known.
- Remove temporary `reveal_type` probes.
- Mind contributors without the stubs: other checkers (and Pyrefly without
  `pyrefly-warp-stubs`) report in-kernel `assert_type`/`Tile[...]` checkpoints as
  errors. Keep them few and meaningful, and mention it in the report.
- Dtypes: `float` and `wp.float32` are the same type (as in Warp), so
  `Tile[float, ...]` and `Tile[wp.float32, ...]` are interchangeable in
  assertions; reveals display `float`.

## Step 5: Post-step check

`pyrefly check` on the file: no new errors versus the baseline other than real
bugs you are reporting, and `verify_port.py` reports no ERROR.

# Verification

1. `python <skill dir>/verify_port.py <files>`: zero ERRORs; review WARN/INFO.
2. `pyrefly check`: compare against the Gate 0 baseline. Every new error is
   either fixed or listed in the report as a real bug or a known limitation.
3. **Run the code**: the project's tests, or launch every touched kernel once
   (CPU is fine: `device="cpu"`). Kernel builds happen at first launch, so an
   import alone proves nothing. This is the gate for every runtime rule.
4. **0 errors does not prove inference.** A gradual value satisfies any precise
   annotation. Confirm the important flows with the probes from Step 2.

# Completion report

```
## Shape typing: <project or files>
- Kernels: N touched, M with in-kernel assertions; tile shapes tracked in: ...
- Host functions: N typed (list), dims bound from: ...
- Untracked / gradual (with reason): op or value → reason
- Real bugs found: file:line → explanation
- Stub limitations hit: ...
- Pyrefly errors: baseline X → now Y (new: list); verify_port: 0 errors
- Runtime: tests / kernel launches run, result
```
