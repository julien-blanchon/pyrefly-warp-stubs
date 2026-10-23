"""Kernel-scope element access: `wp.tid()`, array indexing and composite elements.

`check_*` functions are only type checked (an `E:` comment marks an expected
Pyrefly error); `test_*` functions launch the equivalent kernels on the CPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, assert_type

import numpy as np
import warp as wp

from _launch import DEVICE

if TYPE_CHECKING:
    from warp import Tile


def check_thread_index() -> None:
    i = wp.tid()
    assert_type(i + 1, int)
    j, k = wp.tid()
    assert_type(j, int)
    x, y, z = wp.tid()
    assert_type(z, int)


def check_element_access(
    a1: wp.array[float], a2: wp.array2d[wp.float32], a3: wp.array3d[wp.vec3], i: int, j: int, k: int
) -> None:
    assert_type(a1[i], float)
    assert_type(a2[i, j], wp.float32)
    assert_type(a2[i], wp.array[wp.float32, [int]])
    assert_type(a3[i, j, k], wp.vec3)
    a2[i, j] = 1.0
    a2[i, j, k]  # E: Too many indices for tensor: got 3, expected at most 2


def check_composite_elements(v: wp.array[wp.vec3f], m: wp.array[wp.mat33f], q: wp.array[wp.quatf]) -> None:
    # The `vec3f`/`mat33f` element types work with Warp's builtins and operators.
    assert_type(wp.dot(v[0], v[1]), wp.float32)
    assert_type(wp.length(v[0]), wp.float32)
    assert_type(wp.cross(v[0], v[1]), wp.Vector[wp.float32, Literal[3]])
    assert_type(m[0] * v[0], wp.vec3f)
    assert_type(v[0] * 2.0, wp.vec3f)
    assert_type(v[0].x, wp.float32)
    assert_type(wp.transpose(m[0]), wp.Matrix[wp.float32, Any, Any])
    wp.quat_rotate(q[0], v[0])


def check_scalar_arithmetic(a: wp.array1d[wp.float32], i: int) -> None:
    # Warp scalars keep their type through arithmetic and comparisons.
    assert_type(a[i] * 2.0, wp.float32)
    assert_type(2.0 * a[i] + a[i], wp.float32)
    assert_type(-a[i], wp.float32)
    assert_type(wp.sin(a[i] * 2.0), wp.float32)
    assert_type(a[i] < 0.0, bool)
    assert_type(wp.where(a[i] > 0.0, a[i], -a[i]), wp.float32)


def check_indexed_arrays(ia: wp.indexedarray2d[float], i: int, j: int) -> None:
    assert_type(ia[i, j], float)


# Overloads are looked up by name in the kernel's module, so no `module="unique"`.
@wp.kernel
def generic_kernel(x: wp.array[Any]):
    pass


# `wp.overload` is Warp's kernel-overload API, not `typing.overload`.
@wp.overload
def generic_kernel(x: wp.array[wp.float32]): ...


wp.overload(generic_kernel, {"x": wp.array[wp.float64]})


@wp.kernel(module="unique")
def sin_kernel(a: wp.array1d[wp.float32], out: wp.array1d[wp.float32]):
    i = wp.tid()
    out[i] = wp.where(a[i] > 0.0, wp.sin(a[i] * 2.0), -a[i])


@wp.kernel(module="unique")
def checked_kernel(a: wp.array2d[float], out: wp.array2d[float]):
    t = wp.tile_load(a, shape=(2, 3))
    # Warp drops `if TYPE_CHECKING:` blocks when building the kernel (the
    # condition is a compile-time constant), so shape assertions can live here.
    if TYPE_CHECKING:
        assert_type(t, Tile[float, [2, 3]])
        assert_type(t, Tile[float, [3, 2]])  # E: assert_type(Tile[float, [2, 3]], Tile[float, [3, 2]]) failed
    wp.tile_store(out, t * 2.0)


@wp.kernel(module="unique")
def scale_kernel(a: wp.array2d[float], out: wp.array2d[float]):
    i, j = wp.tid()
    out[i, j] = a[i, j] * 2.0


@wp.kernel(module="unique")
def length_kernel(v: wp.array1d[wp.vec3f], out: wp.array1d[float]):
    i = wp.tid()
    out[i] = wp.length(v[i]) + v[i].x


def test_scale() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = wp.zeros((2, 3), dtype=float, device=DEVICE)
    wp.launch(scale_kernel, dim=(2, 3), inputs=[wp.array(a, dtype=float, device=DEVICE), out], device=DEVICE)
    np.testing.assert_allclose(out.numpy(), a * 2.0)


def test_length() -> None:
    v = np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    out = wp.zeros(2, dtype=float, device=DEVICE)
    wp.launch(length_kernel, dim=2, inputs=[wp.array(v, dtype=wp.vec3f, device=DEVICE), out], device=DEVICE)
    np.testing.assert_allclose(out.numpy(), [8.0, 2.0])


def test_sin() -> None:
    a = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
    out = wp.zeros(3, dtype=wp.float32, device=DEVICE)
    wp.launch(sin_kernel, dim=3, inputs=[wp.array(a, dtype=wp.float32, device=DEVICE), out], device=DEVICE)
    np.testing.assert_allclose(out.numpy(), np.where(a > 0, np.sin(a * 2.0), -a), rtol=1e-6)


def test_type_checking_block_in_kernel() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = wp.zeros((2, 3), dtype=float, device=DEVICE)
    wp.launch_tiled(checked_kernel, dim=1, inputs=[wp.array(a, dtype=float, device=DEVICE), out], block_dim=32, device=DEVICE)
    np.testing.assert_allclose(out.numpy(), a * 2.0)


def test_overload_kernel() -> None:
    wp.launch(generic_kernel, dim=1, inputs=[wp.zeros(1, dtype=wp.float32, device=DEVICE)], device=DEVICE)


def test_quoted_assert_type_runs() -> None:
    # Executed host code must quote shaped types: `assert_type` and `cast`
    # evaluate their type argument at runtime, and Warp rejects `[10, 3]`.
    x = wp.zeros((10, 3), dtype=float, device=DEVICE)
    assert_type(x, "wp.array[float, [10, 3]]")
