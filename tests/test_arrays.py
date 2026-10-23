"""Host-side array shapes: creation, views and the rank aliases.

`check_*` functions are only type checked (an `E:` comment marks an expected Pyrefly
error); `test_*` functions run against the real Warp on the CPU, and their
`assert_shape` calls are checked both statically and at runtime.
"""

from __future__ import annotations

from typing import Any, assert_type

import numpy as np
import pytest
import warp as wp
from shape_extensions import IntTuple, assert_shape


def check_creation(n: int) -> None:
    # `float` *is* `wp.float32` (Warp canonicalises it), so either spelling works.
    assert_type(wp.zeros((128, 3), dtype=wp.float32), wp.array[float, [128, 3]])
    assert_type(wp.zeros((128, 3), dtype=wp.float32), wp.array[wp.float32, [128, 3]])
    assert_type(wp.zeros(10, dtype=wp.vec3), wp.array[wp.vec3, [10]])
    assert_type(wp.zeros([4, 5]), wp.array[float, [4, 5]])  # Warp's default dtype
    assert_type(wp.ones((2, 3, 4), dtype=wp.float64), wp.array[wp.float64, [2, 3, 4]])
    assert_type(wp.empty(7, dtype=int), wp.array[wp.int32, [7]])
    assert_type(wp.full((2, 2), 1.5, dtype=float), wp.array[float, [2, 2]])
    assert_type(wp.full((2, 2), wp.vec3(1.0, 2.0, 3.0)), wp.array[wp.vec3, [2, 2]])
    # A size that is only known at runtime keeps the rank.
    assert_type(wp.zeros(n, dtype=float), wp.array[float, [int]])
    assert_type(wp.zeros((n, 3), dtype=float), wp.array[float, [int, 3]])
    # Types built by Warp's factories are vectors (the length is widened to `int`).
    assert_type(wp.zeros(3, dtype=wp.types.vector(length=5, dtype=float)), wp.array[wp.Vector[float, int], [3]])


def check_constructor(data: np.ndarray, ptr: int) -> None:
    assert_type(wp.array(shape=(2, 3), dtype=float), wp.array[float, [2, 3]])
    assert_type(wp.array(shape=5, dtype=wp.vec3), wp.array[wp.vec3, [5]])
    assert_type(wp.array([[1.0, 2.0], [3.0, 4.0]], dtype=wp.float32), wp.array[float, [2, 2]])
    # Without `dtype`, Warp infers it from the data: not tracked.
    assert_type(wp.array([1, 2, 3]), wp.array[Any, [3]])
    assert_type(wp.array(ptr=ptr, dtype=float, shape=(4,), capacity=16), wp.array[float, [4]])


def check_dtype_identity(
    f32: wp.array1d[wp.float32], f: wp.array1d[float], f64: wp.array1d[wp.float64]
) -> None:
    # Warp passes a `dtype=float` array to a `wp.float32` parameter and back.
    need_f32: wp.array1d[wp.float32] = f
    need_f: wp.array1d[float] = f32
    need_f64: wp.array1d[wp.float64] = f  # E: `array[float, [int]]` is not assignable to `array[float64, [int]]`


def check_like(a: wp.array[wp.float32, [3, 4]]) -> None:
    assert_type(wp.zeros_like(a), wp.array[wp.float32, [3, 4]])
    assert_type(wp.ones_like(a), wp.array[wp.float32, [3, 4]])
    assert_type(wp.empty_like(a), wp.array[wp.float32, [3, 4]])
    assert_type(wp.full_like(a, 2.0), wp.array[wp.float32, [3, 4]])
    assert_type(wp.clone(a), wp.array[wp.float32, [3, 4]])
    assert_type(a.to("cpu"), wp.array[wp.float32, [3, 4]])
    assert_type(a.contiguous(), wp.array[wp.float32, [3, 4]])
    assert_type(a.grad, wp.array[wp.float32, [3, 4]])


def check_views(a: wp.array[wp.float32, [6, 4]]) -> None:
    assert_type(a.shape, IntTuple[6, 4])
    assert_type(a.reshape((3, 8)), wp.array[wp.float32, [3, 8]])
    assert_type(a.reshape((-1, 2)), wp.array[wp.float32, [12, 2]])
    assert_type(a.reshape(24), wp.array[wp.float32, [24]])
    assert_type(a.flatten(), wp.array[wp.float32, [24]])
    assert_type(a.transpose(), wp.array[wp.float32, [4, 6]])
    assert_type(a.transpose((1, 0)), wp.array[wp.float32, [4, 6]])
    assert_type(a[1:5], wp.array[wp.float32, [4, 4]])
    assert_type(a[:, ::2], wp.array[wp.float32, [6, 2]])
    a.reshape((5, 5))  # E: reshape target element count does not match the input
    a.reshape((7, -1))  # E: reshape cannot infer an integral dimension
    a.transpose((0, 0))  # E: transpose axes must be unique


def check_rank_aliases(
    a2: wp.array2d[float], a3: wp.array3d[float], any_rank: wp.array[float]
) -> None:
    assert_type(a2, wp.array[float, [int, int]])
    matrix: wp.array2d[float] = wp.zeros((3, 4), dtype=float)
    flexible: wp.array[float] = wp.zeros((3, 4), dtype=float)
    volume: wp.array3d[float] = wp.zeros((3, 4), dtype=float)  # E: `array[float, [3, 4]]` is not assignable to `array[float, [int, int, int]]`
    # An array of unknown rank is accepted wherever a rank is expected.
    from_any: wp.array2d[float] = any_rank
    same: wp.array3d[float] = a3


def check_numpy_interop(data: np.ndarray) -> None:
    # Without shape-aware NumPy stubs only the rank of `np.zeros(...)` is known.
    assert_type(wp.from_numpy(np.zeros((3, 4))), wp.array[Any, [int, int]])
    assert_type(wp.from_numpy(data, shape=(2, 6)), wp.array[Any, [2, 6]])


def test_creation() -> None:
    assert_shape(wp.zeros((128, 3), dtype=wp.float32, device="cpu").shape, (128, 3))
    assert_shape(wp.zeros(10, dtype=wp.vec3, device="cpu").shape, (10,))
    assert_shape(wp.zeros([4, 5], device="cpu").shape, (4, 5))
    assert_shape(wp.ones((2, 3, 4), dtype=float, device="cpu").shape, (2, 3, 4))
    assert_shape(wp.full((2, 2), 1.5, dtype=float, device="cpu").shape, (2, 2))
    assert_shape(wp.array(shape=(2, 3), dtype=float, device="cpu").shape, (2, 3))
    assert_shape(wp.array([[1.0, 2.0], [3.0, 4.0]], dtype=float, device="cpu").shape, (2, 2))


def test_views() -> None:
    a = wp.zeros((6, 4), dtype=wp.float32, device="cpu")
    assert_shape(a.reshape((3, 8)).shape, (3, 8))
    assert_shape(a.reshape((-1, 2)).shape, (12, 2))
    assert_shape(a.reshape(24).shape, (24,))
    assert_shape(a.flatten().shape, (24,))
    assert_shape(a.transpose().shape, (4, 6))
    assert_shape(a.transpose((1, 0)).shape, (4, 6))
    assert_shape(a[1:5].shape, (4, 4))
    assert_shape(a[:, ::2].shape, (6, 2))
    assert_shape(wp.zeros_like(a).shape, (6, 4))


def test_rejected_views() -> None:
    a = wp.zeros((6, 4), dtype=wp.float32, device="cpu")
    with pytest.raises(RuntimeError):
        a.reshape((5, 5))  # E: reshape target element count does not match the input
    with pytest.raises(RuntimeError):
        a.reshape((7, -1))  # E: reshape cannot infer an integral dimension


def test_dtype_canonicalisation() -> None:
    # The stubs alias `float`/`int`/`bool` to `wp.float32`/`wp.int32`/`wp.bool`
    # because Warp makes them identical:
    assert wp.zeros(1, dtype=float, device="cpu").dtype is wp.float32
    assert wp.zeros(1, dtype=int, device="cpu").dtype is wp.int32
    assert wp.zeros(1, dtype=bool, device="cpu").dtype is wp.bool
    assert_shape(wp.zeros((2, 3), dtype=float, device="cpu").shape, (2, 3))
