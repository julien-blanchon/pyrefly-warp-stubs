"""Warp's host objects and object-like kernel values.

Graph capture, recorded launches, kernel and function objects, atomics, volumes,
sparse matrices, `wp.utils` reductions, tapes and tile stacks. `check_*`
functions are only type checked (an `E:` comment marks an expected Pyrefly
error); `test_*` functions run the same APIs on the CPU.
"""

from __future__ import annotations

from typing import Any, assert_type

import numpy as np
import pytest
import warp as wp
import warp.sparse as sparse

from _launch import DEVICE, assert_build_fails


@wp.kernel(module="unique")
def scale_kernel(a: wp.array1d[float], s: float):
    i = wp.tid()
    a[i] = a[i] * s


@wp.func
def squared(x: float) -> float:
    return x * x


def check_kernels_and_launches(a: wp.array1d[float]) -> None:
    assert_type(scale_kernel, wp.Kernel)
    assert_type(squared.key, str)
    assert_type(squared(2.0), float)  # still callable with its signature in kernels
    assert_type(wp.launch(scale_kernel, dim=4, inputs=[a, 2.0]), None)
    assert_type(wp.launch(scale_kernel, dim=4, inputs=[a, 2.0], record_cmd=True), wp.Launch)
    with wp.ScopedCapture(device="cpu") as capture:
        wp.launch(scale_kernel, dim=4, inputs=[a, 2.0])
    wp.capture_launch(capture.graph)
    wp.launch_tiled(scale_kernel, 1, [a], block_dim=32)  # E: No matching overload found for function `warp.launch_tiled`


def check_atomics(a: wp.array1d[float], d: wp.array1d[wp.float64], v: wp.array2d[wp.vec3], i: int) -> None:
    # Atomics return the previous value, of the array's dtype.
    assert_type(wp.atomic_add(a, i, 1.0), float)
    assert_type(wp.atomic_max(v, i, i, wp.vec3(1.0, 2.0, 3.0)), wp.vec3)
    wp.atomic_add(d, i, 1.0)  # E: No matching overload found for function `warp.atomic_add`


def check_misc(values: wp.array1d[float], vectors: wp.array1d[wp.vec3]) -> None:
    assert_type(wp.static(3), int)
    assert_type(wp.utils.array_sum(values), np.floating[Any])
    assert_type(wp.utils.array_sum(vectors), np.ndarray[tuple[int, ...], np.dtype[Any]])
    assert_type(wp.utils.array_inner(values, values), np.floating[Any])
    assert_type(wp.Tape().gradients, dict[wp.array[Any], wp.array[Any]])
    assert_type(values.numpy(), np.ndarray[tuple[int, ...], np.dtype[Any]])


def check_volumes_and_sparse(volume: wp.Volume, A: sparse.BsrMatrix[Any], x: wp.array1d[float]) -> None:
    assert_type(volume.get_voxels(), wp.array[wp.int32, [int, 3]])
    assert_type(A.nrow, int)
    assert_type(A.values, wp.array[Any])
    assert_type(A @ x, wp.array[Any])
    assert_type(sparse.bsr_mv(A, x), wp.array[Any])


@wp.func
def neighbours(query: wp.HashGridQuery[wp.float64]) -> int:  # precision-specific query type
    count = int(0)
    for _ in query:
        count += 1
    return count


def check_tile_stack() -> None:
    s = wp.tile_stack(capacity=8, dtype=int)
    wp.tile_stack_push(s, 3, True)
    value, ok = wp.tile_stack_pop(s)
    assert_type(value, int)


@wp.kernel(module="unique")
def atomics_kernel(a: wp.array1d[float], v: wp.array2d[wp.vec3], old: wp.array1d[float]):
    old[0] = wp.atomic_add(a, 0, 1.0)
    previous = wp.atomic_max(v, 0, 0, wp.vec3(1.0, 2.0, 3.0))
    old[1] = previous[0]


@wp.kernel(module="unique")
def bad_atomic_kernel(d: wp.array1d[wp.float64]):
    wp.atomic_add(d, 0, 1.0)  # E: No matching overload found for function `warp.atomic_add`


def test_kernel_objects_and_capture() -> None:
    a = wp.array(np.ones(4, dtype=np.float32), dtype=float, device=DEVICE)
    assert isinstance(scale_kernel, wp.Kernel) and isinstance(squared, wp.Function)
    cmd = wp.launch(scale_kernel, dim=4, inputs=[a, 2.0], device=DEVICE, record_cmd=True)
    assert isinstance(cmd, wp.Launch)
    with wp.ScopedCapture(device=DEVICE) as capture:
        wp.launch(scale_kernel, dim=4, inputs=[a, 3.0], device=DEVICE)
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(a.numpy(), [3.0] * 4)
    with pytest.raises(RuntimeError, match="should be passed via. keyword"):
        wp.launch_tiled(scale_kernel, 1, [a, 1.0], block_dim=32, device=DEVICE)  # E: No matching overload


def test_atomics() -> None:
    a = wp.array(np.array([5.0], dtype=np.float32), dtype=float, device=DEVICE)
    v = wp.zeros((1, 1), dtype=wp.vec3, device=DEVICE)
    old = wp.zeros(2, dtype=float, device=DEVICE)
    wp.launch(atomics_kernel, dim=1, inputs=[a, v, old], device=DEVICE)
    np.testing.assert_allclose(old.numpy(), [5.0, 0.0])
    np.testing.assert_allclose(a.numpy(), [6.0])
    assert_build_fails(bad_atomic_kernel, wp.zeros(1, dtype=wp.float64, device=DEVICE))


def test_host_reductions_and_objects() -> None:
    values = wp.array(np.arange(4, dtype=np.float32), dtype=float, device=DEVICE)
    vectors = wp.array(np.ones((2, 3), dtype=np.float32), dtype=wp.vec3, device=DEVICE)
    assert isinstance(wp.utils.array_sum(values), np.floating)
    assert isinstance(wp.utils.array_sum(vectors), np.ndarray)
    assert isinstance(wp.utils.array_inner(values, values), np.floating)
    volume = wp.Volume.allocate(min=(0.0, 0.0, 0.0), max=(1.0, 1.0, 1.0), voxel_size=0.25, device=DEVICE)
    voxels = volume.get_voxels()
    assert voxels.dtype is wp.int32 and voxels.shape[1] == 3
    A = sparse.bsr_identity(3, block_type=wp.float32, device=DEVICE)
    x = wp.array(np.arange(3, dtype=np.float32), dtype=float, device=DEVICE)
    assert A.nrow == 3
    np.testing.assert_allclose((A @ x).numpy(), [0.0, 1.0, 2.0])
