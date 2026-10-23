"""Tile shapes inside kernels.

`check_*` functions are only type checked (an `E:` comment marks an expected
Pyrefly error). `test_*` functions launch the equivalent kernels with the real
Warp on the CPU: well-shaped kernels must compute what NumPy computes, and every
kernel Pyrefly rejects must also be rejected by Warp when it builds the kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, assert_type

import numpy as np
import warp as wp

from _launch import array, assert_rejected, run_tiled, zeros

if TYPE_CHECKING:
    from warp import Tile
    from warp.shape_checks import expect_shape

M = 4
K = 8
N = 2
TILE = wp.constant(16)

# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------


def check_construction() -> None:
    assert_type(wp.tile_zeros(shape=(M, K), dtype=float), Tile[float, [4, 8]])
    assert_type(wp.tile_ones(shape=TILE, dtype=wp.float32), Tile[wp.float32, [16]])
    assert_type(wp.tile_full(shape=(2, 3), value=1.0, dtype=float), Tile[float, [2, 3]])
    assert_type(wp.tile_empty(shape=(M, N, K), dtype=int), Tile[int, [4, 2, 8]])
    assert_type(wp.tile_zeros(shape=(M, K)), Tile[Any, [4, 8]])
    assert_type(wp.tile_arange(8), Tile[wp.float32, [8]])
    assert_type(wp.tile_arange(2, 10, dtype=int), Tile[int, [8]])
    assert_type(wp.tile_arange(2, 10, 3, dtype=int), Tile[int, [3]])
    assert_type(wp.tile_arange(10, 0, -3, dtype=int), Tile[int, [4]])
    assert_type(wp.tile_randf(shape=(M, K), rng=wp.uint32(0)), Tile[wp.float32, [4, 8]])


def check_load_store(a2: wp.array2d[float], a1: wp.array1d[float], any_rank: wp.array[float]) -> None:
    t = wp.tile_load(a2, shape=(M, K))
    assert_type(t, Tile[float, [4, 8]])
    assert_type(wp.tile_load(a2, shape=(M, K), offset=(1, 0)), Tile[float, [4, 8]])
    assert_type(wp.tile_load(a1, shape=K), Tile[float, [8]])
    assert_type(wp.tile_load(a1, shape=(K,)), Tile[float, [8]])
    # An array of unknown rank (`wp.array[float]`) takes the tile's rank.
    assert_type(wp.tile_load(any_rank, shape=(M, K)), Tile[float, [4, 8]])
    wp.tile_store(a2, t)
    wp.tile_store(any_rank, t)
    assert_type(wp.tile_atomic_add(a2, t), Tile[float, [4, 8]])
    wp.tile_load(a2, shape=K)  # E: No matching overload found for function `warp.tile_load`
    wp.tile_store(a1, t)  # E: No matching overload found for function `warp.tile_store`


def check_matmul(a: wp.array2d[float], b: wp.array2d[float]) -> None:
    ta = wp.tile_load(a, shape=(M, K))
    tb = wp.tile_load(b, shape=(K, N))
    assert_type(wp.tile_matmul(ta, tb), Tile[float, [4, 2]])
    out = wp.tile_zeros(shape=(M, N), dtype=float)
    wp.tile_matmul(ta, tb, out)
    wp.tile_matmul(tb, ta)  # E: Shape dimension mismatch: expected Int[2], got Int[4]
    wp.tile_matmul(ta, tb, wp.tile_zeros(shape=(N, M), dtype=float))  # E: No matching overload


def check_elementwise(a: wp.array2d[float]) -> None:
    t = wp.tile_load(a, shape=(M, K))
    assert_type(t + t, Tile[float, [4, 8]])
    assert_type(t * 2.0, Tile[float, [4, 8]])
    assert_type(2.0 * t, Tile[float, [4, 8]])
    assert_type(-t, Tile[float, [4, 8]])
    column = wp.tile_load(a, shape=(M, 1))
    t + column  # E: `+` is not supported between `Tile[float, [4, 8]]` and `Tile[float, [4, 1]]`
    assert_type(t + wp.tile_broadcast(column, shape=(M, K)), Tile[float, [4, 8]])
    wp.tile_broadcast(t, shape=(M, 3))  # E: tile shape cannot be broadcast to the target shape
    assert_type(wp.tile_map(wp.sin, t), Tile[Any, [4, 8]])
    assert_type(wp.tile_map(wp.add, t, t), Tile[Any, [4, 8]])
    wp.tile_map(wp.add, t, column)  # E: Argument `Tile[float, [4, 1]]` is not assignable to parameter `b`


def check_operators(a: wp.array2d[float], ai: wp.array2d[int]) -> None:
    t = wp.tile_load(a, shape=(M, K))
    i = wp.tile_load(ai, shape=(M, K))
    assert_type(t - t, Tile[float, [4, 8]])
    assert_type(t / t, Tile[float, [4, 8]])
    assert_type(t / 2.0, Tile[float, [4, 8]])
    assert_type(2.0 / t, Tile[float, [4, 8]])
    assert_type(t @ t, Tile[float, [4, 8]])  # element-wise, like `*`
    assert_type(t @ 2.0, Tile[float, [4, 8]])
    assert_type(i & i, Tile[int, [4, 8]])
    assert_type(i | i, Tile[int, [4, 8]])
    assert_type(i ^ i, Tile[int, [4, 8]])
    assert_type(len(t), int)
    # Warp has no tile/scalar `add`/`sub`, no unary `+` and no `%` on tiles.
    t + 1.0  # E: `+` is not supported between `Tile[float, [4, 8]]` and `float`
    1.0 - t  # E: `-` is not supported between `float` and `Tile[float, [4, 8]]`
    +t  # E: Unary `+` is not supported on `Tile[float, [4, 8]]`
    t % t  # E: `%` is not supported between `Tile[float, [4, 8]]` and `Tile[float, [4, 8]]`


def check_float_only_linear_algebra(ai: wp.array2d[wp.int64], a: wp.array2d[float]) -> None:
    # `wp.int32` is `int`, which typing lets pass as `float`, so int32 tiles are
    # not caught; the other integer dtypes are.
    i = wp.tile_load(ai, shape=(M, M))
    wp.tile_matmul(i, i)  # E: of type variable `D` # E: of type variable `E`
    wp.tile_cholesky(i)  # E: `int64` is not assignable to any of constraints
    t = wp.tile_load(a, shape=(M, M))
    assert_type(wp.tile_matmul(t, wp.tile_zeros(shape=(M, N), dtype=wp.float32)), Tile[float, [4, 2]])
    assert_type(wp.tile_astype(i, wp.float32), Tile[wp.float32, [4, 4]])
    assert_type(wp.tile_astype(t, wp.int64), Tile[wp.int64, [4, 4]])


def check_expect_shape(a: wp.array2d[float], n: int) -> None:
    t = wp.tile_load(a, shape=(M, TILE))
    if TYPE_CHECKING:  # `expect_shape` exists only for the type checker
        expect_shape(t, (M, TILE))
        expect_shape(t, (4, 16))
        expect_shape(wp.zeros((n, 3)), (n, 3))
        expect_shape(t, (TILE, M))  # E: Argument `tuple[Int[16], Literal[4]]` is not assignable to parameter `shape`
        expect_shape(t, (M,))  # E: Argument `tuple[Literal[4]]` is not assignable to parameter `shape`


def check_layout(a: wp.array2d[float]) -> None:
    t = wp.tile_load(a, shape=(M, K))
    assert_type(wp.tile_transpose(t), Tile[float, [8, 4]])
    assert_type(wp.tile_reshape(t, shape=(K, M)), Tile[float, [8, 4]])
    assert_type(wp.tile_reshape(t, shape=(-1,)), Tile[float, [32]])
    assert_type(wp.tile_reshape(t, shape=32), Tile[float, [32]])
    assert_type(wp.tile_squeeze(wp.tile_load(a, shape=(1, K))), Tile[float, [8]])
    assert_type(wp.tile_squeeze(wp.tile_load(a, shape=(1, 1)), axis=(0,)), Tile[float, [1]])
    assert_type(wp.tile_view(t, offset=(0, 0), shape=(2, 2)), Tile[float, [2, 2]])
    assert_type(wp.tile_astype(t, wp.float64), Tile[wp.float64, [4, 8]])
    wp.tile_reshape(t, shape=(3, 3))  # E: reshape target element count does not match the input
    wp.tile_transpose(wp.tile_zeros(shape=(M, K, N), dtype=float))  # E: `Tile[float, [4, 8, 2]]` is not assignable to parameter `a`


def check_indexing(a: wp.array2d[float]) -> None:
    t = wp.tile_load(a, shape=(M, K))
    assert_type(t[0, 1], float)
    assert_type(wp.tile_extract(t, 0, 1), float)
    assert_type(t[1], Tile[float, [8]])
    assert_type(t[1:3, :], Tile[float, [2, 8]])
    assert_type(t[:, ::2], Tile[float, [4, 4]])
    wp.tile_extract(t, 0)  # E: `Tile[float, [4, 8]]` is not assignable to parameter `a`


def check_reductions(a: wp.array2d[float]) -> None:
    t = wp.tile_load(a, shape=(M, K))
    assert_type(wp.tile_sum(t), Tile[float, [1]])
    assert_type(wp.tile_sum(t, axis=0), Tile[float, [8]])
    assert_type(wp.tile_sum(t, axis=1), Tile[float, [4]])
    assert_type(wp.tile_sum(t, axis=-1), Tile[float, [4]])
    assert_type(wp.tile_max(t), Tile[float, [1]])
    assert_type(wp.tile_argmin(t), Tile[wp.int32, [1]])
    assert_type(wp.tile_reduce(wp.add, t, axis=0), Tile[float, [8]])
    assert_type(wp.tile_scan_inclusive(t), Tile[float, [4, 8]])
    assert_type(wp.tile_dot(t, t), Tile[float, [1]])
    wp.tile_sum(t, axis=2)  # E: reduction axis out of bounds


def check_linear_algebra(a: wp.array2d[float], b: wp.array1d[float]) -> None:
    A = wp.tile_load(a, shape=(M, M))
    y = wp.tile_load(b, shape=M)
    L = wp.tile_cholesky(A)
    assert_type(L, Tile[float, [4, 4]])
    assert_type(wp.tile_cholesky_solve(L, y), Tile[float, [4]])
    assert_type(wp.tile_lower_solve(L, wp.tile_load(a, shape=(M, N))), Tile[float, [4, 2]])
    assert_type(wp.tile_diag_add(A, y), Tile[float, [4, 4]])
    wp.tile_cholesky(wp.tile_load(a, shape=(M, K)))  # E: Shape dimension mismatch
    wp.tile_cholesky_solve(L, wp.tile_load(b, shape=K))  # E: No matching overload


# ---------------------------------------------------------------------------
# Runtime: the same operations, launched on the CPU
# ---------------------------------------------------------------------------


@wp.kernel(module="unique")
def matmul_kernel(a: wp.array2d[float], b: wp.array2d[float], out: wp.array2d[float]):
    ta = wp.tile_load(a, shape=(M, K))
    tb = wp.tile_load(b, shape=(K, N))
    product = wp.tile_matmul(ta, tb)
    if TYPE_CHECKING:  # dropped by Warp when it builds the kernel
        expect_shape(product, (M, N))
        assert_type(product, Tile[float, [4, 2]])
    wp.tile_store(out, product)


def test_matmul() -> None:
    a = np.arange(M * K, dtype=np.float32).reshape(M, K)
    b = np.arange(K * N, dtype=np.float32).reshape(K, N)
    out = zeros((M, N))
    run_tiled(matmul_kernel, array(a), array(b), out)
    np.testing.assert_allclose(out.numpy(), a @ b)


@wp.kernel(module="unique")
def layout_kernel(
    a: wp.array2d[float],
    transposed: wp.array2d[float],
    flat: wp.array1d[float],
    rows: wp.array2d[float],
    column_sums: wp.array1d[float],
    total: wp.array1d[float],
):
    t = wp.tile_load(a, shape=(M, K))
    wp.tile_store(transposed, wp.tile_transpose(t))
    wp.tile_store(flat, wp.tile_reshape(t, shape=(-1,)))
    wp.tile_store(rows, t[1:3, :])
    wp.tile_store(column_sums, wp.tile_sum(t, axis=0))
    wp.tile_store(total, wp.tile_sum(t))


def test_layout() -> None:
    a = np.arange(M * K, dtype=np.float32).reshape(M, K)
    transposed, flat, rows = zeros((K, M)), zeros((M * K,)), zeros((2, K))
    column_sums, total = zeros((K,)), zeros((1,))
    run_tiled(layout_kernel, array(a), transposed, flat, rows, column_sums, total)
    np.testing.assert_allclose(transposed.numpy(), a.T)
    np.testing.assert_allclose(flat.numpy(), a.reshape(-1))
    np.testing.assert_allclose(rows.numpy(), a[1:3, :])
    np.testing.assert_allclose(column_sums.numpy(), a.sum(axis=0))
    np.testing.assert_allclose(total.numpy(), [a.sum()])


@wp.kernel(module="unique")
def broadcast_kernel(a: wp.array2d[float], column: wp.array2d[float], out: wp.array2d[float]):
    t = wp.tile_load(a, shape=(M, K))
    c = wp.tile_load(column, shape=(M, 1))
    wp.tile_store(out, t + wp.tile_broadcast(c, shape=(M, K)))


def test_broadcast() -> None:
    a = np.arange(M * K, dtype=np.float32).reshape(M, K)
    column = np.arange(M, dtype=np.float32).reshape(M, 1)
    out = zeros((M, K))
    run_tiled(broadcast_kernel, array(a), array(column), out)
    np.testing.assert_allclose(out.numpy(), a + column)


@wp.kernel(module="unique")
def arange_kernel(out: wp.array1d[float]):
    wp.tile_store(out, wp.tile_arange(2, 10, 3, dtype=float))


def test_arange() -> None:
    out = zeros((3,))
    run_tiled(arange_kernel, out)
    np.testing.assert_allclose(out.numpy(), np.arange(2, 10, 3))


# Kernels Pyrefly rejects. Each must also fail to build in Warp.


@wp.kernel(module="unique")
def bad_matmul_kernel(a: wp.array2d[float], b: wp.array2d[float], out: wp.array2d[float]):
    ta = wp.tile_load(a, shape=(M, K))
    tb = wp.tile_load(b, shape=(K, N))
    wp.tile_store(out, wp.tile_matmul(tb, ta))  # E: Shape dimension mismatch


@wp.kernel(module="unique")
def bad_add_kernel(a: wp.array2d[float], out: wp.array2d[float]):
    t = wp.tile_load(a, shape=(M, K))
    c = wp.tile_load(a, shape=(M, 1))
    wp.tile_store(out, t + c)  # E: `+` is not supported


@wp.kernel(module="unique")
def bad_store_rank_kernel(a: wp.array2d[float], out: wp.array1d[float]):
    t = wp.tile_load(a, shape=(M, K))
    wp.tile_store(out, t)  # E: No matching overload


@wp.kernel(module="unique")
def bad_reduce_axis_kernel(a: wp.array2d[float], out: wp.array1d[float]):
    t = wp.tile_load(a, shape=(M, K))
    wp.tile_store(out, wp.tile_sum(t, axis=2))  # E: reduction axis out of bounds


@wp.kernel(module="unique")
def bad_broadcast_kernel(a: wp.array2d[float], out: wp.array2d[float]):
    t = wp.tile_load(a, shape=(M, K))
    wp.tile_store(out, wp.tile_broadcast(t, shape=(M, 3)))  # E: cannot be broadcast


@wp.kernel(module="unique")
def bad_reshape_kernel(a: wp.array2d[float], out: wp.array2d[float]):
    t = wp.tile_load(a, shape=(M, K))
    wp.tile_store(out, wp.tile_reshape(t, shape=(3, 3)))  # E: element count does not match


@wp.kernel(module="unique")
def operators_kernel(a: wp.array2d[float], ai: wp.array2d[int], out: wp.array2d[float], out_i: wp.array2d[int]):
    t = wp.tile_load(a, shape=(M, K))
    i = wp.tile_load(ai, shape=(M, K))
    wp.tile_store(out, (t @ t) - t / 2.0 + 2.0 / (t + t) + (t @ 2.0) - (2.0 * t))
    wp.tile_store(out_i, (i & i) | (i ^ i))


def test_operators() -> None:
    a = np.arange(1, M * K + 1, dtype=np.float32).reshape(M, K)
    ai = np.arange(M * K, dtype=np.int32).reshape(M, K)
    out, out_i = zeros((M, K)), zeros((M, K), dtype=int)
    run_tiled(operators_kernel, array(a), wp.array(ai, dtype=int, device="cpu"), out, out_i)
    np.testing.assert_allclose(out.numpy(), a * a - a / 2.0 + 2.0 / (a + a), rtol=1e-6)
    np.testing.assert_array_equal(out_i.numpy(), (ai & ai) | (ai ^ ai))


@wp.kernel(module="unique")
def bad_scalar_add_kernel(a: wp.array2d[float], out: wp.array2d[float]):
    t = wp.tile_load(a, shape=(M, K))
    wp.tile_store(out, t + 1.0)  # E: `+` is not supported


@wp.kernel(module="unique")
def bad_unary_plus_kernel(a: wp.array2d[float], out: wp.array2d[float]):
    t = wp.tile_load(a, shape=(M, K))
    wp.tile_store(out, +t)  # E: Unary `+` is not supported


@wp.kernel(module="unique")
def bad_int_matmul_kernel(a: wp.array2d[wp.int64], out: wp.array2d[wp.int64]):
    t = wp.tile_load(a, shape=(M, M))
    wp.tile_store(out, wp.tile_matmul(t, t))  # E: of type variable `D` # E: of type variable `E`


def test_rejected_kernels() -> None:
    a, b = zeros((M, K)), zeros((K, N))
    assert_rejected(bad_matmul_kernel, a, b, zeros((M, N)))
    assert_rejected(bad_add_kernel, a, zeros((M, K)))
    assert_rejected(bad_store_rank_kernel, a, zeros((M * K,)))
    assert_rejected(bad_reduce_axis_kernel, a, zeros((K,)))
    assert_rejected(bad_broadcast_kernel, a, zeros((M, 3)))
    assert_rejected(bad_reshape_kernel, a, zeros((3, 3)))
    assert_rejected(bad_scalar_add_kernel, a, zeros((M, K)))
    assert_rejected(bad_unary_plus_kernel, a, zeros((M, K)))
    assert_rejected(bad_int_matmul_kernel, zeros((M, M), dtype=wp.int64), zeros((M, M), dtype=wp.int64))
