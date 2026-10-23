"""Blocked matrix multiplication with tiles, typed `[M, K] @ [K, N] -> [M, N]`.

Shows: a host function generic in its dimensions (`M`, `K`, `N` are bound at
each call and flow to the result), tile loads at offsets, and in-place
accumulation with `wp.tile_matmul(a, b, out)`. Swapped operands are reported
before anything runs:

    matmul(b, a)
    # error: Argument `array[float, [64, 32]]` is not assignable to parameter
    #        `b` with type `array[float, [48, @_]]`

Run it with `python examples/blocked_matmul.py` (CPU is fine).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from warp.shape_checks import expect_shape

TILE_M = 16
TILE_N = 16
TILE_K = 8


@wp.kernel
def gemm_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    acc = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=float)
    for k in range(a.shape[1] // TILE_K):
        ta = wp.tile_load(a, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
        tb = wp.tile_load(b, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))
        wp.tile_matmul(ta, tb, acc)  # acc += ta @ tb; the shapes must chain
    if TYPE_CHECKING:
        expect_shape(acc, (TILE_M, TILE_N))
    wp.tile_store(c, acc, offset=(i * TILE_M, j * TILE_N))


def matmul[M: IntVar, K: IntVar, N: IntVar](
    a: wp.array[float, [M, K]], b: wp.array[float, [K, N]]
) -> wp.array[float, [M, N]]:
    """`a @ b`; every dimension must be a multiple of the tile sizes."""
    c = wp.zeros((a.shape[0], b.shape[1]), dtype=float, device=a.device)
    wp.launch_tiled(
        gemm_kernel,
        dim=(a.shape[0] // TILE_M, b.shape[1] // TILE_N),
        inputs=[a, b, c],
        block_dim=64,
        device=a.device,
    )
    return c


def main() -> None:
    rng = np.random.default_rng(0)
    a = wp.empty((64, 32), dtype=float, device="cpu")
    b = wp.empty((32, 48), dtype=float, device="cpu")
    a.assign(rng.standard_normal(a.shape).astype(np.float32))
    b.assign(rng.standard_normal(b.shape).astype(np.float32))

    c = matmul(a, b)  # array[float, [64, 48]]

    np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy(), rtol=1e-4, atol=1e-4)
    print("ok: product", c.shape, "matches NumPy")


if __name__ == "__main__":
    main()
