"""Row-wise softmax with tile reductions and explicit broadcasts.

Shows: axis reductions (`wp.tile_reduce(wp.max, t, axis=1)` drops the axis:
`[ROWS, COLS] -> [ROWS]`), `wp.tile_reshape` to put it back as a column, and
`wp.tile_broadcast` to stretch it, because Warp never broadcasts tiles
implicitly. Forgetting the broadcast is caught:

    shifted = t - row_max
    # error: `-` is not supported between `Tile[float, [8, 16]]` and
    #        `Tile[float, [8, 1]]`

Run it with `python examples/row_softmax.py` (CPU is fine).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from warp.shape_checks import expect_shape

ROWS = 8
COLS = 16


@wp.func
def exp(x: float) -> float:
    # `wp.tile_map(wp.exp, ...)` fails to build in Warp 1.18 (its generated
    # backward pass calls `adj_exp` with the wrong arity); a `wp.func` works.
    return wp.exp(x)


@wp.kernel
def softmax_kernel(x: wp.array2d[float], out: wp.array2d[float]):
    i = wp.tid()
    t = wp.tile_load(x, shape=(ROWS, COLS), offset=(i * ROWS, 0), storage="shared")
    row_max = wp.tile_reshape(wp.tile_reduce(wp.max, t, axis=1), shape=(ROWS, 1))
    e = wp.tile_map(exp, t - wp.tile_broadcast(row_max, shape=(ROWS, COLS)))
    row_sum = wp.tile_reshape(wp.tile_sum(e, axis=1), shape=(ROWS, 1))
    result = e / wp.tile_broadcast(row_sum, shape=(ROWS, COLS))
    if TYPE_CHECKING:
        expect_shape(row_max, (ROWS, 1))
        expect_shape(e, (ROWS, COLS))
        expect_shape(result, (ROWS, COLS))
    wp.tile_store(out, result, offset=(i * ROWS, 0))


def softmax[R: IntVar](x: wp.array[float, [R, 16]]) -> wp.array[float, [R, 16]]:
    """Softmax over each row of `x`; the row count must be a multiple of 8."""
    out = wp.empty_like(x)
    wp.launch_tiled(softmax_kernel, dim=x.shape[0] // ROWS, inputs=[x, out], block_dim=32, device=x.device)
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    x = wp.empty((4 * ROWS, COLS), dtype=float, device="cpu")
    x.assign(rng.standard_normal(x.shape).astype(np.float32))

    y = softmax(x)  # array[float, [32, 16]]

    ref = np.exp(x.numpy() - x.numpy().max(axis=1, keepdims=True))
    ref /= ref.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(y.numpy(), ref, rtol=1e-5, atol=1e-6)
    print("ok: softmax", y.shape, "rows sum to 1:", np.allclose(y.numpy().sum(axis=1), 1.0))


if __name__ == "__main__":
    main()
