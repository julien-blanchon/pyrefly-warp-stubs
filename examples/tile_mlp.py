"""A two-layer MLP forward pass with Warp tiles, shape-checked by Pyrefly.

Hover any tile in an editor running Pyrefly to see its shape. The
`if TYPE_CHECKING:` block inside the kernel pins shapes with `assert_type`:
Pyrefly checks it, and Warp drops it when building the kernel because the
condition is a compile-time constant. Swapping the operands of the first
`wp.tile_matmul` is reported before the kernel is ever built:

    Argument `Tile[float, [64, 16]]` is not assignable to parameter `b` with
    type `Tile[Any, [32, @_]]` in function `warp.tile_matmul`

Run it with `python examples/tile_mlp.py` (CPU is fine).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, assert_type

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from warp import Tile

DIM_IN = wp.constant(16)
DIM_HID = 64
DIM_OUT = 3
BATCH = 32


@wp.func
def relu(x: float) -> float:
    return wp.max(x, 0.0)


@wp.kernel
def mlp_forward(
    x: wp.array2d[float],
    w0: wp.array2d[float],
    b0: wp.array2d[float],
    w1: wp.array2d[float],
    b1: wp.array2d[float],
    out: wp.array2d[float],
):
    inputs = wp.tile_load(x, shape=(DIM_IN, BATCH))  # [16, 32]
    weights0 = wp.tile_load(w0, shape=(DIM_HID, DIM_IN))  # [64, 16]
    bias0 = wp.tile_load(b0, shape=(DIM_HID, 1))  # [64, 1]
    # Tiles never broadcast implicitly: `+ bias0` alone would be a shape error.
    hidden = wp.tile_map(
        relu, wp.tile_matmul(weights0, inputs) + wp.tile_broadcast(bias0, shape=(DIM_HID, BATCH))
    )  # [64, 32]

    weights1 = wp.tile_load(w1, shape=(DIM_OUT, DIM_HID))  # [3, 64]
    bias1 = wp.tile_load(b1, shape=(DIM_OUT, 1))  # [3, 1]
    output = wp.tile_matmul(weights1, hidden) + wp.tile_broadcast(bias1, shape=(DIM_OUT, BATCH))  # [3, 32]
    if TYPE_CHECKING:
        assert_type(inputs, Tile[float, [16, 32]])
        assert_type(hidden, Tile[Any, [64, 32]])
        assert_type(output, Tile[float, [3, 32]])
    wp.tile_store(out, output)


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((DIM_IN, BATCH)).astype(np.float32)
    w0 = rng.standard_normal((DIM_HID, DIM_IN)).astype(np.float32)
    b0 = rng.standard_normal((DIM_HID, 1)).astype(np.float32)
    w1 = rng.standard_normal((DIM_OUT, DIM_HID)).astype(np.float32)
    b1 = rng.standard_normal((DIM_OUT, 1)).astype(np.float32)

    device = "cpu"
    out = wp.zeros((DIM_OUT, BATCH), dtype=float, device=device)
    arrays = [wp.array(v, dtype=float, device=device) for v in (x, w0, b0, w1, b1)]
    wp.launch_tiled(mlp_forward, dim=1, inputs=[*arrays, out], block_dim=64, device=device)

    expected = w1 @ np.maximum(w0 @ x + b0, 0.0) + b1
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-4, atol=1e-4)
    print("ok: output", out.shape, "matches NumPy")


if __name__ == "__main__":
    main()
