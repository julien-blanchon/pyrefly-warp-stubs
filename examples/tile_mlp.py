"""A two-layer MLP forward pass with Warp tiles, shape-checked by Pyrefly.

Shows: tile shapes inferred inside a kernel, in-kernel checks under
`if TYPE_CHECKING:` (Warp drops that block when building the kernel), and a
typed host function that carries the input and output shapes. Kernel
signatures stay plain Warp (`wp.array2d[float]`); shapes live on the host
function, where Pyrefly checks every call:

    mlp_forward(x, w1, b0, w0, b1)   # w0 and w1 swapped
    # error: Argument `array[float, [3, 64]]` is not assignable to parameter
    #        `w0` with type `array[float, [64, 16]]`

No Pyrefly package is needed to run this file: shape types appear only in
annotations (deferred by the `__future__` import) and `TYPE_CHECKING` blocks.

Run it with `python examples/tile_mlp.py` (CPU is fine).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from warp.shape_checks import expect_shape

DIM_IN = wp.constant(16)
DIM_HID = 64
DIM_OUT = 3
TILE_BATCH = 32  # samples handled by one block


@wp.func
def relu(x: float) -> float:
    return wp.max(x, 0.0)


@wp.kernel
def mlp_kernel(
    x: wp.array2d[float],
    w0: wp.array2d[float],
    b0: wp.array2d[float],
    w1: wp.array2d[float],
    b1: wp.array2d[float],
    out: wp.array2d[float],
):
    i = wp.tid()
    inputs = wp.tile_load(x, shape=(DIM_IN, TILE_BATCH), offset=(0, i * TILE_BATCH))
    weights0 = wp.tile_load(w0, shape=(DIM_HID, DIM_IN))
    bias0 = wp.tile_load(b0, shape=(DIM_HID, 1))
    # Tiles never broadcast implicitly: `+ bias0` alone would be a shape error.
    hidden = wp.tile_map(
        relu, wp.tile_matmul(weights0, inputs) + wp.tile_broadcast(bias0, shape=(DIM_HID, TILE_BATCH))
    )

    weights1 = wp.tile_load(w1, shape=(DIM_OUT, DIM_HID))
    bias1 = wp.tile_load(b1, shape=(DIM_OUT, 1))
    output = wp.tile_matmul(weights1, hidden) + wp.tile_broadcast(bias1, shape=(DIM_OUT, TILE_BATCH))
    if TYPE_CHECKING:
        expect_shape(inputs, (DIM_IN, TILE_BATCH))
        expect_shape(hidden, (DIM_HID, TILE_BATCH))
        expect_shape(output, (DIM_OUT, TILE_BATCH))
    wp.tile_store(out, output, offset=(0, i * TILE_BATCH))


# Annotations need literal sizes (Pyrefly cannot read module constants there);
# the batch `B` is generic and flows to the result.
def mlp_forward[B: IntVar](
    x: wp.array[float, [16, B]],
    w0: wp.array[float, [64, 16]],
    b0: wp.array[float, [64, 1]],
    w1: wp.array[float, [3, 64]],
    b1: wp.array[float, [3, 1]],
) -> wp.array[float, [3, B]]:
    """Apply the MLP to `B` samples (columns of `x`); `B` must be a multiple of 32."""
    out = wp.zeros((DIM_OUT, x.shape[1]), dtype=float, device=x.device)
    wp.launch_tiled(
        mlp_kernel, dim=x.shape[1] // TILE_BATCH, inputs=[x, w0, b0, w1, b1, out], block_dim=64, device=x.device
    )
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    device = "cpu"
    batch = 4 * TILE_BATCH

    # Allocating with shapes, then filling, keeps the shapes known to Pyrefly.
    x = wp.empty((DIM_IN, batch), dtype=float, device=device)
    w0 = wp.empty((DIM_HID, DIM_IN), dtype=float, device=device)
    b0 = wp.empty((DIM_HID, 1), dtype=float, device=device)
    w1 = wp.empty((DIM_OUT, DIM_HID), dtype=float, device=device)
    b1 = wp.empty((DIM_OUT, 1), dtype=float, device=device)
    for a in (x, w0, b0, w1, b1):
        a.assign(rng.standard_normal(a.shape).astype(np.float32))

    y = mlp_forward(x, w0, b0, w1, b1)  # array[float, [3, 128]]

    expected = w1.numpy() @ np.maximum(w0.numpy() @ x.numpy() + b0.numpy(), 0.0) + b1.numpy()
    np.testing.assert_allclose(y.numpy(), expected, rtol=1e-4, atol=1e-4)
    print("ok: output", y.shape, "matches NumPy")


if __name__ == "__main__":
    main()
