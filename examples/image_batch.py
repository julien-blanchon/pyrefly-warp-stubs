"""Batches of images with symbolic dimensions `B, C, H, W`.

Shows: named dimensions as PEP 695 type parameters (`B: IntVar`), which work
with no runtime dependency (`IntVar` is imported under `TYPE_CHECKING`);
symbolic arithmetic in shapes (`[B, C, H * W]`, inferred from `reshape(-1)`);
views (`transpose`); and a 4-D kernel launch. Mismatched channel counts are
reported at the call:

    mean2 = wp.array([0.5, 0.5], dtype=float)
    normalize(images, mean2, std)
    # error: Argument `array[float, [2]]` is not assignable to parameter `mean`
    #        with type `array[float, [3]]`

The same signatures can be written with jaxtyping strings, e.g.
`Float[wp.array, "B C H W"]` under `@static_jaxtyping("B C H W")`, at the cost
of importing `jaxtyping` and `shape_extensions` at runtime.

Run it with `python examples/image_batch.py` (CPU is fine).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from shape_extensions import IntVar


@wp.kernel
def normalize_kernel(
    images: wp.array4d[float], mean: wp.array1d[float], std: wp.array1d[float], out: wp.array4d[float]
):
    b, c, y, x = wp.tid()
    out[b, c, y, x] = (images[b, c, y, x] - mean[c]) / std[c]


def normalize[B: IntVar, C: IntVar, H: IntVar, W: IntVar](
    images: wp.array[float, [B, C, H, W]], mean: wp.array[float, [C]], std: wp.array[float, [C]]
) -> wp.array[float, [B, C, H, W]]:
    """Per-channel normalisation."""
    out = wp.empty_like(images)
    wp.launch(normalize_kernel, dim=images.shape, inputs=[images, mean, std, out], device=images.device)
    return out


def flatten_pixels[B: IntVar, C: IntVar, H: IntVar, W: IntVar](
    images: wp.array[float, [B, C, H, W]],
) -> wp.array[float, [B, C, H * W]]:
    return images.reshape((images.shape[0], images.shape[1], -1))  # -1 is inferred as H * W


def channels_last[B: IntVar, C: IntVar, H: IntVar, W: IntVar](
    images: wp.array[float, [B, C, H, W]],
) -> wp.array[float, [B, H, W, C]]:
    return images.transpose((0, 2, 3, 1))


def main() -> None:
    rng = np.random.default_rng(0)
    images = wp.empty((2, 3, 8, 8), dtype=float, device="cpu")
    images.assign(rng.random(images.shape, dtype=np.float32))
    mean = wp.array([0.5, 0.4, 0.3], dtype=float, device="cpu")  # array[float, [3]]
    std = wp.array([0.2, 0.2, 0.1], dtype=float, device="cpu")

    normalized = normalize(images, mean, std)  # array[float, [2, 3, 8, 8]]
    pixels = flatten_pixels(normalized)  # array[float, [2, 3, 64]]
    nhwc = channels_last(normalized)  # array[float, [2, 8, 8, 3]]

    ref = (images.numpy() - mean.numpy()[:, None, None]) / std.numpy()[:, None, None]
    np.testing.assert_allclose(normalized.numpy(), ref, rtol=1e-5)
    np.testing.assert_allclose(pixels.numpy(), ref.reshape(2, 3, 64), rtol=1e-5)
    np.testing.assert_allclose(nhwc.contiguous().numpy(), ref.transpose(0, 2, 3, 1), rtol=1e-5)
    print("ok:", normalized.shape, "->", pixels.shape, "and", nhwc.shape)


if __name__ == "__main__":
    main()
