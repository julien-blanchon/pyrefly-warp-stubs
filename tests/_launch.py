"""Run tile kernels on the CPU, and assert that Warp rejects ill-shaped ones."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import warp as wp

DEVICE = "cpu"


def array(values: Any, dtype: Any = float) -> wp.array:
    return wp.array(np.asarray(values, dtype=np.float32), dtype=dtype, device=DEVICE)


def zeros(shape: tuple[int, ...], dtype: Any = float) -> wp.array:
    return wp.zeros(shape, dtype=dtype, device=DEVICE)


def run_tiled(kernel: Any, *arrays: wp.array, block_dim: int = 32) -> None:
    wp.launch_tiled(kernel, dim=1, inputs=list(arrays), block_dim=block_dim, device=DEVICE)


def assert_rejected(kernel: Any, *arrays: wp.array) -> None:
    """The static rejection must match a real one: Warp refuses to build the kernel."""
    # Warp reports build failures as `ValueError` or `RuntimeError` (incl. `WarpCodegenError`).
    with pytest.raises((ValueError, RuntimeError), match="Error while parsing function"):
        run_tiled(kernel, *arrays)


def assert_build_fails(kernel: Any, *arrays: wp.array) -> None:
    """Like `assert_rejected`, for an ordinary (`wp.launch`) kernel."""
    with pytest.raises((ValueError, RuntimeError), match="Error while parsing function"):
        wp.launch(kernel, dim=1, inputs=list(arrays), device=DEVICE)
