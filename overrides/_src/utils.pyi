# `ScopedCapture.graph` is `None` until the `with` block exits; upstream leaves it
# untyped (`Unknown | None`), so `wp.capture_launch(capture.graph)` was an error.

from typing import Any, Self, overload

import numpy as np

from warp._src.context import Graph

class ScopedCapture:
    graph: Graph
    def __enter__(self) -> Self: ...

# Without `out` and `axis`, `array_sum` returns a NumPy scalar for scalar arrays
# and a NumPy array for vector/matrix arrays; `array_inner` always a NumPy
# scalar (both checked on Warp 1.18).
@overload
def array_sum(
    values: wp.array[wp.Vector[Any, Any]] | wp.array[wp.Matrix[Any, Any, Any]],
    out: None = None,
    value_count: int | None = None,
    axis: None = None,
) -> np.ndarray[tuple[int, ...], np.dtype[Any]]: ...
@overload
def array_sum(values: wp.array, out: None = None, value_count: int | None = None, axis: None = None) -> np.floating[Any]: ...
@overload
def array_sum[A: wp.array](values: wp.array, out: A, value_count: int | None = None, axis: int | None = None) -> A: ...
@overload
def array_sum(values: wp.array, out: None = None, value_count: int | None = None, *, axis: int) -> wp.array: ...
@overload
def array_inner(a: wp.array, b: wp.array, out: None = None, count: int | None = None, axis: None = None) -> np.floating[Any]: ...
@overload
def array_inner[A: wp.array](a: wp.array, b: wp.array, out: A, count: int | None = None, axis: int | None = None) -> A: ...
@overload
def array_inner(a: wp.array, b: wp.array, out: None = None, count: int | None = None, *, axis: int) -> wp.array: ...
