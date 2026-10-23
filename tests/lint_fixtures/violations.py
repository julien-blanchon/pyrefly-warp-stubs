# Fixture for skills/add-shape-types-to-warp-project/verify_port.py: every
# line marked `LINT: <LEVEL>` must be reported at that level, and no other line.
# Not imported by the tests (several lines crash at import, which is the point).
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, assert_type, cast

import warp as wp
from shape_extensions import IntVar  # LINT: WARN
from warp import Tile  # LINT: ERROR

if TYPE_CHECKING:
    from warp import Array
    from warp.shape_checks import expect_shape
from warp.shape_checks import expect_shape as runtime_expect  # LINT: ERROR

Points: TypeAlias = wp.array[Any, [int, 3]]  # LINT: ERROR
Points2 = wp.array[Any, [int, 3]]  # LINT: ERROR
type Points3 = wp.array[Any, [int, 3]]
Vec3 = wp.Vector[float, Literal[3]]  # LINT: ERROR
type Vec3Ok = wp.Vector[float, Literal[3]]


def uses_vector(v: wp.Vector[float, Literal[3]]) -> wp.Vector[float, Literal[3]]:
    return cast("wp.Vector[float, Literal[3]]", v)


def builds_vector_type() -> Any:
    return wp.Vector[float, Literal[3]]  # LINT: WARN


def apply(f: Callable[[int], float]) -> None: ...


@wp.struct
class State:
    x: wp.array[float, [8, 3]]  # LINT: ERROR
    y: wp.array2d[float]


@wp.kernel
def kernel(
    a: wp.array[float],  # LINT: INFO
    b: wp.array[float, [4, 8]],  # LINT: ERROR
    c: wp.array2d[float],
):
    t = wp.tile_load(c, shape=(4, 8))
    if TYPE_CHECKING:
        assert_type(t, Tile[float, [4, 8]])
        reveal_type(t)
        expect_shape(t, (4, 8))
    reveal_type(t)  # LINT: ERROR
    expect_shape(t, (4, 8))  # LINT: ERROR
    u: float = 1.0  # LINT: ERROR


@wp.func
def scale(t: wp.tile[float, 4, 8]) -> wp.tile[float, 4, 8]:
    return t


def body(a: wp.array1d[float]):
    assert_type(a, wp.array1d[float])  # LINT: ERROR


wrapped = wp.kernel(body)

x = wp.zeros((10, 3), dtype=float)
assert_type(x, wp.array[Any, [10, 3]])  # LINT: ERROR
assert_type(x, "wp.array[Any, [10, 3]]")
expect_shape(x, (10, 3))  # LINT: ERROR


def host[N: IntVar](p: wp.array[Any, [N, 3]]) -> None:
    cast(wp.array[Any, [N, 3]], p)  # LINT: WARN
    cast("wp.array[Any, [N, 3]]", p)


ignored = 1  # type: ignore  # LINT: WARN
