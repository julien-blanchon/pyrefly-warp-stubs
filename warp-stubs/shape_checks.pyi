# Type-checking-only helpers (hand-written; this module does not exist at runtime).
#
# Import it under `if TYPE_CHECKING:` and call it under `if TYPE_CHECKING:` too,
# including inside kernels, where Warp drops such blocks when building:
#
#     if TYPE_CHECKING:
#         from warp.shape_checks import expect_shape
#     ...
#     t = wp.tile_load(a, shape=(TILE_M, TILE_N))
#     if TYPE_CHECKING:
#         expect_shape(t, (TILE_M, TILE_N))

from typing import Any

from shape_extensions import IntTuple, IntTupleOrList
from warp._src.types import Array, Tile

def expect_shape[S: IntTuple](value: Tile[Any, S] | Array[Any, S], shape: IntTupleOrList[S], /) -> None:
    """Check that a tile or array has `shape`, given as values (constants, `wp.constant`s, ints).

    Unlike `assert_type(t, Tile[float, [8, 4]])`, which needs literal dimensions,
    the shape can name the module constants the code uses, so the check follows
    them when they change. Dimensions only known at runtime match anything.
    """
