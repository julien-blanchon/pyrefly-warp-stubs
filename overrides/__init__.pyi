# Shape-aware overrides merged into warp/__init__.pyi by tools/generate_stubs.py.
#
# Each function defined here replaces every upstream overload of that name; the
# upstream docstring is carried over. Signatures follow the upstream ones, with
# `Tile[Any, tuple[int, ...]]` and `Array[Any]` replaced by shaped types.
#
# `Int` is imported as `_Int`: warp exports its own `Int` type variable. Overloads
# use upstream's `over` alias: `overload` is `wp.overload`, not `typing.overload`.

from collections.abc import Iterator
from typing import Any, Self, TypeAlias

from shape_extensions import Flag, IntTuple, IntTupleOrList, IntVar
from shape_extensions import Int as _Int
from warp._shapes import (
    arange_shape,
    broadcast_to_int_shape,
    broadcast_to_shape,
    remove_axis_shape,
    reshape_int_shape,
    reshape_shape,
    squeeze_shape,
)
from warp._src.types import Array as Array
from warp._src.types import Tile as Tile
from warp._src.types import _HasShape, _TileScalar
from warp._src.types import scalar_base as _scalar_base

_Idx = int32 | _builtins.int
_ArangeArg = _builtins.int | _builtins.float | _scalar_base

# ===========================================================================
# Vector, matrix, quaternion and transform values
#
# Upstream declares `vec3f`, `mat33f`, ... as standalone classes, while builtins
# take and return `Vector[...]`, `Matrix[...]`: two types for one Warp type, so
# `wp.cw_min(wp.cross(a, b), c)` failed once values were typed. Here each named
# type is an alias of its generic form (`vec3f = Vector[float32, Literal[3]]`),
# and the generic classes carry Warp's operators and component access, typed as
# Warp builds them (tests/test_composites.py).
# ===========================================================================

# Extra dimension of a matrix product (`[R, C] @ [C, _K] -> [R, _K]`).
_K = TypeVar("_K", bound=int)

class Vector(Generic[Scalar, Length]):
    # Warp builds values from components, a sequence, an array or another value;
    # upstream's per-class signatures rejected valid calls such as `wp.vec3(np_array)`.
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __iter__(self) -> Iterator[Scalar]: ...
    # Components: `.x` ... `.w` (Warp rejects `.w` on a vec3; not modelled).
    x: Scalar
    y: Scalar
    z: Scalar
    w: Scalar
    def __add__(self, other: Vector[Scalar, Length], /) -> Self: ...
    def __sub__(self, other: Vector[Scalar, Length], /) -> Self: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...
    # The scalar must have the vector's scalar type (`vec3 * 2` is an error).
    @over
    def __mul__(self, other: Scalar, /) -> Self: ...
    @over
    def __mul__(self, other: Matrix[Scalar, Length, _K], /) -> Vector[Scalar, _K]: ...
    def __rmul__(self, other: Scalar, /) -> Self: ...
    def __truediv__(self, other: Scalar, /) -> Self: ...
    def __rtruediv__(self, other: Scalar, /) -> Self: ...
    def __matmul__(self, other: Matrix[Scalar, Length, _K], /) -> Vector[Scalar, _K]: ...
    def __getitem__(self, index: _builtins.int, /) -> Scalar: ...
    def __setitem__(self, index: _builtins.int, value: Scalar, /) -> None: ...
    def __len__(self) -> _builtins.int: ...

class Matrix(Generic[Scalar, Rows, Cols]):
    # Warp builds values from components, a sequence, an array or another value;
    # upstream's per-class signatures rejected valid calls such as `wp.vec3(np_array)`.
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __iter__(self) -> Iterator[Vector[Scalar, Cols]]: ...
    def __add__(self, other: Matrix[Scalar, Rows, Cols], /) -> Self: ...
    def __sub__(self, other: Matrix[Scalar, Rows, Cols], /) -> Self: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...
    @over
    def __mul__(self, other: Scalar, /) -> Self: ...
    @over
    def __mul__(self, other: Vector[Scalar, Cols], /) -> Vector[Scalar, Rows]: ...
    @over
    def __mul__(self, other: Matrix[Scalar, Cols, _K], /) -> Matrix[Scalar, Rows, _K]: ...
    @over
    def __matmul__(self, other: Scalar, /) -> Self: ...
    @over
    def __matmul__(self, other: Vector[Scalar, Cols], /) -> Vector[Scalar, Rows]: ...
    @over
    def __matmul__(self, other: Matrix[Scalar, Cols, _K], /) -> Matrix[Scalar, Rows, _K]: ...
    def __rmul__(self, other: Scalar, /) -> Self: ...
    def __truediv__(self, other: Scalar, /) -> Self: ...
    @over
    def __getitem__(self, index: tuple[_builtins.int, _builtins.int], /) -> Scalar: ...
    @over
    def __getitem__(self, index: _builtins.int, /) -> Vector[Scalar, Cols]: ...
    def __setitem__(self, index: Any, value: Any, /) -> None: ...
    def __len__(self) -> _builtins.int: ...

class Quaternion(Generic[Float]):
    # Warp builds values from components, a sequence, an array or another value;
    # upstream's per-class signatures rejected valid calls such as `wp.vec3(np_array)`.
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __iter__(self) -> Iterator[Float]: ...
    x: Float
    y: Float
    z: Float
    w: Float
    def __add__(self, other: Quaternion[Float], /) -> Self: ...
    def __sub__(self, other: Quaternion[Float], /) -> Self: ...
    def __neg__(self) -> Self: ...
    @over
    def __mul__(self, other: Quaternion[Float], /) -> Self: ...
    @over
    def __mul__(self, other: Float, /) -> Self: ...
    def __rmul__(self, other: Float, /) -> Self: ...
    def __truediv__(self, other: Float, /) -> Self: ...
    def __getitem__(self, index: _builtins.int, /) -> Float: ...
    def __setitem__(self, index: _builtins.int, value: Float, /) -> None: ...
    def __len__(self) -> _builtins.int: ...

class Transformation(Generic[Float]):
    # Warp builds values from components, a sequence, an array or another value;
    # upstream's per-class signatures rejected valid calls such as `wp.vec3(np_array)`.
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __iter__(self) -> Iterator[Float]: ...
    p: Vector[Float, Literal[3]]
    q: Quaternion[Float]
    def __add__(self, other: Transformation[Float], /) -> Self: ...
    def __sub__(self, other: Transformation[Float], /) -> Self: ...
    @over
    def __mul__(self, other: Transformation[Float], /) -> Self: ...
    @over
    def __mul__(self, other: Float, /) -> Self: ...
    def __rmul__(self, other: Float, /) -> Self: ...
    def __getitem__(self, index: _builtins.int, /) -> Float: ...
    def __setitem__(self, index: _builtins.int, value: Float, /) -> None: ...
    def __len__(self) -> _builtins.int: ...

vec2h: TypeAlias = Vector[float16, Literal[2]]
vec2f: TypeAlias = Vector[float32, Literal[2]]
vec2d: TypeAlias = Vector[float64, Literal[2]]
vec2b: TypeAlias = Vector[int8, Literal[2]]
vec2ub: TypeAlias = Vector[uint8, Literal[2]]
vec2s: TypeAlias = Vector[int16, Literal[2]]
vec2us: TypeAlias = Vector[uint16, Literal[2]]
vec2i: TypeAlias = Vector[int32, Literal[2]]
vec2ui: TypeAlias = Vector[uint32, Literal[2]]
vec2l: TypeAlias = Vector[int64, Literal[2]]
vec2ul: TypeAlias = Vector[uint64, Literal[2]]
vec3h: TypeAlias = Vector[float16, Literal[3]]
vec3f: TypeAlias = Vector[float32, Literal[3]]
vec3d: TypeAlias = Vector[float64, Literal[3]]
vec3b: TypeAlias = Vector[int8, Literal[3]]
vec3ub: TypeAlias = Vector[uint8, Literal[3]]
vec3s: TypeAlias = Vector[int16, Literal[3]]
vec3us: TypeAlias = Vector[uint16, Literal[3]]
vec3i: TypeAlias = Vector[int32, Literal[3]]
vec3ui: TypeAlias = Vector[uint32, Literal[3]]
vec3l: TypeAlias = Vector[int64, Literal[3]]
vec3ul: TypeAlias = Vector[uint64, Literal[3]]
vec4h: TypeAlias = Vector[float16, Literal[4]]
vec4f: TypeAlias = Vector[float32, Literal[4]]
vec4d: TypeAlias = Vector[float64, Literal[4]]
vec4b: TypeAlias = Vector[int8, Literal[4]]
vec4ub: TypeAlias = Vector[uint8, Literal[4]]
vec4s: TypeAlias = Vector[int16, Literal[4]]
vec4us: TypeAlias = Vector[uint16, Literal[4]]
vec4i: TypeAlias = Vector[int32, Literal[4]]
vec4ui: TypeAlias = Vector[uint32, Literal[4]]
vec4l: TypeAlias = Vector[int64, Literal[4]]
vec4ul: TypeAlias = Vector[uint64, Literal[4]]
mat22h: TypeAlias = Matrix[float16, Literal[2], Literal[2]]
mat22f: TypeAlias = Matrix[float32, Literal[2], Literal[2]]
mat22d: TypeAlias = Matrix[float64, Literal[2], Literal[2]]
mat33h: TypeAlias = Matrix[float16, Literal[3], Literal[3]]
mat33f: TypeAlias = Matrix[float32, Literal[3], Literal[3]]
mat33d: TypeAlias = Matrix[float64, Literal[3], Literal[3]]
mat44h: TypeAlias = Matrix[float16, Literal[4], Literal[4]]
mat44f: TypeAlias = Matrix[float32, Literal[4], Literal[4]]
mat44d: TypeAlias = Matrix[float64, Literal[4], Literal[4]]
quath: TypeAlias = Quaternion[float16]
quatf: TypeAlias = Quaternion[float32]
quatd: TypeAlias = Quaternion[float64]
transformh: TypeAlias = Transformation[float16]
transformf: TypeAlias = Transformation[float32]
transformd: TypeAlias = Transformation[float64]

# ===========================================================================
# Thread indices
# ===========================================================================

class _ThreadIndex(int):
    """Result of `wp.tid()`: an `int` for 1D launches, unpackable for nD launches.

    Upstream declares `int | tuple[int, int] | ...`, which a type checker cannot
    unpack as `i, j = wp.tid()`; this type supports both spellings.
    """

    def __iter__(self) -> Iterator[int]: ...

def tid() -> _ThreadIndex: ...

# ===========================================================================
# Array creation (Python scope; `zeros`/`empty` are also kernel builtins that
# return fixed-size arrays, which these signatures describe as well)
# ===========================================================================

# Array constructors. `dtype=float` gives `array[float, ...]`, which is
# `float32` here, as in Warp (see `float32` in _src/types.pyi); without a dtype
# the default is `float`. A dtype that is not a class (a `wp.types.vector(...)`
# result, say) gives `Any`. The last overload keeps upstream's signature for
# shapes that are not statically known.
@over
def zeros[N: IntVar, D](
    shape: _Int[N],
    dtype: type[D],
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[D, [N]]: ...
@over
def zeros[S: IntTuple, D](
    shape: IntTupleOrList[S],
    dtype: type[D],
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[D, S]: ...
@over
def zeros[N: IntVar](
    shape: _Int[N],
    *,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[float, [N]]: ...
@over
def zeros[S: IntTuple](
    shape: IntTupleOrList[S],
    *,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[float, S]: ...
@over
def zeros[N: IntVar](
    shape: _Int[N],
    dtype: Any = float,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[Any, [N]]: ...
@over
def zeros[S: IntTuple](
    shape: IntTupleOrList[S],
    dtype: Any = float,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[Any, S]: ...
@over
def zeros(
    shape: _builtins.int | tuple[_builtins.int, ...] | list[_builtins.int] | None = None,
    dtype: Any = float,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[Any, IntTuple]: ...
@over
def ones[N: IntVar, D](
    shape: _Int[N],
    dtype: type[D],
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[D, [N]]: ...
@over
def ones[S: IntTuple, D](
    shape: IntTupleOrList[S],
    dtype: type[D],
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[D, S]: ...
@over
def ones[N: IntVar](
    shape: _Int[N],
    *,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[float, [N]]: ...
@over
def ones[S: IntTuple](
    shape: IntTupleOrList[S],
    *,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[float, S]: ...
@over
def ones[N: IntVar](
    shape: _Int[N],
    dtype: Any = float,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[Any, [N]]: ...
@over
def ones[S: IntTuple](
    shape: IntTupleOrList[S],
    dtype: Any = float,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[Any, S]: ...
@over
def ones(
    shape: _builtins.int | tuple[_builtins.int, ...] | list[_builtins.int] | None = None,
    dtype: Any = float,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[Any, IntTuple]: ...
@over
def empty[N: IntVar, D](
    shape: _Int[N],
    dtype: type[D],
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[D, [N]]: ...
@over
def empty[S: IntTuple, D](
    shape: IntTupleOrList[S],
    dtype: type[D],
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[D, S]: ...
@over
def empty[N: IntVar](
    shape: _Int[N],
    *,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[float, [N]]: ...
@over
def empty[S: IntTuple](
    shape: IntTupleOrList[S],
    *,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[float, S]: ...
@over
def empty[N: IntVar](
    shape: _Int[N],
    dtype: Any = float,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[Any, [N]]: ...
@over
def empty[S: IntTuple](
    shape: IntTupleOrList[S],
    dtype: Any = float,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[Any, S]: ...
@over
def empty(
    shape: _builtins.int | tuple[_builtins.int, ...] | list[_builtins.int] | None = None,
    dtype: Any = float,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[Any, IntTuple]: ...
@over
def full[N: IntVar, D](
    shape: _Int[N],
    value: Any = 0,
    *,
    dtype: type[D],
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[D, [N]]: ...
@over
def full[S: IntTuple, D](
    shape: IntTupleOrList[S],
    value: Any = 0,
    *,
    dtype: type[D],
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[D, S]: ...
@over
def full[N: IntVar, V](
    shape: _Int[N],
    value: V,
    dtype: None = None,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[V, [N]]: ...
@over
def full[S: IntTuple, V](
    shape: IntTupleOrList[S],
    value: V,
    dtype: None = None,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[V, S]: ...
@over
def full(
    shape: _builtins.int | tuple[_builtins.int, ...] | list[_builtins.int] | None = None,
    value: Any = 0,
    dtype: Any = None,
    device: DeviceLike = None,
    requires_grad: _builtins.bool = False,
    pinned: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
    **kwargs: Any,
) -> array[Any, IntTuple]: ...
def zeros_like[D, S: IntTuple](
    src: Array[D, S],
    device: DeviceLike = None,
    requires_grad: _builtins.bool | None = None,
    pinned: _builtins.bool | None = None,
    retain_grad: _builtins.bool = False,
) -> array[D, S]: ...
def ones_like[D, S: IntTuple](
    src: Array[D, S],
    device: DeviceLike = None,
    requires_grad: _builtins.bool | None = None,
    pinned: _builtins.bool | None = None,
    retain_grad: _builtins.bool = False,
) -> array[D, S]: ...
def empty_like[D, S: IntTuple](
    src: Array[D, S],
    device: DeviceLike = None,
    requires_grad: _builtins.bool | None = None,
    pinned: _builtins.bool | None = None,
    retain_grad: _builtins.bool = False,
) -> array[D, S]: ...
def full_like[D, S: IntTuple](
    src: Array[D, S],
    value: Any,
    device: DeviceLike = None,
    requires_grad: _builtins.bool | None = None,
    pinned: _builtins.bool | None = None,
    retain_grad: _builtins.bool = False,
) -> array[D, S]: ...
def clone[D, S: IntTuple](
    src: Array[D, S],
    device: DeviceLike = None,
    requires_grad: _builtins.bool | None = None,
    pinned: _builtins.bool | None = None,
    retain_grad: _builtins.bool = False,
) -> array[D, S]: ...
@over
def from_numpy[S: IntTuple](
    arr: _HasShape[S],
    dtype: Any = None,
    shape: None = None,
    device: DeviceLike | None = None,
    requires_grad: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
) -> array[Any, S]: ...
@over
def from_numpy[S: IntTuple](
    arr: Any,
    dtype: Any = None,
    *,
    shape: IntTupleOrList[S],
    device: DeviceLike | None = None,
    requires_grad: _builtins.bool = False,
    retain_grad: _builtins.bool = False,
) -> array[Any, S]: ...

# ===========================================================================
# Tile construction
# ===========================================================================

@over
def tile_zeros[N: IntVar, D](shape: _Int[N], dtype: type[D], storage: str = "register") -> Tile[D, [N]]: ...
@over
def tile_zeros[S: IntTuple, D](shape: IntTupleOrList[S], dtype: type[D], storage: str = "register") -> Tile[D, S]: ...
@over
def tile_zeros[N: IntVar](shape: _Int[N], dtype: Any = float, storage: str = "register") -> Tile[Any, [N]]: ...
@over
def tile_zeros[S: IntTuple](shape: IntTupleOrList[S], dtype: Any = float, storage: str = "register") -> Tile[Any, S]: ...
@over
def tile_ones[N: IntVar, D](shape: _Int[N], dtype: type[D], storage: str = "register") -> Tile[D, [N]]: ...
@over
def tile_ones[S: IntTuple, D](shape: IntTupleOrList[S], dtype: type[D], storage: str = "register") -> Tile[D, S]: ...
@over
def tile_ones[N: IntVar](shape: _Int[N], dtype: Any = float, storage: str = "register") -> Tile[Any, [N]]: ...
@over
def tile_ones[S: IntTuple](shape: IntTupleOrList[S], dtype: Any = float, storage: str = "register") -> Tile[Any, S]: ...
@over
def tile_empty[N: IntVar, D](shape: _Int[N], dtype: type[D], storage: str = "register") -> Tile[D, [N]]: ...
@over
def tile_empty[S: IntTuple, D](shape: IntTupleOrList[S], dtype: type[D], storage: str = "register") -> Tile[D, S]: ...
@over
def tile_empty[N: IntVar](shape: _Int[N], dtype: Any = float, storage: str = "register") -> Tile[Any, [N]]: ...
@over
def tile_empty[S: IntTuple](shape: IntTupleOrList[S], dtype: Any = float, storage: str = "register") -> Tile[Any, S]: ...
@over
def tile_full[N: IntVar, D](shape: _Int[N], value: Any, dtype: type[D], storage: str = "register") -> Tile[D, [N]]: ...
@over
def tile_full[S: IntTuple, D](
    shape: IntTupleOrList[S], value: Any, dtype: type[D], storage: str = "register"
) -> Tile[D, S]: ...
@over
def tile_full[N: IntVar](shape: _Int[N], value: Any, dtype: Any = ..., storage: str = "register") -> Tile[Any, [N]]: ...
@over
def tile_full[S: IntTuple](
    shape: IntTupleOrList[S], value: Any, dtype: Any = ..., storage: str = "register"
) -> Tile[Any, S]: ...
@over
def tile_randi[N: IntVar](shape: _Int[N], rng: uint32, storage: str = "register") -> Tile[Any, [N]]: ...
@over
def tile_randi[S: IntTuple](shape: IntTupleOrList[S], rng: uint32, storage: str = "register") -> Tile[Any, S]: ...
@over
def tile_randi[N: IntVar](
    shape: _Int[N], rng: uint32, min: _Idx, max: _Idx, storage: str = "register"
) -> Tile[Any, [N]]: ...
@over
def tile_randi[S: IntTuple](
    shape: IntTupleOrList[S], rng: uint32, min: _Idx, max: _Idx, storage: str = "register"
) -> Tile[Any, S]: ...
@over
def tile_randf[N: IntVar](shape: _Int[N], rng: uint32, storage: str = "register") -> Tile[float32, [N]]: ...
@over
def tile_randf[S: IntTuple](shape: IntTupleOrList[S], rng: uint32, storage: str = "register") -> Tile[float32, S]: ...
@over
def tile_randf[N: IntVar](
    shape: _Int[N], rng: uint32, min: float32 | float, max: float32 | float, storage: str = "register"
) -> Tile[float32, [N]]: ...
@over
def tile_randf[S: IntTuple](
    shape: IntTupleOrList[S], rng: uint32, min: float32 | float, max: float32 | float, storage: str = "register"
) -> Tile[float32, S]: ...
@over
def tile_arange[N: IntVar, D: (int, float, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, bfloat16, float32, float64)](stop: _Int[N], /, *, dtype: type[D], storage: str = "register") -> Tile[D, [N]]: ...
@over
def tile_arange[N: IntVar](stop: _Int[N], /, *, storage: str = "register") -> Tile[float32, [N]]: ...
@over
def tile_arange[A: IntVar, B: IntVar, D: (int, float, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, bfloat16, float32, float64)](
    start: _Int[A], stop: _Int[B], /, *, dtype: type[D], storage: str = "register"
) -> Tile[D, [B - A]]: ...
@over
def tile_arange[A: IntVar, B: IntVar](
    start: _Int[A], stop: _Int[B], /, *, storage: str = "register"
) -> Tile[float32, [B - A]]: ...
@over
def tile_arange[A: IntVar, B: IntVar, C: IntVar, D: (int, float, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, bfloat16, float32, float64)](
    start: _Int[A], stop: _Int[B], step: _Int[C], /, *, dtype: type[D], storage: str = "register"
) -> Tile[D, arange_shape(_Int[A], _Int[B], _Int[C])]: ...
@over
def tile_arange[A: IntVar, B: IntVar, C: IntVar](
    start: _Int[A], stop: _Int[B], step: _Int[C], /, *, storage: str = "register"
) -> Tile[float32, arange_shape(_Int[A], _Int[B], _Int[C])]: ...
# Float ranges: the length is not tracked.
@over
def tile_arange[D: (int, float, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, bfloat16, float32, float64)](*args: _ArangeArg, dtype: type[D], storage: str = "register") -> Tile[D, [int]]: ...
@over
def tile_arange(*args: _ArangeArg, storage: str = "register") -> Tile[float32, [int]]: ...

# ===========================================================================
# Global memory <-> tiles. The tile needs one dimension per array dimension.
# ===========================================================================

# One overload per rank: an array of known rank only matches the overload whose
# tile has the same rank, while an array of unknown rank (`wp.array[float]`)
# matches whichever one the `shape` argument selects.
@over
def tile_load[D, A0: IntVar, T0: IntVar](
    a: Array[D, [A0]],
    shape: _Int[T0] | tuple[_Int[T0]],
    offset: _Idx | tuple[_Idx] = ...,
    storage: str = "register",
    bounds_check: bool | _builtins.bool = True,
    aligned: bool | _builtins.bool = False,
) -> Tile[D, [T0]]: ...
@over
def tile_load[D, A0: IntVar, A1: IntVar, T0: IntVar, T1: IntVar](
    a: Array[D, [A0, A1]],
    shape: tuple[_Int[T0], _Int[T1]],
    offset: tuple[_Idx, _Idx] = ...,
    storage: str = "register",
    bounds_check: bool | _builtins.bool = True,
    aligned: bool | _builtins.bool = False,
) -> Tile[D, [T0, T1]]: ...
@over
def tile_load[D, A0: IntVar, A1: IntVar, A2: IntVar, T0: IntVar, T1: IntVar, T2: IntVar](
    a: Array[D, [A0, A1, A2]],
    shape: tuple[_Int[T0], _Int[T1], _Int[T2]],
    offset: tuple[_Idx, _Idx, _Idx] = ...,
    storage: str = "register",
    bounds_check: bool | _builtins.bool = True,
    aligned: bool | _builtins.bool = False,
) -> Tile[D, [T0, T1, T2]]: ...
@over
def tile_load[D, A0: IntVar, A1: IntVar, A2: IntVar, A3: IntVar, T0: IntVar, T1: IntVar, T2: IntVar, T3: IntVar](
    a: Array[D, [A0, A1, A2, A3]],
    shape: tuple[_Int[T0], _Int[T1], _Int[T2], _Int[T3]],
    offset: tuple[_Idx, _Idx, _Idx, _Idx] = ...,
    storage: str = "register",
    bounds_check: bool | _builtins.bool = True,
    aligned: bool | _builtins.bool = False,
) -> Tile[D, [T0, T1, T2, T3]]: ...
@over
def tile_store[A0: IntVar, T0: IntVar](
    a: Array[Any, [A0]],
    t: Tile[Any, [T0]],
    offset: _Idx | tuple[_Idx] = ...,
    bounds_check: bool | _builtins.bool = True,
    aligned: bool | _builtins.bool = False,
) -> None: ...
@over
def tile_store[A0: IntVar, A1: IntVar, T0: IntVar, T1: IntVar](
    a: Array[Any, [A0, A1]],
    t: Tile[Any, [T0, T1]],
    offset: tuple[_Idx, _Idx] = ...,
    bounds_check: bool | _builtins.bool = True,
    aligned: bool | _builtins.bool = False,
) -> None: ...
@over
def tile_store[A0: IntVar, A1: IntVar, A2: IntVar, T0: IntVar, T1: IntVar, T2: IntVar](
    a: Array[Any, [A0, A1, A2]],
    t: Tile[Any, [T0, T1, T2]],
    offset: tuple[_Idx, _Idx, _Idx] = ...,
    bounds_check: bool | _builtins.bool = True,
    aligned: bool | _builtins.bool = False,
) -> None: ...
@over
def tile_store[A0: IntVar, A1: IntVar, A2: IntVar, A3: IntVar, T0: IntVar, T1: IntVar, T2: IntVar, T3: IntVar](
    a: Array[Any, [A0, A1, A2, A3]],
    t: Tile[Any, [T0, T1, T2, T3]],
    offset: tuple[_Idx, _Idx, _Idx, _Idx] = ...,
    bounds_check: bool | _builtins.bool = True,
    aligned: bool | _builtins.bool = False,
) -> None: ...
@over
def tile_atomic_add[D, A0: IntVar, T0: IntVar](
    a: Array[D, [A0]],
    t: Tile[Any, [T0]],
    offset: _Idx | tuple[_Idx] = ...,
    bounds_check: bool | _builtins.bool = True,
) -> Tile[D, [T0]]: ...
@over
def tile_atomic_add[D, A0: IntVar, A1: IntVar, T0: IntVar, T1: IntVar](
    a: Array[D, [A0, A1]],
    t: Tile[Any, [T0, T1]],
    offset: tuple[_Idx, _Idx] = ...,
    bounds_check: bool | _builtins.bool = True,
) -> Tile[D, [T0, T1]]: ...
@over
def tile_atomic_add[D, A0: IntVar, A1: IntVar, A2: IntVar, T0: IntVar, T1: IntVar, T2: IntVar](
    a: Array[D, [A0, A1, A2]],
    t: Tile[Any, [T0, T1, T2]],
    offset: tuple[_Idx, _Idx, _Idx] = ...,
    bounds_check: bool | _builtins.bool = True,
) -> Tile[D, [T0, T1, T2]]: ...
@over
def tile_atomic_add[D, A0: IntVar, A1: IntVar, A2: IntVar, A3: IntVar, T0: IntVar, T1: IntVar, T2: IntVar, T3: IntVar](
    a: Array[D, [A0, A1, A2, A3]],
    t: Tile[Any, [T0, T1, T2, T3]],
    offset: tuple[_Idx, _Idx, _Idx, _Idx] = ...,
    bounds_check: bool | _builtins.bool = True,
) -> Tile[D, [T0, T1, T2, T3]]: ...

# ===========================================================================
# Views and layout
# ===========================================================================

@over
def tile_view[D, S: IntTuple](t: Tile[D, IntTuple], offset: tuple, shape: IntTupleOrList[S]) -> Tile[D, S]: ...
@over
def tile_view[D](t: Tile[D, IntTuple], offset: tuple) -> Tile[D, IntTuple]: ...
def tile_squeeze[D, S: IntTuple, Axis: Flag[_builtins.int | tuple[_builtins.int, ...] | None] = None](
    t: Tile[D, S], axis: Axis = None
) -> Tile[D, squeeze_shape(S, Axis)]: ...
@over
def tile_reshape[D, S: IntTuple, N: IntVar](t: Tile[D, S], shape: _Int[N]) -> Tile[D, reshape_int_shape(S, _Int[N])]: ...
@over
def tile_reshape[D, S: IntTuple, T: IntTuple](
    t: Tile[D, S], shape: IntTupleOrList[T]
) -> Tile[D, reshape_shape(S, T)]: ...
def tile_astype[S: IntTuple, Src: (int, float, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, bfloat16, float32, float64), D: (int, float, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, bfloat16, float32, float64)](t: Tile[Src, S], dtype: type[D]) -> Tile[D, S]: ...
def tile_transpose[D, M: IntVar, N: IntVar](a: Tile[D, [M, N]]) -> Tile[D, [N, M]]: ...
@over
def tile_broadcast[D, S: IntTuple, N: IntVar](a: Tile[D, S], shape: _Int[N]) -> Tile[D, broadcast_to_int_shape(S, _Int[N])]: ...
@over
def tile_broadcast[D, S: IntTuple, T: IntTuple](
    a: Tile[D, S], shape: IntTupleOrList[T]
) -> Tile[D, broadcast_to_shape(S, T)]: ...
@over
def tile_extract[D, N: IntVar](a: Tile[D, [N]], i: _Idx) -> D: ...
@over
def tile_extract[D, N0: IntVar, N1: IntVar](a: Tile[D, [N0, N1]], i: _Idx, j: _Idx) -> D: ...
@over
def tile_extract[D, N0: IntVar, N1: IntVar, N2: IntVar](a: Tile[D, [N0, N1, N2]], i: _Idx, j: _Idx, k: _Idx) -> D: ...
@over
def tile_extract[D, N0: IntVar, N1: IntVar, N2: IntVar, N3: IntVar](
    a: Tile[D, [N0, N1, N2, N3]], i: _Idx, j: _Idx, k: _Idx, l: _Idx
) -> D: ...
# One or two extra indices select a vector or matrix component of the element.
@over
def tile_extract[N: IntVar](a: Tile[Any, [N]], i: _Idx, j: _Idx) -> Any: ...
@over
def tile_extract[N: IntVar](a: Tile[Any, [N]], i: _Idx, j: _Idx, k: _Idx) -> Any: ...
@over
def tile_extract[N0: IntVar, N1: IntVar](a: Tile[Any, [N0, N1]], i: _Idx, j: _Idx, k: _Idx) -> Any: ...
@over
def tile_extract[N0: IntVar, N1: IntVar](a: Tile[Any, [N0, N1]], i: _Idx, j: _Idx, k: _Idx, l: _Idx) -> Any: ...
@over
def tile_extract[N0: IntVar, N1: IntVar, N2: IntVar](
    a: Tile[Any, [N0, N1, N2]], i: _Idx, j: _Idx, k: _Idx, l: _Idx
) -> Any: ...
@over
def tile_extract[N0: IntVar, N1: IntVar, N2: IntVar](
    a: Tile[Any, [N0, N1, N2]], i: _Idx, j: _Idx, k: _Idx, l: _Idx, m: _Idx
) -> Any: ...
@over
def tile_extract[N0: IntVar, N1: IntVar, N2: IntVar, N3: IntVar](
    a: Tile[Any, [N0, N1, N2, N3]], i: _Idx, j: _Idx, k: _Idx, l: _Idx, m: _Idx
) -> Any: ...
@over
def tile_extract[N0: IntVar, N1: IntVar, N2: IntVar, N3: IntVar](
    a: Tile[Any, [N0, N1, N2, N3]], i: _Idx, j: _Idx, k: _Idx, l: _Idx, m: _Idx, n: _Idx
) -> Any: ...

# ===========================================================================
# Reductions, scans and maps
# ===========================================================================

@over
def tile_sum[D](a: Tile[D, IntTuple]) -> Tile[D, [1]]: ...
@over
def tile_sum[D, S: IntTuple, Axis: Flag[int]](a: Tile[D, S], axis: Axis) -> Tile[D, remove_axis_shape(S, Axis)]: ...
def tile_min[D](a: Tile[D, IntTuple]) -> Tile[D, [1]]: ...
def tile_max[D](a: Tile[D, IntTuple]) -> Tile[D, [1]]: ...
def tile_argmin(a: Tile[Any, IntTuple]) -> Tile[int32, [1]]: ...
def tile_argmax(a: Tile[Any, IntTuple]) -> Tile[int32, [1]]: ...
@over
def tile_reduce[D](op: Callable, a: Tile[D, IntTuple]) -> Tile[D, [1]]: ...
@over
def tile_reduce[D, S: IntTuple, Axis: Flag[int]](
    op: Callable, a: Tile[D, S], axis: Axis
) -> Tile[D, remove_axis_shape(S, Axis)]: ...
def tile_dot[D, S: IntTuple](a: Tile[D, S], b: Tile[Any, S]) -> Tile[D, [1]]: ...
def tile_scan_inclusive[D, S: IntTuple](a: Tile[D, S]) -> Tile[D, S]: ...
def tile_scan_exclusive[D, S: IntTuple](a: Tile[D, S]) -> Tile[D, S]: ...
def tile_scan_max_inclusive[D, S: IntTuple](a: Tile[D, S]) -> Tile[D, S]: ...
def tile_scan_min_inclusive[D, S: IntTuple](a: Tile[D, S]) -> Tile[D, S]: ...
@over
def tile_map[S: IntTuple](op: Callable, a: Tile[Any, S], /) -> Tile[Any, S]: ...
@over
def tile_map[S: IntTuple](op: Callable, a: Tile[Any, S], b: Tile[Any, S] | _TileScalar, /) -> Tile[Any, S]: ...
# Three or more operands: extra operands may be non-tile values (scalars, ids,
# structs), so only the first two are checked.
@over
def tile_map[S: IntTuple](
    op: Callable, a: Tile[Any, S], b: Tile[Any, S] | _TileScalar, c: Any, /, *args: Any
) -> Tile[Any, S]: ...
def tile_axpy[S: IntTuple](alpha: Any, src: Tile[Any, S], dest: Tile[Any, S]) -> None: ...
def tile_sort[N: IntVar](keys: Tile[Any, [N]], values: Tile[Any, [N]]) -> None: ...

# ===========================================================================
# Linear algebra. Warp only builds these for floating-point tiles, which upstream
# expresses with the `Float` type variable. Each operand gets its own
# float-constrained parameter: int tiles are rejected, while `float` and
# `float32` operands (the same type to Warp) may still be mixed.
# ===========================================================================

@over
def tile_matmul[D: (float, float16, bfloat16, float32, float64), E: (float, float16, bfloat16, float32, float64), M: IntVar, K: IntVar, N: IntVar](
    a: Tile[D, [M, K]], b: Tile[E, [K, N]], alpha: D | float = 1.0
) -> Tile[D, [M, N]]: ...
@over
def tile_matmul[D: (float, float16, bfloat16, float32, float64), E: (float, float16, bfloat16, float32, float64), O: (float, float16, bfloat16, float32, float64), M: IntVar, K: IntVar, N: IntVar](
    a: Tile[D, [M, K]],
    b: Tile[E, [K, N]],
    out: Tile[O, [M, N]],
    alpha: D | float = 1.0,
    beta: D | float = 1.0,
) -> None: ...
def tile_diag_add[D, N: IntVar](a: Tile[D, [N, N]], d: Tile[Any, [N]]) -> Tile[D, [N, N]]: ...
def tile_cholesky[D: (float, float16, bfloat16, float32, float64), N: IntVar](A: Tile[D, [N, N]], fill_mode: str = "lower") -> Tile[D, [N, N]]: ...
def tile_cholesky_inplace[D: (float, float16, bfloat16, float32, float64), N: IntVar](A: Tile[D, [N, N]], fill_mode: str = "lower") -> None: ...
@over
def tile_cholesky_solve[D: (float, float16, bfloat16, float32, float64), E: (float, float16, bfloat16, float32, float64), N: IntVar](
    L: Tile[E, [N, N]], y: Tile[D, [N]], fill_mode: str = "lower"
) -> Tile[D, [N]]: ...
@over
def tile_cholesky_solve[D: (float, float16, bfloat16, float32, float64), E: (float, float16, bfloat16, float32, float64), N: IntVar, K: IntVar](
    L: Tile[E, [N, N]], y: Tile[D, [N, K]], fill_mode: str = "lower"
) -> Tile[D, [N, K]]: ...
@over
def tile_lower_solve[D: (float, float16, bfloat16, float32, float64), E: (float, float16, bfloat16, float32, float64), N: IntVar](L: Tile[E, [N, N]], y: Tile[D, [N]]) -> Tile[D, [N]]: ...
@over
def tile_lower_solve[D: (float, float16, bfloat16, float32, float64), E: (float, float16, bfloat16, float32, float64), N: IntVar, K: IntVar](L: Tile[E, [N, N]], y: Tile[D, [N, K]]) -> Tile[D, [N, K]]: ...
@over
def tile_upper_solve[D: (float, float16, bfloat16, float32, float64), E: (float, float16, bfloat16, float32, float64), N: IntVar](U: Tile[E, [N, N]], z: Tile[D, [N]]) -> Tile[D, [N]]: ...
@over
def tile_upper_solve[D: (float, float16, bfloat16, float32, float64), E: (float, float16, bfloat16, float32, float64), N: IntVar, K: IntVar](U: Tile[E, [N, N]], z: Tile[D, [N, K]]) -> Tile[D, [N, K]]: ...
@over
def tile_cholesky_solve_inplace[D: (float, float16, bfloat16, float32, float64), E: (float, float16, bfloat16, float32, float64), N: IntVar](L: Tile[E, [N, N]], y: Tile[D, [N]], fill_mode: str = "lower") -> None: ...
@over
def tile_cholesky_solve_inplace[D: (float, float16, bfloat16, float32, float64), E: (float, float16, bfloat16, float32, float64), N: IntVar, K: IntVar](
    L: Tile[E, [N, N]], y: Tile[D, [N, K]], fill_mode: str = "lower"
) -> None: ...
@over
def tile_lower_solve_inplace[D: (float, float16, bfloat16, float32, float64), E: (float, float16, bfloat16, float32, float64), N: IntVar](L: Tile[E, [N, N]], y: Tile[D, [N]]) -> None: ...
@over
def tile_lower_solve_inplace[D: (float, float16, bfloat16, float32, float64), E: (float, float16, bfloat16, float32, float64), N: IntVar, K: IntVar](L: Tile[E, [N, N]], y: Tile[D, [N, K]]) -> None: ...
@over
def tile_upper_solve_inplace[D: (float, float16, bfloat16, float32, float64), E: (float, float16, bfloat16, float32, float64), N: IntVar](U: Tile[E, [N, N]], z: Tile[D, [N]]) -> None: ...
@over
def tile_upper_solve_inplace[D: (float, float16, bfloat16, float32, float64), E: (float, float16, bfloat16, float32, float64), N: IntVar, K: IntVar](U: Tile[E, [N, N]], z: Tile[D, [N, K]]) -> None: ...

# ===========================================================================
# FFT and indexed access. Upstream pins element types (`Tile[int32, ...]`,
# `Tile[Vector[Float, Literal[2]], ...]`), which rejects the equivalent `int`
# and `vec2f`/`vec2d` element types that typed arrays now produce.
# ===========================================================================

def tile_fft(inout: Tile[Any, IntTuple]) -> None: ...
def tile_ifft(inout: Tile[Any, IntTuple]) -> None: ...
def tile_load_indexed[D, S: IntTuple, N: IntVar](
    a: Array[D, IntTuple],
    indices: Tile[Any, [N]],
    shape: IntTupleOrList[S],
    offset: tuple[_Idx, ...] = ...,
    axis: _Idx = 0,
    storage: str = "register",
) -> Tile[D, S]: ...
def tile_store_indexed[N: IntVar](
    a: Array[Any, IntTuple],
    indices: Tile[Any, [N]],
    t: Tile[Any, IntTuple],
    offset: tuple[_Idx, ...] = ...,
    axis: _Idx = 0,
) -> None: ...
def tile_atomic_add_indexed[D, S: IntTuple, N: IntVar](
    a: Array[D, IntTuple],
    indices: Tile[Any, [N]],
    t: Tile[Any, S],
    offset: tuple[_Idx, ...] = ...,
    axis: _Idx = 0,
) -> Tile[D, S]: ...

# ===========================================================================
# Selection: the result has the type of the two values (upstream: `Any`).
# ===========================================================================

@over
def where[T](
    cond: bool | int8 | uint8 | int16 | uint16 | int32 | uint32 | int64 | uint64 | _builtins.bool | int,
    value_if_true: T,
    value_if_false: T,
) -> T: ...
@over
def where[T](arr: Array[Any, IntTuple], value_if_true: T, value_if_false: T) -> T: ...
@over
def select[T](
    cond: bool | int8 | uint8 | int16 | uint16 | int32 | uint32 | int64 | uint64 | _builtins.bool | int,
    value_if_false: T,
    value_if_true: T,
) -> T: ...
@over
def select[T](arr: Array[Any, IntTuple], value_if_false: T, value_if_true: T) -> T: ...

# ===========================================================================
# Builtins whose result type is their `dtype` argument (upstream: `Any`):
# `wp.texture_sample(tex, uvw, dtype=wp.vec4)` is a `vec4`, `wp.cast(x, dtype=int)`
# an `int` (a bit reinterpretation, not a conversion).
# ===========================================================================

@over
def texture_sample[D](tex: Texture1D, u: float32 | float, dtype: type[D], lod: float32 | float = -1.0) -> D: ...
@over
def texture_sample[D](tex: Texture2D, uv: vec2f, dtype: type[D], lod: float32 | float = -1.0) -> D: ...
@over
def texture_sample[D](
    tex: Texture2D, u: float32 | float, v: float32 | float, dtype: type[D], lod: float32 | float = -1.0
) -> D: ...
@over
def texture_sample[D](tex: Texture3D, uvw: vec3f, dtype: type[D], lod: float32 | float = -1.0) -> D: ...
@over
def texture_sample[D](
    tex: Texture3D,
    u: float32 | float,
    v: float32 | float,
    w: float32 | float,
    dtype: type[D],
    lod: float32 | float = -1.0,
) -> D: ...
def volume_sample[D](id: uint64, uvw: vec3f, sampling_mode: int32 | int, dtype: type[D]) -> D: ...
def volume_sample_grad[D](id: uint64, uvw: vec3f, sampling_mode: int32 | int, grad: Any, dtype: type[D]) -> D: ...
def volume_lookup[D](id: uint64, i: int32 | int, j: int32 | int, k: int32 | int, dtype: type[D]) -> D: ...
def cast[D](a: Any, dtype: type[D]) -> D: ...
