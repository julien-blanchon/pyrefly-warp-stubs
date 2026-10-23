# Shape-aware overrides merged into warp/_src/types.py by tools/generate_stubs.py.
#
# `Array`, `array` and `Tile` get a `Shape: IntTuple` type parameter, which is
# all Pyrefly needs to treat them as shaped arrays. Shape stays the *second*
# parameter so Warp's `wp.array[float]` keeps meaning "dtype float".
#
# `Int` is deliberately not imported from shape_extensions under that name:
# warp/_src/types.py already defines an `Int` type variable.

import builtins
from collections.abc import Iterator, Sequence
from typing import Any, Literal, Protocol, Self, TypeAlias, TypeVar, overload

from shape_extensions import (
    Flag,
    Index,
    IntTuple,
    IntTupleOrList,
    IntVar,
    RegularNestedList,
    index_shape,
)
from shape_extensions import Int as _Int
from warp._shapes import flatten_shape, reshape_int_shape, reshape_shape, reverse_shape, transpose_shape

__generator__ = {
    "replace": ["Tile"],
    # `Volume.allocate_by_voxels(voxel_size=...)` also takes a vector.
    "rewrite_params": {
        "int | float | list[float] | tuple[float, float, float] | None": (
            "int | float | list[float] | tuple[float, float, float] | Vector[Any, Literal[3]] | None"
        ),
        # `wp.indexedarray(a, [idx, None])`: `None` leaves a dimension unindexed.
        "array | list[array] | None": "array | Sequence[array | None] | None",
    },
}

@overload
def constant[N: IntVar](x: _Int[N]) -> _Int[N]: ...
@overload
def constant[T](x: T) -> T: ...

class _HasShape[S: IntTuple](Protocol):
    @property
    def shape(self) -> S: ...

# Index types accepted for one array or tile dimension.
_Idx: TypeAlias = builtins.int | int_base

class Array[DType, Shape: IntTuple = IntTuple]:
    shape: Shape

    # Element access with one integer per dimension (kernel scope).
    @overload
    def __getitem__[N: IntVar](self: Array[DType, [N]], key: _Idx) -> DType: ...
    @overload
    def __getitem__[N0: IntVar, N1: IntVar](
        self: Array[DType, [N0, N1]], key: tuple[_Idx, _Idx]
    ) -> DType: ...
    @overload
    def __getitem__[N0: IntVar, N1: IntVar, N2: IntVar](
        self: Array[DType, [N0, N1, N2]], key: tuple[_Idx, _Idx, _Idx]
    ) -> DType: ...
    @overload
    def __getitem__[N0: IntVar, N1: IntVar, N2: IntVar, N3: IntVar](
        self: Array[DType, [N0, N1, N2, N3]], key: tuple[_Idx, _Idx, _Idx, _Idx]
    ) -> DType: ...
    # Fewer integers than dimensions, or slices: a view (both scopes).
    @overload
    def __getitem__[I: Index](self, key: I) -> array[DType, index_shape(Shape, I)]: ...
    @overload
    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...

class array[DType, Shape: IntTuple = IntTuple](Array[DType, Shape]):
    # Redeclared from `Array`: upstream `array.__getitem__` would shadow it.
    # `wp.array(shape=..., dtype=...)` / `wp.array(data, dtype=...)`: the dtype is
    # kept when it is a class (`float` is `float32` here, as in Warp); otherwise,
    # or when Warp infers it from `data`, it is `Any`.
    @overload
    def __init__[D, S: IntTuple](
        self: array[D, S],
        data: None = None,
        *,
        dtype: type[D],
        shape: IntTupleOrList[S],
        strides: tuple[builtins.int, ...] | None = None,
        ptr: builtins.int | None = None,
        capacity: builtins.int | None = None,
        device: warp.DeviceLike = None,
        pinned: builtins.bool = False,
        copy: builtins.bool = True,
        deleter: Callable[[builtins.int, builtins.int], None] | None = None,
        ndim: builtins.int | None = None,
        grad: array | None = None,
        requires_grad: builtins.bool = False,
        retain_grad: builtins.bool = False,
    ) -> None: ...
    @overload
    def __init__[S: IntTuple](
        self: array[Any, S],
        data: None = None,
        dtype: Any = Any,
        *,
        shape: IntTupleOrList[S],
        strides: tuple[builtins.int, ...] | None = None,
        ptr: builtins.int | None = None,
        capacity: builtins.int | None = None,
        device: warp.DeviceLike = None,
        pinned: builtins.bool = False,
        copy: builtins.bool = True,
        deleter: Callable[[builtins.int, builtins.int], None] | None = None,
        ndim: builtins.int | None = None,
        grad: array | None = None,
        requires_grad: builtins.bool = False,
        retain_grad: builtins.bool = False,
    ) -> None: ...
    @overload
    def __init__[D, N: IntVar](
        self: array[D, [N]],
        data: None = None,
        *,
        dtype: type[D],
        shape: _Int[N],
        strides: tuple[builtins.int, ...] | None = None,
        ptr: builtins.int | None = None,
        capacity: builtins.int | None = None,
        device: warp.DeviceLike = None,
        pinned: builtins.bool = False,
        copy: builtins.bool = True,
        deleter: Callable[[builtins.int, builtins.int], None] | None = None,
        ndim: builtins.int | None = None,
        grad: array | None = None,
        requires_grad: builtins.bool = False,
        retain_grad: builtins.bool = False,
    ) -> None: ...
    @overload
    def __init__[N: IntVar](
        self: array[Any, [N]],
        data: None = None,
        dtype: Any = Any,
        *,
        shape: _Int[N],
        strides: tuple[builtins.int, ...] | None = None,
        ptr: builtins.int | None = None,
        capacity: builtins.int | None = None,
        device: warp.DeviceLike = None,
        pinned: builtins.bool = False,
        copy: builtins.bool = True,
        deleter: Callable[[builtins.int, builtins.int], None] | None = None,
        ndim: builtins.int | None = None,
        grad: array | None = None,
        requires_grad: builtins.bool = False,
        retain_grad: builtins.bool = False,
    ) -> None: ...
    @overload
    def __init__[D, S: IntTuple](
        self: array[D, S],
        data: RegularNestedList[S, Any],
        dtype: type[D],
        shape: None = None,
        strides: tuple[builtins.int, ...] | None = None,
        ptr: None = None,
        capacity: builtins.int | None = None,
        device: warp.DeviceLike = None,
        pinned: builtins.bool = False,
        copy: builtins.bool = True,
        deleter: Callable[[builtins.int, builtins.int], None] | None = None,
        ndim: builtins.int | None = None,
        grad: array | None = None,
        requires_grad: builtins.bool = False,
        retain_grad: builtins.bool = False,
    ) -> None: ...
    @overload
    def __init__[S: IntTuple](
        self: array[Any, S],
        data: RegularNestedList[S, Any],
        dtype: Any = Any,
        shape: None = None,
        strides: tuple[builtins.int, ...] | None = None,
        ptr: None = None,
        capacity: builtins.int | None = None,
        device: warp.DeviceLike = None,
        pinned: builtins.bool = False,
        copy: builtins.bool = True,
        deleter: Callable[[builtins.int, builtins.int], None] | None = None,
        ndim: builtins.int | None = None,
        grad: array | None = None,
        requires_grad: builtins.bool = False,
        retain_grad: builtins.bool = False,
    ) -> None: ...
    @overload
    def __init__[D, S: IntTuple](
        self: array[D, S],
        data: _HasShape[S],
        dtype: type[D],
        shape: None = None,
        strides: tuple[builtins.int, ...] | None = None,
        ptr: None = None,
        capacity: builtins.int | None = None,
        device: warp.DeviceLike = None,
        pinned: builtins.bool = False,
        copy: builtins.bool = True,
        deleter: Callable[[builtins.int, builtins.int], None] | None = None,
        ndim: builtins.int | None = None,
        grad: array | None = None,
        requires_grad: builtins.bool = False,
        retain_grad: builtins.bool = False,
    ) -> None: ...
    # Anything else (pointers, untyped data, annotation-only construction).
    @overload
    def __init__[S: IntTuple](
        self: array[Any, S],
        data: _HasShape[S],
        dtype: Any = Any,
        shape: None = None,
        strides: tuple[builtins.int, ...] | None = None,
        ptr: None = None,
        capacity: builtins.int | None = None,
        device: warp.DeviceLike = None,
        pinned: builtins.bool = False,
        copy: builtins.bool = True,
        deleter: Callable[[builtins.int, builtins.int], None] | None = None,
        ndim: builtins.int | None = None,
        grad: array | None = None,
        requires_grad: builtins.bool = False,
        retain_grad: builtins.bool = False,
    ) -> None: ...
    # Anything else (pointers, untyped data, annotation-only construction).
    @overload
    def __init__(
        self: array[Any, IntTuple],
        data: list | tuple | np.ndarray | None = None,
        dtype: Any = Any,
        shape: builtins.int | tuple[builtins.int, ...] | list[builtins.int] | None = None,
        strides: tuple[builtins.int, ...] | None = None,
        ptr: builtins.int | None = None,
        capacity: builtins.int | None = None,
        device: warp.DeviceLike = None,
        pinned: builtins.bool = False,
        copy: builtins.bool = True,
        deleter: Callable[[builtins.int, builtins.int], None] | None = None,
        ndim: builtins.int | None = None,
        grad: array | None = None,
        requires_grad: builtins.bool = False,
        retain_grad: builtins.bool = False,
    ) -> None: ...
    # Element access with one integer per dimension (kernel scope).
    @overload
    def __getitem__[N: IntVar](self: array[DType, [N]], key: _Idx) -> DType: ...
    @overload
    def __getitem__[N0: IntVar, N1: IntVar](
        self: array[DType, [N0, N1]], key: tuple[_Idx, _Idx]
    ) -> DType: ...
    @overload
    def __getitem__[N0: IntVar, N1: IntVar, N2: IntVar](
        self: array[DType, [N0, N1, N2]], key: tuple[_Idx, _Idx, _Idx]
    ) -> DType: ...
    @overload
    def __getitem__[N0: IntVar, N1: IntVar, N2: IntVar, N3: IntVar](
        self: array[DType, [N0, N1, N2, N3]], key: tuple[_Idx, _Idx, _Idx, _Idx]
    ) -> DType: ...
    # Fewer integers than dimensions, or slices: a view (both scopes).
    @overload
    def __getitem__[I: Index](self, key: I) -> array[DType, index_shape(Shape, I)]: ...
    @overload
    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    @property
    def grad(self) -> array[DType, Shape]: ...
    @grad.setter
    def grad(self, grad: array[DType, Shape] | None) -> None: ...
    def to(self, device: warp.DeviceLike, requires_grad: builtins.bool | None = None) -> array[DType, Shape]: ...
    def contiguous(self) -> array[DType, Shape]: ...
    def numpy(self, *, _suppress_bfloat16_warning: builtins.bool = False) -> np.ndarray[tuple[builtins.int, ...], np.dtype[Any]]: ...
    def list(self) -> builtins.list[Any]: ...
    def flatten(self) -> array[DType, flatten_shape(Shape)]: ...
    @overload
    def reshape[N: IntVar](self, shape: _Int[N]) -> array[DType, reshape_int_shape(Shape, _Int[N])]: ...
    @overload
    def reshape[Target: IntTuple](
        self, shape: IntTupleOrList[Target]
    ) -> array[DType, reshape_shape(Shape, Target)]: ...
    @overload
    def transpose(self, axes: None = None) -> array[DType, reverse_shape(Shape)]: ...
    @overload
    def transpose[Axes: IntTuple](
        self, axes: IntTupleOrList[Axes]
    ) -> array[DType, transpose_shape(Shape, Axes)]: ...
    # Reinterpreting as a dtype of another size changes the shape; keep it gradual.
    def view[D](self, dtype: type[D]) -> array[D, IntTuple]: ...

# `wp.array2d[float]` is a rank-2 array of float with unknown extents. Shape
# lists must spell dimensions as bare `int`: Pyrefly reads `builtins.int` there
# as an unknown shape.
_ArrayDType = TypeVar("_ArrayDType", default=Any)

array1d: TypeAlias = array[_ArrayDType, [int]]
array2d: TypeAlias = array[_ArrayDType, [int, int]]
array3d: TypeAlias = array[_ArrayDType, [int, int, int]]
array4d: TypeAlias = array[_ArrayDType, [int, int, int, int]]

class noncontiguous_array_base[DType, Shape: IntTuple = IntTuple](Array[DType, Shape]): ...
class indexedarray[DType, Shape: IntTuple = IntTuple](noncontiguous_array_base[DType, Shape]): ...

# (`wp.indexedarray1d`–`4d` stay functions, as in Warp: `wp.indexedarray2d[float]`
# is not a valid annotation, and Warp raises when it meets one in a kernel.)

# ---------------------------------------------------------------------------
# Vector, matrix, quaternion and transform types. Upstream defines these twice:
# runtime-built classes here (`class vec3f(vector(length=3, ...))`, opaque to a
# checker) and typed declarations in warp/__init__.pyi. Re-exporting the latter
# makes `warp._src.types.vec3f`, `wp.vec3f` and `wp.vec3` one type everywhere.
# ---------------------------------------------------------------------------

from warp import Vector as Vector
from warp import Matrix as Matrix
from warp import Quaternion as Quaternion
from warp import Transformation as Transformation
from warp import quath as quath
from warp import quatf as quatf
from warp import quatd as quatd
from warp import transformh as transformh
from warp import transformf as transformf
from warp import transformd as transformd
from warp import vec2h as vec2h
from warp import vec3h as vec3h
from warp import vec4h as vec4h
from warp import vec2f as vec2f
from warp import vec3f as vec3f
from warp import vec4f as vec4f
from warp import vec2d as vec2d
from warp import vec3d as vec3d
from warp import vec4d as vec4d
from warp import vec2b as vec2b
from warp import vec3b as vec3b
from warp import vec4b as vec4b
from warp import vec2ub as vec2ub
from warp import vec3ub as vec3ub
from warp import vec4ub as vec4ub
from warp import vec2s as vec2s
from warp import vec3s as vec3s
from warp import vec4s as vec4s
from warp import vec2us as vec2us
from warp import vec3us as vec3us
from warp import vec4us as vec4us
from warp import vec2i as vec2i
from warp import vec3i as vec3i
from warp import vec4i as vec4i
from warp import vec2ui as vec2ui
from warp import vec3ui as vec3ui
from warp import vec4ui as vec4ui
from warp import vec2l as vec2l
from warp import vec3l as vec3l
from warp import vec4l as vec4l
from warp import vec2ul as vec2ul
from warp import vec3ul as vec3ul
from warp import vec4ul as vec4ul
from warp import mat22h as mat22h
from warp import mat33h as mat33h
from warp import mat44h as mat44h
from warp import mat22f as mat22f
from warp import mat33f as mat33f
from warp import mat44f as mat44f
from warp import mat22d as mat22d
from warp import mat33d as mat33d
from warp import mat44d as mat44d

# Type factories (`wp.types.vector(length=5, dtype=float)`). Upstream they are
# untyped, and a checker sees the internal class each one builds (`vec_t`),
# which is not a `Vector`. Also gives the classes built from them, such as
# `wp.spatial_vector` (`vector(length=6, ...)`), their `Vector` base.
@overload
def vector[S, L: builtins.int](length: L, dtype: type[S]) -> type[Vector[S, L]]: ...
@overload
def vector(length: builtins.int, dtype: Any) -> type[Vector[Any, Any]]: ...
@overload
def matrix[S, R: builtins.int, C: builtins.int](shape: tuple[R, C], dtype: type[S]) -> type[Matrix[S, R, C]]: ...
@overload
def matrix(shape: tuple[builtins.int, ...], dtype: Any) -> type[Matrix[Any, Any, Any]]: ...
@overload
def quaternion[F](dtype: type[F]) -> type[Quaternion[F]]: ...
@overload
def quaternion(dtype: Any = Any) -> type[Quaternion[Any]]: ...
@overload
def transformation[F](dtype: type[F]) -> type[Transformation[F]]: ...
@overload
def transformation(dtype: Any = Any) -> type[Transformation[Any]]: ...

# ---------------------------------------------------------------------------
# Scalars. Upstream operators are untyped, and Pyrefly infers
# `NotImplementedType | ...` from their bodies, so `wp.sin(a[i] * 2.0)` on a
# `wp.array[wp.float32]` failed once array elements became typed.
# ---------------------------------------------------------------------------

_PyNumber: TypeAlias = builtins.int | builtins.float

# Warp canonicalises Python `float`, `int` and `bool` to `float32`, `int32` and
# `wp.bool`: they are the same type in kernels and arrays. Making them aliases
# here gives the checker the same view (`max(f32, 0.0)` is fine, `max(f64, 0.0)`
# is an error, as in Warp).
float32: TypeAlias = builtins.float
int32: TypeAlias = builtins.int
bool: TypeAlias = builtins.bool

class scalar_base:
    def __add__(self, y: Self, /) -> Self: ...
    def __radd__(self, y: Self, /) -> Self: ...
    def __sub__(self, y: Self, /) -> Self: ...
    def __rsub__(self, y: Self, /) -> Self: ...
    def __mul__(self, y: Self, /) -> Self: ...
    def __rmul__(self, x: Self, /) -> Self: ...
    def __truediv__(self, y: Self, /) -> Self: ...
    def __rtruediv__(self, x: Self, /) -> Self: ...
    def __mod__(self, x: Self, /) -> Self: ...
    def __rmod__(self, x: Self, /) -> Self: ...
    def __pos__(self) -> Self: ...
    def __neg__(self) -> Self: ...

class float_base(scalar_base):
    def __ge__(self, x: Self, /) -> builtins.bool: ...
    def __gt__(self, x: Self, /) -> builtins.bool: ...
    def __le__(self, x: Self, /) -> builtins.bool: ...
    def __lt__(self, x: Self, /) -> builtins.bool: ...
    def __pow__(self, x: Self, /) -> Self: ...
    def __rpow__(self, x: Self, /) -> Self: ...
    def __abs__(self) -> Self: ...

class int_base(scalar_base):
    def __ge__(self, x: Self, /) -> builtins.bool: ...
    def __gt__(self, x: Self, /) -> builtins.bool: ...
    def __le__(self, x: Self, /) -> builtins.bool: ...
    def __lt__(self, x: Self, /) -> builtins.bool: ...
    def __abs__(self) -> Self: ...

# ---------------------------------------------------------------------------
# Tiles (kernel scope). Element-wise operators require identical shapes; Warp
# does not broadcast tiles implicitly (use `wp.tile_broadcast`).
# ---------------------------------------------------------------------------

_TileScalar: TypeAlias = builtins.int | builtins.float | scalar_base | Vector[Any, Any] | Matrix[Any, Any, Any]

class Tile[DType, Shape: IntTuple = IntTuple]:
    shape: Shape
    dtype: Any

    @overload
    def __getitem__[N: IntVar](self: Tile[DType, [N]], key: _Idx) -> DType: ...
    @overload
    def __getitem__[N0: IntVar, N1: IntVar](self: Tile[DType, [N0, N1]], key: tuple[_Idx, _Idx]) -> DType: ...
    @overload
    def __getitem__[N0: IntVar, N1: IntVar, N2: IntVar](
        self: Tile[DType, [N0, N1, N2]], key: tuple[_Idx, _Idx, _Idx]
    ) -> DType: ...
    @overload
    def __getitem__[N0: IntVar, N1: IntVar, N2: IntVar, N3: IntVar](
        self: Tile[DType, [N0, N1, N2, N3]], key: tuple[_Idx, _Idx, _Idx, _Idx]
    ) -> DType: ...
    @overload
    def __getitem__[I: Index](self, key: I) -> Tile[DType, index_shape(Shape, I)]: ...
    @overload
    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...

    # Operators follow what Warp builds (checked by tests/test_tiles.py):
    # `+`/`-` only between tiles, `*`/`/`/`@` with tiles or scalars, `&|^` between
    # tiles, unary `-` but not unary `+`. Shapes must be identical.
    def __add__(self, other: Tile[Any, Shape], /) -> Tile[DType, Shape]: ...
    def __sub__(self, other: Tile[Any, Shape], /) -> Tile[DType, Shape]: ...
    @overload
    def __mul__(self, other: Tile[Any, Shape], /) -> Tile[DType, Shape]: ...
    @overload
    def __mul__(self, other: _TileScalar, /) -> Tile[DType, Shape]: ...
    def __rmul__(self, other: _TileScalar, /) -> Tile[DType, Shape]: ...
    @overload
    def __truediv__(self, other: Tile[Any, Shape], /) -> Tile[DType, Shape]: ...
    @overload
    def __truediv__(self, other: _TileScalar, /) -> Tile[DType, Shape]: ...
    def __rtruediv__(self, other: _TileScalar, /) -> Tile[DType, Shape]: ...
    # `@` is element-wise multiplication for tiles (use `wp.tile_matmul` for matmul).
    @overload
    def __matmul__(self, other: Tile[Any, Shape], /) -> Tile[DType, Shape]: ...
    @overload
    def __matmul__(self, other: _TileScalar, /) -> Tile[DType, Shape]: ...
    def __rmatmul__(self, other: _TileScalar, /) -> Tile[DType, Shape]: ...
    def __and__(self, other: Tile[Any, Shape], /) -> Tile[DType, Shape]: ...
    def __or__(self, other: Tile[Any, Shape], /) -> Tile[DType, Shape]: ...
    def __xor__(self, other: Tile[Any, Shape], /) -> Tile[DType, Shape]: ...
    def __neg__(self) -> Tile[DType, Shape]: ...
    def __len__(self) -> builtins.int: ...

# Generic in the coordinate precision, so `wp.HashGridQuery[wp.float64]`
# annotations (upstream: a dynamic `__class_getitem__`, read as `Unknown`) are
# a type. Bare `wp.HashGridQuery` is the float32 query, as in Warp.
class HashGridQuery[D = float32]:
    # Kernel scope: `for index in wp.hash_grid_query(grid, point, radius)`.
    def __iter__(self) -> Iterator[int]: ...

# Kernels receive these objects by their `id`. Upstream assigns it from untyped
# native calls; it is typed `uint64`, the type every kernel parameter, builtin
# and `@wp.struct` field uses for it (a Python int at runtime, which Warp accepts
# wherever a `uint64` is expected).
class Mesh:
    id: uint64
    # Untyped properties upstream; Warp's registry declares `array(dtype=vec3)`.
    @property
    def points(self) -> array[vec3]: ...
    @points.setter
    def points(self, points_new: array[vec3]) -> None: ...
    @property
    def velocities(self) -> array[vec3]: ...
    @velocities.setter
    def velocities(self, velocities_new: array[vec3]) -> None: ...

class Bvh:
    id: uint64

class HashGrid:
    id: uint64

# Inside `class Volume`, the method `Volume.array` shadows the `array` class, so
# every later `-> array` annotation (get_voxels, get_tiles, feature_array,
# rebuild, allocate_by_*) was the *method*: results were `Unknown`.
_VolumeArray = array

class Volume:
    id: uint64
    def array(self) -> _VolumeArray[uint8, [int]]: ...
    def get_tiles(self, out: _VolumeArray | None = None) -> _VolumeArray[int32, [int, 3]]: ...
    def get_voxels(self, out: _VolumeArray | None = None) -> _VolumeArray[int32, [int, 3]]: ...
    def feature_array(self, feature_index: builtins.int, dtype: Any = None) -> _VolumeArray[Any, [int]]: ...
    # `min`/`max` take any 3-sequence (tuple, vec3, NumPy array, float list);
    # upstream: `list[int]`.
    @classmethod
    def allocate(
        cls,
        min: Sequence[builtins.float] | Vector[Any, Literal[3]] | np.ndarray,
        max: Sequence[builtins.float] | Vector[Any, Literal[3]] | np.ndarray,
        voxel_size: builtins.int | builtins.float | list[builtins.float] | tuple[builtins.float, builtins.float, builtins.float],
        bg_value: Any = 0.0,
        translation: Any = (0.0, 0.0, 0.0),
        points_in_world_space: builtins.bool = False,
        device: warp.DeviceLike = None,
    ) -> Volume: ...
    def rebuild(
        self, points: _VolumeArray, status: _VolumeArray | None = None, point_mask: _VolumeArray | None = None
    ) -> _VolumeArray | None: ...

class MeshQuery:
    # Kernel scope: `for face in wp.mesh_query_aabb(mesh, low, high)`. (BVH
    # queries are not iterable in Warp: loop with `wp.bvh_query_next`.)
    def __iter__(self) -> Iterator[int]: ...
