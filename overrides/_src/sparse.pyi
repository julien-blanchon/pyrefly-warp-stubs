# Stubbed so that Warp's `@wp.func` overloads by redefinition become
# `typing.overload` groups (see tools/generate_stubs.py); no other changes.

# BsrMatrix is a dynamically built struct: its fields are only listed in the
# docstring and assigned by `bsr_zeros` & co, so they were missing attributes.
# Array parameters with a union dtype (`Array[Vector[Scalar, Cols] | Scalar]`)
# reject every concrete array, since the array dtype is invariant.

__generator__ = {
    "rewrite_params": {
        "Array[Scalar | BlockType[Rows, Cols, Scalar]]": "Array[Any]",
        "Array[Scalar | BlockType[Rows, Cols, Scalar]] | None": "Array[Any] | None",
        "BlockType": "type | Any",
        "BlockType | None": "type | None",
        "BlockType[Rows, Rows, Scalar]": "type",
    }
}

from typing import Any, overload

# `y = A @ x` / `bsr_mv(A, x)` return the concrete `wp.array` (upstream: the
# `Array` base with a union dtype, which has no `.numpy()` and rejects every
# concrete array argument because the dtype is invariant).
def bsr_mv(
    A: BsrMatrixOrExpression[Any],
    x: wp.array[Any],
    y: wp.array[Any] | None = None,
    alpha: Any = 1.0,
    beta: Any = 0.0,
    transpose: bool = False,
    work_buffer: wp.array[Any] | None = None,
    tile_size: int = 0,
) -> wp.array[Any]: ...
def bsr_get_diag(A: BsrMatrixOrExpression[Any], out: wp.array[Any] | None = None) -> wp.array[Any]: ...

class BsrMatrix(Generic[_BlockType]):
    nrow: int
    ncol: int
    nnz: int
    offsets: wp.array[int]
    row_counts: wp.array[int] | None
    columns: wp.array[int]
    values: wp.array[Any]
    @overload
    def __matmul__(self, y: wp.array[Any]) -> wp.array[Any]: ...
    @overload
    def __matmul__(self, y: BsrMatrixOrExpression[Any]) -> BsrMatrix[Any]: ...
    @overload
    def __rmatmul__(self, x: wp.array[Any]) -> wp.array[Any]: ...
    @overload
    def __rmatmul__(self, x: BsrMatrixOrExpression[Any]) -> BsrMatrix[Any]: ...
