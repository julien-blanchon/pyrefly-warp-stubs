# `from_torch` returns a `wp.array` unless `return_ctype=True` (upstream:
# `array | array_t` always, so the result could not be passed where a
# `wp.array` is expected). The dtype is kept when given.

from typing import Any, Literal, overload

@overload
def from_torch[D](
    t: torch.Tensor,
    dtype: type[D],
    requires_grad: bool | None = None,
    grad: Any = None,
    return_ctype: Literal[False] = False,
    retain_grad: bool = False,
) -> warp.array[D]: ...
@overload
def from_torch(
    t: torch.Tensor,
    dtype: None = None,
    requires_grad: bool | None = None,
    grad: Any = None,
    return_ctype: Literal[False] = False,
    retain_grad: bool = False,
) -> warp.array[Any]: ...
@overload
def from_torch(
    t: torch.Tensor,
    dtype: type | None = None,
    requires_grad: bool | None = None,
    grad: Any = None,
    *,
    return_ctype: Literal[True],
    retain_grad: bool = False,
) -> warp._src.types.array_t: ...
