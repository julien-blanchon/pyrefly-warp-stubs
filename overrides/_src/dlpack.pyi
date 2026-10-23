# Without a texture `hint`, `from_dlpack` returns an array (upstream: always
# `array | Texture`).

from typing import Any, overload

@overload
def from_dlpack(source: Any, dtype: Any = None, hint: None = None) -> warp.array[Any]: ...
@overload
def from_dlpack(source: Any, dtype: Any = None, hint: Any = None) -> warp.array[Any] | warp.Texture: ...
