# Kernel-scope element access for Fabric arrays. Upstream `__getitem__` only
# describes the host form (`fa[index_array]` -> `indexedfabricarray`) and shadows
# the element access inherited from `Array`, so `fa[i]` was an
# `indexedfabricarray` inside kernels.

from typing import overload

from shape_extensions import IntTuple
from warp._src.types import _Idx

class fabricarray[DType, Shape: IntTuple = IntTuple](noncontiguous_array_base[DType, Shape]):
    @overload
    def __getitem__(self, key: _Idx) -> DType: ...
    @overload
    def __getitem__(self, key: tuple[_Idx, _Idx]) -> DType: ...
    @overload
    def __getitem__(self, key: array) -> indexedfabricarray[DType, Shape]: ...

class indexedfabricarray[DType, Shape: IntTuple = IntTuple](noncontiguous_array_base[DType, Shape]):
    @overload
    def __getitem__(self, key: _Idx) -> DType: ...
    @overload
    def __getitem__(self, key: tuple[_Idx, _Idx]) -> DType: ...
