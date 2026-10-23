# Overrides merged into warp/_src/fem/geometry/grid_3d.py by tools/generate_stubs.py.
#
# `Grid3D(bounds_lo=..., bounds_hi=...)` is annotated `wp.vec3`, but converts any
# 3-component value to the grid's scalar type; double-precision grids pass
# `wp.vec3d` bounds (warp/examples/fem).

from typing import Any, Literal

from warp import Vector as _Vector

__generator__ = {"rewrite_params": {"wp.vec3 | None": "_Vector[Any, Literal[3]] | None"}}
