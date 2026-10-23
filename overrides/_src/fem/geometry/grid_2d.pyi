# Overrides merged into warp/_src/fem/geometry/grid_2d.py by tools/generate_stubs.py.
#
# `Grid2D(bounds_lo=..., bounds_hi=...)` is annotated `wp.vec2`, but converts any
# 2-component value to the grid's scalar type; double-precision grids pass
# `wp.vec2d` bounds (warp/examples/fem).

from typing import Any, Literal

from warp import Vector as _Vector

__generator__ = {"rewrite_params": {"wp.vec2 | None": "_Vector[Any, Literal[2]] | None"}}
