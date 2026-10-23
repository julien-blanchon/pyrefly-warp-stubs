# Overrides merged into warp/_src/render/render_usd.py by tools/generate_stubs.py.
#
# Renderer methods annotate positions, rotations, scales and colors as bare
# tuples, but they index into whatever sequence they get, and Warp's own
# examples pass `wp.vec3` / `wp.quat` values. Accept those too.

from typing import Any, Literal

from warp import Quaternion as _Quaternion
from warp import Vector as _Vector

__generator__ = {
    "rewrite_params": {
        "tuple": "tuple | _Vector[Any, Any] | _Quaternion[Any]",
        "tuple | None": "tuple | _Vector[Any, Any] | _Quaternion[Any] | None",
        "tuple[float, float, float]": "tuple[float, float, float] | _Vector[Any, Literal[3]]",
        "tuple[float, float, float] | None": "tuple[float, float, float] | _Vector[Any, Literal[3]] | None",
    }
}
