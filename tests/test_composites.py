"""Vector, matrix, quaternion and transform values in kernels.

`wp.vec3`, `wp.vec3f` and `Vector[float32, Literal[3]]` are one type (upstream
had separate, incompatible classes). Operators are typed as Warp builds them:
the scalar of `v * s` must match the vector's, `mat44 * vec3` is a shape error.
`check_*` functions are only type checked (an `E:` comment marks an expected
Pyrefly error); every rejected kernel below is also rejected by Warp.
"""

from __future__ import annotations

from typing import Any, Literal, assert_type

import numpy as np
import warp as wp

from _launch import DEVICE, assert_build_fails

type Vec3 = wp.Vector[float, Literal[3]]  # PEP 695: lazy, `wp.Vector` is stub-only


def check_one_type(v: wp.array1d[wp.vec3], i: int) -> None:
    assert_type(v[i], wp.vec3f)
    assert_type(v[i], Vec3)
    assert_type(wp.vec3(1.0, 2.0, 3.0), wp.vec3f)
    # Generic builtins return the same type as the named one.
    assert_type(wp.cross(v[i], v[i]), wp.vec3)
    x: wp.vec3 = wp.cross(v[i], v[i])
    assert_type(wp.vec3(np.zeros(3)), wp.vec3)  # any sequence or array


def check_vector_operators(v: wp.array1d[wp.vec3], s: wp.array1d[float], d: wp.array1d[wp.float64], i: int) -> None:
    assert_type(v[i] + v[i], Vec3)
    assert_type(v[i] - v[i], Vec3)
    assert_type(-v[i], Vec3)
    assert_type(v[i] * s[i], Vec3)
    assert_type(2.0 * v[i], Vec3)
    assert_type(v[i] / s[i], Vec3)
    assert_type(s[i] / v[i], Vec3)
    assert_type(v[i].x, float)
    assert_type(v[i][0], float)
    assert_type(wp.dot(v[i], v[i]), float)
    assert_type(wp.length(v[i]), float)
    v[i].x = 1.0
    v[i][0] = 1.0
    v[i] * v[i]  # E: `*` is not supported between `Vector[float, Literal[3]]` and `Vector[float, Literal[3]]`
    v[i] + 1.0  # E: `+` is not supported between `Vector[float, Literal[3]]` and `float`
    v[i] * d[i]  # E: `*` is not supported between `Vector[float, Literal[3]]` and `float64`


def check_matrix_operators(
    m: wp.array1d[wp.mat33], m4: wp.array1d[wp.mat44], v: wp.array1d[wp.vec3], s: wp.array1d[float], i: int
) -> None:
    assert_type(m[i] * v[i], Vec3)
    assert_type(m[i] @ v[i], Vec3)
    assert_type(v[i] * m[i], Vec3)
    assert_type(m[i] @ m[i], wp.Matrix[float, Literal[3], Literal[3]])
    assert_type(m[i] * s[i], wp.mat33)
    assert_type(m[i][0], Vec3)
    assert_type(m[i][0, 1], float)
    m4[i] * v[i]  # E: `*` is not supported between `Matrix[float, Literal[4], Literal[4]]` and `Vector[float, Literal[3]]`


def check_quaternions_and_transforms(q: wp.array1d[wp.quat], t: wp.array1d[wp.transform], v: wp.array1d[wp.vec3], i: int) -> None:
    assert_type(q[i] * q[i], wp.quat)
    assert_type(q[i].w, float)
    assert_type(t[i] * t[i], wp.transform)
    assert_type(t[i].p, Vec3)
    assert_type(t[i].q, wp.quat)
    q[i] * v[i]  # E: `*` is not supported between `Quaternion[float]` and `Vector[float, Literal[3]]`
    -t[i]  # E: Unary `-` is not supported on `Transformation[float]`


def check_host_values(t: wp.transform) -> None:
    assert_type([*t.p, *t.q], list[float])


@wp.kernel(module="unique")
def composite_kernel(
    v: wp.array1d[wp.vec3], m: wp.array1d[wp.mat33], q: wp.array1d[wp.quat], s: wp.array1d[float], out: wp.array1d[wp.vec3]
):
    i = wp.tid()
    w = m[i] * v[i] + 2.0 * v[i] - v[i] / s[i]
    out[i] = wp.quat_rotate(q[i] * q[i], w) + wp.cross(v[i], w)


@wp.kernel(module="unique")
def bad_vector_product_kernel(v: wp.array1d[wp.vec3], out: wp.array1d[wp.vec3]):
    i = wp.tid()
    out[i] = v[i] * v[i]  # E: `*` is not supported


@wp.kernel(module="unique")
def bad_vector_shape_kernel(m: wp.array1d[wp.mat44], v: wp.array1d[wp.vec3], out: wp.array1d[wp.vec3]):
    i = wp.tid()
    out[i] = m[i] * v[i]  # E: `*` is not supported


def test_composites() -> None:
    v = wp.array(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), dtype=wp.vec3, device=DEVICE)
    m = wp.array(np.eye(3, dtype=np.float32)[None], dtype=wp.mat33, device=DEVICE)
    q = wp.array(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.quat, device=DEVICE)
    s = wp.array(np.array([2.0], dtype=np.float32), dtype=float, device=DEVICE)
    out = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
    wp.launch(composite_kernel, dim=1, inputs=[v, m, q, s, out], device=DEVICE)
    expected = np.array([1.0, 2.0, 3.0]) * (1 + 2 - 0.5)  # identity rotation, cross(v, k*v) = 0
    np.testing.assert_allclose(out.numpy()[0], expected, rtol=1e-6)


def test_rejected_composites() -> None:
    v = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
    assert_build_fails(bad_vector_product_kernel, v, wp.zeros(1, dtype=wp.vec3, device=DEVICE))
    assert_build_fails(bad_vector_shape_kernel, wp.zeros(1, dtype=wp.mat44, device=DEVICE), v, wp.zeros(1, dtype=wp.vec3, device=DEVICE))
