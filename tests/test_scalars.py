"""Scalar types in kernels: Python `float`/`int` *are* `wp.float32`/`wp.int32`.

Warp canonicalises them and performs no implicit conversion between scalar
types: `wp.max(f32, 0.0)` builds, `wp.max(f64, 0.0)` and `f32 + f64` do not.
`check_*` functions are only type checked (an `E:` comment marks an expected
Pyrefly error); every rejected kernel below is also rejected by Warp.
"""

from __future__ import annotations

from typing import assert_type

import numpy as np
import warp as wp

from _launch import DEVICE, assert_build_fails


def check_literals_are_float32(a: wp.array1d[wp.float32], b: wp.array1d[float], i: int) -> None:
    assert_type(a[i], float)  # `wp.float32` is `float`
    assert_type(wp.max(a[i], 0.0), float)
    assert_type(wp.max(0.0, a[i]), float)
    assert_type(wp.max(a[i], b[i]), float)
    assert_type(wp.clamp(a[i], 0.0, 1.0), float)
    assert_type(a[i] * 2.0, float)


def check_no_implicit_conversion(a: wp.array1d[wp.float32], d: wp.array1d[wp.float64], i: int) -> None:
    assert_type(d[i], wp.float64)
    assert_type(d[i] * d[i], wp.float64)
    assert_type(wp.sin(d[i]), wp.float64)
    wp.max(d[i], 0.0)  # E: No matching overload found for function `warp.max`
    wp.max(a[i], d[i])  # E: No matching overload found for function `warp.max`
    a[i] + d[i]  # E: `+` is not supported between `float` and `float64`
    d[i] + 0.0  # E: `+` is not supported between `float64` and `float`


@wp.kernel(module="unique")
def mixed_ok_kernel(a: wp.array1d[wp.float32], b: wp.array1d[float], out: wp.array1d[float]):
    i = wp.tid()
    out[i] = wp.clamp(wp.max(a[i], 0.0) + b[i], 0.0, 1.0) * 2.0


@wp.kernel(module="unique")
def max_f64_literal_kernel(d: wp.array1d[wp.float64], out: wp.array1d[wp.float64]):
    i = wp.tid()
    out[i] = wp.max(d[i], 0.0)  # E: No matching overload found for function `warp.max`


@wp.kernel(module="unique")
def add_f32_f64_kernel(a: wp.array1d[wp.float32], d: wp.array1d[wp.float64], out: wp.array1d[float]):
    i = wp.tid()
    out[i] = a[i] + d[i]  # E: `+` is not supported between `float` and `float64`


def test_float32_and_float_mix() -> None:
    a = wp.array(np.array([-1.0, 0.25], dtype=np.float32), dtype=wp.float32, device=DEVICE)
    b = wp.array(np.array([0.5, 0.5], dtype=np.float32), dtype=float, device=DEVICE)
    out = wp.zeros(2, dtype=float, device=DEVICE)
    wp.launch(mixed_ok_kernel, dim=2, inputs=[a, b, out], device=DEVICE)
    np.testing.assert_allclose(out.numpy(), [1.0, 1.5])


def test_no_implicit_conversion() -> None:
    d = wp.zeros(1, dtype=wp.float64, device=DEVICE)
    assert_build_fails(max_f64_literal_kernel, d, wp.zeros(1, dtype=wp.float64, device=DEVICE))
    assert_build_fails(add_f32_f64_kernel, wp.zeros(1, dtype=wp.float32, device=DEVICE), d, wp.zeros(1, device=DEVICE))


def check_integer_operators(a: wp.array1d[wp.uint32], b: wp.array1d[wp.int64], i: int) -> None:
    s = wp.uint32(2)
    assert_type(a[i] >> s, wp.uint32)
    assert_type(a[i] << s, wp.uint32)
    assert_type((a[i] & s) | (a[i] ^ s), wp.uint32)
    assert_type(~a[i], wp.uint32)
    assert_type(b[i] // wp.int64(3), wp.int64)


def check_scalar_conversions(c: wp.array1d[wp.uint8], h: wp.array1d[wp.float16], i: int) -> None:
    assert_type(wp.uint32(c[i]), wp.uint32)  # any Warp scalar converts
    assert_type(wp.int64(h[i]), wp.int64)
    assert_type(wp.uint64(np.uint64(3)), wp.uint64)  # and NumPy scalars on the host
    assert_type(wp.float16(np.float32(1.5)), wp.float16)


@wp.kernel(module="unique")
def integer_operators_kernel(
    a: wp.array1d[wp.uint32],
    b: wp.array1d[wp.int64],
    c: wp.array1d[wp.uint8],
    out: wp.array1d[wp.uint32],
    out64: wp.array1d[wp.int64],
):
    i = wp.tid()
    x = a[i]
    s = wp.uint32(2)
    out[i] = ((x >> s) | (x << s)) & ~(x ^ wp.uint32(c[i])) // s
    out64[i] = b[i] // wp.int64(3)


@wp.kernel(module="unique")
def shift_by_int_kernel(a: wp.array1d[wp.uint32], out: wp.array1d[wp.uint32]):
    i = wp.tid()
    out[i] = a[i] >> 1  # E: `>>` is not supported between `uint32` and `Literal[1]`


def test_integer_operators() -> None:
    a = np.array([7, 1024], dtype=np.uint32)
    b = np.array([10, -7], dtype=np.int64)
    c = np.array([3, 5], dtype=np.uint8)
    out = wp.zeros(2, dtype=wp.uint32, device=DEVICE)
    out64 = wp.zeros(2, dtype=wp.int64, device=DEVICE)
    inputs = [wp.array(a, dtype=wp.uint32, device=DEVICE), wp.array(b, dtype=wp.int64, device=DEVICE)]
    inputs += [wp.array(c, dtype=wp.uint8, device=DEVICE), out, out64]
    wp.launch(integer_operators_kernel, dim=2, inputs=inputs, device=DEVICE)
    s = np.uint32(2)
    np.testing.assert_array_equal(out.numpy(), ((a >> s) | (a << s)) & ~(a ^ c.astype(np.uint32)) // s)
    np.testing.assert_array_equal(out64.numpy(), [3, -3])  # Warp truncates towards zero, like C
    assert_build_fails(shift_by_int_kernel, inputs[0], out)


def test_scalar_conversions() -> None:
    assert int(wp.uint64(np.uint64(3))) == 3
    assert float(wp.float16(np.float32(1.5))) == 1.5
