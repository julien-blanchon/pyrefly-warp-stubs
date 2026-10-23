"""Textures, meshes, BVHs, hash grids and volumes in kernels.

`check_*` functions are only type checked (an `E:` comment marks an expected
Pyrefly error); `test_*` functions launch the equivalent kernels on the CPU.
"""

from __future__ import annotations

from typing import assert_type

import numpy as np
import warp as wp

from _launch import DEVICE, assert_build_fails


def check_dtype_results(tex: wp.Texture3D, volume: wp.uint64, uvw: wp.vec3) -> None:
    # The result type is the `dtype` argument (upstream: `Any`).
    assert_type(wp.texture_sample(tex, uvw, dtype=float), float)
    assert_type(wp.texture_sample(tex, uvw, dtype=wp.vec4), wp.vec4)
    assert_type(wp.texture_sample(tex, 0.5, 0.5, 0.5, dtype=float), float)
    assert_type(wp.volume_sample(volume, uvw, wp.Volume.LINEAR, dtype=float), float)
    assert_type(wp.cast(1.5, dtype=int), int)


def check_query_fields(mesh: wp.uint64, p: wp.vec3, d: wp.vec3) -> None:
    q = wp.mesh_query_point(mesh, p, 10.0)
    assert_type(q.result, bool)
    assert_type(q.face, int)
    assert_type(q.u, float)
    r = wp.mesh_query_ray(mesh, p, d, 10.0)
    assert_type(r.t, float)
    assert_type(r.normal, wp.vec3)
    assert_type(wp.mesh_get(mesh).points[0], wp.vec3)


def check_host_objects(mesh: wp.Mesh, grid: wp.HashGrid, bvh: wp.Bvh, tex: wp.Texture3D) -> None:
    # `uint64`, as kernel parameters and `@wp.struct` fields declare ids.
    assert_type(mesh.id, wp.uint64)
    assert_type(grid.id, wp.uint64)
    assert_type(bvh.id, wp.uint64)
    assert_type(tex.depth, int)


@wp.kernel(module="unique")
def spatial_kernel(tex: wp.Texture3D, mesh: wp.uint64, out: wp.array1d[float]):
    color = wp.texture_sample(tex, wp.vec3(0.5, 0.5, 0.5), dtype=wp.vec4)
    out[0] = color[0] - wp.texture_sample(tex, 0.5, 0.5, 0.5, dtype=float)  # same texel, 1 channel
    q = wp.mesh_query_point(mesh, wp.vec3(0.2, 0.2, 1.0), 10.0)
    out[1] = wp.where(q.result, q.u + q.v, -1.0)
    r = wp.mesh_query_ray(mesh, wp.vec3(0.2, 0.2, 1.0), wp.vec3(0.0, 0.0, -1.0), 10.0)
    out[2] = r.t
    for face in wp.mesh_query_aabb(mesh, wp.vec3(-1.0, -1.0, -1.0), wp.vec3(2.0, 2.0, 2.0)):
        out[3] += 1.0 + float(face)
    out[4] = wp.mesh_get(mesh).points[1][0]


@wp.kernel(module="unique")
def bad_bvh_iteration_kernel(bvh: wp.uint64, out: wp.array1d[float]):
    # BVH queries are walked with `wp.bvh_query_next`; Warp cannot iterate them.
    for i in wp.bvh_query_aabb(bvh, wp.vec3(-1.0, -1.0, -1.0), wp.vec3(2.0, 2.0, 2.0)):  # E: Type `BvhQuery` is not iterable
        out[0] += 1.0


def test_spatial_queries() -> None:
    points = wp.array(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32), dtype=wp.vec3, device=DEVICE)
    mesh = wp.Mesh(points=points, indices=wp.array([0, 1, 2], dtype=int, device=DEVICE))
    tex = wp.Texture3D(np.random.default_rng(0).random((4, 4, 4)).astype(np.float32), device=DEVICE)
    out = wp.zeros(5, dtype=float, device=DEVICE)
    wp.launch(spatial_kernel, dim=1, inputs=[tex, mesh.id, out], device=DEVICE)
    np.testing.assert_allclose(out.numpy(), [0.0, 0.8, 1.0, 1.0, 1.0], rtol=1e-5, atol=1e-6)


def test_rejected_bvh_iteration() -> None:
    points = wp.array(np.zeros((2, 3), dtype=np.float32), dtype=wp.vec3, device=DEVICE)
    bvh = wp.Bvh(points, points)
    assert_build_fails(bad_bvh_iteration_kernel, bvh.id, wp.zeros(1, dtype=float, device=DEVICE))
