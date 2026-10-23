"""Particles: vectors, matrices and Warp's scalar rules, without tiles.

Shows: `wp.vec3` / `wp.mat33` values typed through kernels (`m * v` is a
vector, `wp.dot` a scalar), `float` being `wp.float32` (`wp.max(s, 0.0)` is
fine), class-level array declarations, and a symbolic particle count `N` tying
host-side arrays together. Mistakes that Warp would reject when building the
kernel are reported in the editor instead:

    x[i] * v[i]           # error: `*` is not supported between `Vector[float, Literal[3]]`
                          #        and `Vector[float, Literal[3]]` (use wp.cw_mul)
    v[i] * wp.float64(dt) # error: `*` is not supported between `Vector[float, Literal[3]]`
                          #        and `float64` (no implicit conversion)
    rotation44 * x[i]     # error: `*` is not supported between
                          #        `Matrix[float, Literal[4], Literal[4]]` and `Vector[float, Literal[3]]`

Run it with `python examples/particles.py` (CPU is fine).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from shape_extensions import IntVar


@wp.kernel
def step_kernel(
    x: wp.array1d[wp.vec3],
    v: wp.array1d[wp.vec3],
    rotation: wp.mat33,
    gravity: wp.vec3,
    dt: float,
    max_speed: float,
):
    i = wp.tid()
    velocity = v[i] + gravity * dt  # vec3 + vec3 * float -> vec3
    speed = wp.length(velocity)  # float
    velocity = velocity * (wp.min(speed, max_speed) / wp.max(speed, 1.0e-6))  # clamp the speed
    v[i] = velocity
    x[i] = rotation * (x[i] + velocity * dt)  # mat33 * vec3 -> vec3


@wp.kernel
def energy_kernel(v: wp.array1d[wp.vec3], mass: wp.array1d[float], total: wp.array1d[float]):
    i = wp.tid()
    wp.atomic_add(total, 0, 0.5 * mass[i] * wp.dot(v[i], v[i]))


class ParticleSystem:
    # Declared once, the rank and dtype hold for every method that uses them.
    x: wp.array1d[wp.vec3]
    v: wp.array1d[wp.vec3]
    mass: wp.array1d[float]

    def __init__(self, positions: np.ndarray, masses: np.ndarray, device: str = "cpu") -> None:
        self.x = wp.array(positions, dtype=wp.vec3, device=device)
        self.v = wp.zeros(len(positions), dtype=wp.vec3, device=device)
        self.mass = wp.array(masses, dtype=float, device=device)

    def step(self, dt: float, rotation: wp.mat33, gravity: wp.vec3 = wp.vec3(0.0, -9.8, 0.0)) -> None:
        wp.launch(
            step_kernel,
            dim=len(self.x),
            inputs=[self.x, self.v, rotation, gravity, dt, 5.0],
            device=self.x.device,
        )

    def kinetic_energy(self) -> float:
        return kinetic_energy(self.v, self.mass)


def kinetic_energy[N: IntVar](v: wp.array[wp.vec3, [N]], mass: wp.array[float, [N]]) -> float:
    """Total kinetic energy; `v` and `mass` must describe the same `N` particles."""
    total = wp.zeros(1, dtype=float, device=v.device)
    wp.launch(energy_kernel, dim=v.shape[0], inputs=[v, mass, total], device=v.device)
    return float(total.numpy()[0])


def main() -> None:
    rng = np.random.default_rng(0)
    n = 64
    system = ParticleSystem(rng.standard_normal((n, 3)).astype(np.float32), np.ones(n, dtype=np.float32))
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    x0 = system.x.numpy().copy()

    for _ in range(10):
        system.step(0.01, identity)

    v = np.zeros((n, 3), dtype=np.float32)
    x = x0.copy()
    for _ in range(10):  # NumPy reference of the same update
        v = v + np.array([0.0, -9.8, 0.0], dtype=np.float32) * 0.01
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        v = v * (np.minimum(speed, 5.0) / np.maximum(speed, 1.0e-6))
        x = x + v * 0.01
    np.testing.assert_allclose(system.x.numpy(), x, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(system.kinetic_energy(), 0.5 * (v * v).sum(), rtol=1e-5)
    print("ok:", n, "particles, kinetic energy", round(system.kinetic_energy(), 4))


if __name__ == "__main__":
    main()
