# `Tape.gradients` maps each recorded array to its adjoint (upstream: `{}`,
# read as `dict[Unknown, Unknown]`).

from typing import Any

class Tape:
    gradients: dict[wp.array[Any], wp.array[Any]]
