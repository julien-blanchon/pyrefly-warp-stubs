# Shape rules for the Warp stubs, evaluated by Pyrefly at type-check time.
#
# Hand-written (not generated). `reshape_shape`, `transpose_shape` and
# `squeeze_shape` are adapted from pyrefly-numpy-stubs (Copyright (c) Meta
# Platforms, Inc. and affiliates, MIT license).
#
# This module does not exist at runtime: it is only imported by the stubs.

import shape_extensions.dsl as dsl
from shape_extensions import Int, IntTuple, type_shape_dsl_function

@type_shape_dsl_function
def reshape_shape(shape: IntTuple, target: IntTuple) -> IntTuple:
    """`numpy.reshape` semantics: at most one `-1`, element counts must agree."""
    inferred = tuple(
        (1 for dimension in target if dsl.is_concrete_int(dimension) and dimension == -1)
    )
    if len(inferred) > 1:
        return dsl.Invalid("reshape allows only one inferred dimension")
    if any(dsl.is_concrete_int(dimension) and dimension < -1 for dimension in target):
        return dsl.Invalid("reshape dimensions must be at least -1")
    known_shape = dsl.IntTuple(
        (
            dimension
            for dimension in target
            if not (dsl.is_concrete_int(dimension) and dimension == -1)
        )
    )
    known = dsl.prod(known_shape)
    total = dsl.prod(shape)
    if len(inferred) == 0:
        if dsl.is_concrete_int(total) and dsl.is_concrete_int(known) and total != known:
            return dsl.Invalid("reshape target element count does not match the input")
        return target
    if dsl.is_concrete_int(known):
        if known == 0:
            return dsl.Invalid("reshape cannot infer a dimension from zero elements")
        if dsl.is_concrete_int(total) and total % known != 0:
            return dsl.Invalid("reshape cannot infer an integral dimension")
    return dsl.IntTuple(
        (
            total // known if dsl.is_concrete_int(dimension) and dimension == -1 else dimension
            for dimension in target
        )
    )

@type_shape_dsl_function
def flatten_shape(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((dsl.prod(shape),))

@type_shape_dsl_function
def transpose_shape(shape: IntTuple, axes: IntTuple) -> IntTuple:
    """`numpy.transpose` with an explicit permutation."""
    if len(axes) != len(shape):
        return dsl.Invalid("transpose axes must match the array rank")
    if any(axis < 0 - len(shape) or axis >= len(shape) for axis in axes):
        return dsl.Invalid("transpose axis out of bounds")
    normalized = dsl.IntTuple((axis + len(shape) if axis < 0 else axis for axis in axes))
    duplicates = dsl.IntTuple(
        (
            1 if any(normalized[index] == normalized[other] for other in range(index)) else 0
            for index in range(len(shape))
        )
    )
    if any(duplicate == 1 for duplicate in duplicates):
        return dsl.Invalid("transpose axes must be unique")
    return dsl.IntTuple((shape[axis] for axis in normalized))

@type_shape_dsl_function
def reverse_shape(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((shape[len(shape) - 1 - index] for index in range(len(shape))))

@type_shape_dsl_function
def squeeze_shape(shape: IntTuple, axis: int | tuple[int, ...] | None) -> IntTuple:
    """`wp.tile_squeeze`: drop all extent-1 dimensions, or the selected ones."""
    if axis is None:
        return dsl.IntTuple((extent for extent in shape if extent != 1))
    if dsl.is_int_value(axis):
        axes = (axis,)
    else:
        axes = axis
    if any(item < 0 - len(shape) or item >= len(shape) for item in axes):
        return dsl.Invalid("squeeze axis out of bounds")
    normalized = tuple(item + len(shape) if item < 0 else item for item in axes)
    if any(normalized.count(item) > 1 for item in normalized):
        return dsl.Invalid("squeeze axes must be unique")
    if any(shape[item] != 1 for item in normalized):
        return dsl.Invalid("squeeze axis must have length 1")
    return dsl.IntTuple((shape[index] for index in range(len(shape)) if index not in normalized))

@type_shape_dsl_function
def remove_axis_shape(shape: IntTuple, axis: int) -> IntTuple:
    """Axis reductions (`wp.tile_sum(t, axis=...)`) drop the reduced dimension."""
    if axis < 0 - len(shape) or axis >= len(shape):
        return dsl.Invalid("reduction axis out of bounds")
    if axis < 0:
        return dsl.IntTuple(
            (shape[index] for index in range(len(shape)) if index != axis + len(shape))
        )
    return dsl.IntTuple((shape[index] for index in range(len(shape)) if index != axis))

@type_shape_dsl_function
def broadcast_to_shape(shape: IntTuple, target: IntTuple) -> IntTuple:
    """`numpy.broadcast_to`: one-way, right-aligned, source extents of 1 stretch."""
    if len(target) < len(shape):
        return dsl.Invalid("broadcast target must have at least as many dimensions as the tile")
    aligned = target[len(target) - len(shape) :]
    mismatches = dsl.IntTuple(
        (
            1
            if dsl.is_concrete_int(source)
            and dsl.is_concrete_int(extent)
            and source != 1
            and source != extent
            else 0
            for source, extent in zip(shape, aligned)
        )
    )
    if any(mismatch == 1 for mismatch in mismatches):
        return dsl.Invalid("tile shape cannot be broadcast to the target shape")
    return target

@type_shape_dsl_function
def arange_shape(start: Int, stop: Int, step: Int) -> IntTuple:
    """`wp.tile_arange(start, stop, step)`: `ceil((stop - start) / step)` elements."""
    if dsl.is_concrete_int(step) and step == 0:
        return dsl.Invalid("tile_arange step must be nonzero")
    if dsl.is_concrete_int(step) and step == 1:
        return dsl.IntTuple((stop - start,))
    if dsl.is_concrete_int(start) and dsl.is_concrete_int(stop) and dsl.is_concrete_int(step):
        if 0 < step:
            return dsl.IntTuple(((stop - start + step - 1) // step,))
        return dsl.IntTuple(((start - stop - step - 1) // (0 - step),))
    return dsl.IntTuple.gradual()

# Variants taking a single integer, for APIs that accept `shape=8` as well as
# `shape=(8,)`. A list literal such as `[N]` is not accepted as a DSL argument
# in a return annotation, so the wrapping happens here.

@type_shape_dsl_function
def reshape_int_shape(shape: IntTuple, n: Int) -> IntTuple:
    target = dsl.IntTuple((n,))
    return reshape_shape(shape, target)

@type_shape_dsl_function
def broadcast_to_int_shape(shape: IntTuple, n: Int) -> IntTuple:
    target = dsl.IntTuple((n,))
    return broadcast_to_shape(shape, target)
