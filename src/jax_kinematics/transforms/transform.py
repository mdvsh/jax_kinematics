"""SE(3) rigid-body transforms implemented with JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .rotation import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    normalize_quaternions,
)

Array = jax.Array

@register_pytree_node_class  # let Transform3d work with jit / grad / vmap …
@dataclass(frozen=True)
class Transform3d:
    """Immutable homogeneous transform(s) – batch-friendly & JIT-friendly."""
    matrix: Array  # shape (..., 4, 4)

    # Constructors
    @classmethod
    def from_matrix(cls, matrix: Array) -> "Transform3d":
        if matrix.ndim == 2:         # single transform, no batch dim
            matrix = matrix[None]    # add batch dim
        if matrix.shape[-2:] != (4, 4):
            raise ValueError(f"matrix must have shape (...,4,4), got {matrix.shape}")
        return cls(matrix)

    @classmethod
    def from_pos_quat(cls, pos: Array, quat: Optional[Array] = None) -> "Transform3d":
        batch_shape = pos.shape[:-1]
        m = jnp.broadcast_to(jnp.eye(4, dtype=pos.dtype), batch_shape + (4, 4)).copy()

        m = m.at[..., :3, 3].set(pos)
        if quat is not None:
            rot = quaternion_to_matrix(normalize_quaternions(quat))
            m = m.at[..., :3, :3].set(rot)
        return cls(m)

    @classmethod
    def identity(cls, batch_shape: Tuple[int, ...] = (), *, dtype=jnp.float32) -> "Transform3d":
        m = jnp.eye(4, dtype=dtype)
        return cls(jnp.broadcast_to(m, batch_shape + (4, 4)))

    # PyTree boiler-plate
    def tree_flatten(self):
        return (self.matrix,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (matrix,) = children
        return cls(matrix)

    # Basic operations
    def compose(self, other: "Transform3d") -> "Transform3d":
        """Self ∘ other (apply *other* first, then self)."""
        return Transform3d(jnp.matmul(self.matrix, other.matrix))

    def inverse(self) -> "Transform3d":
        """SE(3) inverse using the block structure."""
        R = self.matrix[..., :3, :3]
        t = self.matrix[..., :3, 3]

        R_inv = jnp.swapaxes(R, -1, -2)
        t_inv = -jnp.matmul(R_inv, t[..., None])[..., 0]

        bottom = jnp.broadcast_to(
            jnp.array([[0.0, 0.0, 0.0, 1.0]], dtype=self.matrix.dtype),
            self.matrix.shape[:-2] + (1, 4),
        )
        return Transform3d(jnp.concatenate([jnp.concatenate([R_inv, t_inv[..., None]], -1), bottom], -2))

    # Point transformation
    def transform_points(self, points: Array) -> Array:
        """
        Apply the transform(s) to *points*.

        Accepted shapes
        ---------------
        * (3,)          – single point
        * (N, 3)        – many points
        * (B, N, 3)     – batched points

        Returns
        -------
        * (B, 3)        – if a single point was given
        * (B, N, 3)     – otherwise
        """
        # normalise input to (P_batch, N, 3)
        single_point = points.ndim == 1
        if single_point:                         # (3,) -> (1,3)
            points = points[None, :]

        if points.ndim == 2:                     # (N,3) -> (1,N,3)
            points = points[None, ...]

        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (3,), (N,3) or (B,N,3)")

        # homogeneous coordinates
        ones = jnp.ones(points.shape[:-1] + (1,), dtype=points.dtype)
        points_h = jnp.concatenate([points, ones], axis=-1)           # (..., N, 4)

        # apply transforms
        #   matrix:   (..., 4, 4)
        #   points_h: (..., N, 4)
        transformed_h = jnp.einsum("...ij,...nj->...ni", self.matrix, points_h)
        transformed   = transformed_h[..., :3] / transformed_h[..., 3:]

        # squeeze dummy point axis if caller passed a single pt -
        if single_point:
            transformed = transformed[..., 0, :]                      # (..., 3)

        return transformed

    # Convenience helpers
    def get_position(self) -> Array:
        return self.matrix[..., :3, 3]

    def get_rotation_matrix(self) -> Array:
        return self.matrix[..., :3, :3]

    def get_quaternion(self) -> Array:
        return matrix_to_quaternion(self.get_rotation_matrix())
