"""SE(3) and se(3) Lie group operations in JAX.

This module implements SE(3) rigid body transforms using homogeneous matrices
and 6D twist vectors. All functions are pure, JIT-able, and operate on JAX arrays.
This implementation focuses on numerical stability, especially for small angles.
"""


import jax
import jax.numpy as jnp

from . import so3

Array = jax.Array


def from_position_and_rotation(p: Array, R: Array) -> Array:
    """
    Construct SE(3) transform from position and rotation.

    Args:
        p: (..., 3) position vector
        R: (..., 3, 3) rotation matrix

    Returns:
        (..., 4, 4) homogeneous transformation matrix
    """
    # Ensure consistent batch shapes
    batch_shape = jnp.broadcast_shapes(p.shape[:-1], R.shape[:-2])
    p = jnp.broadcast_to(p, batch_shape + (3,))
    R = jnp.broadcast_to(R, batch_shape + (3, 3))

    # Create an identity matrix and fill it
    T = jnp.zeros(batch_shape + (4, 4), dtype=p.dtype)
    T = T.at[..., :3, :3].set(R)
    T = T.at[..., :3, 3].set(p)
    T = T.at[..., 3, 3].set(1.0)

    return T


def exp(twist: Array) -> Array:
    """
    SE(3) exponential map: convert twist to transformation matrix.

    This function is numerically stable, using Taylor series approximations
    for small angles to avoid division by zero.

    Args:
        twist: (..., 6) array of twists [vx, vy, vz, wx, wy, wz].
               The first 3 elements are linear velocity, last 3 are angular.

    Returns:
        (..., 4, 4) array of transformation matrices.
    """
    v, w = twist[..., :3], twist[..., 3:]
    angle = jnp.linalg.norm(w, axis=-1, keepdims=True)

    # Use a small epsilon for safe division
    eps = jnp.finfo(twist.dtype).eps

    # Rotation part is just the SO(3) exponential map
    R = so3.exp(w)

    # Translation part requires the V matrix.
    # The coefficients of V are computed with numerically stable approximations.
    angle_sq = angle * angle

    # Check for small angles
    is_small_angle = angle < 1e-6

    # Coefficient A = (1 - cos(theta)) / theta^2
    # Taylor expansion for small theta: A ≈ 1/2 - theta^2/24 + theta^4/720
    A = jnp.where(is_small_angle, 0.5 - angle_sq / 24.0, (1.0 - jnp.cos(angle)) / (angle_sq + eps))

    # Coefficient B = (theta - sin(theta)) / theta^3
    # Taylor expansion for small theta: B ≈ 1/6 - theta^2/120 + theta^4/5040
    B = jnp.where(is_small_angle, 1.0 / 6.0 - angle_sq / 120.0, (angle - jnp.sin(angle)) / (angle_sq * angle + eps))

    K = so3.skew_symmetric(w)
    K_sq = jnp.matmul(K, K)

    I = jnp.eye(3, dtype=twist.dtype)
    I = jnp.broadcast_to(I, K.shape)

    # V = I + A*K + B*K^2
    V = I + A * K + B * K_sq

    t = jnp.einsum("...ij,...j->...i", V, v)

    return from_position_and_rotation(t, R)


def log(T: Array) -> Array:
    """
    SE(3) logarithm map: convert transformation matrix to twist.

    This function is numerically stable, especially for small rotation angles.

    Args:
        T: (..., 4, 4) array of transformation matrices.

    Returns:
        (..., 6) array of twists [vx, vy, vz, wx, wy, wz].
    """
    R, t = T[..., :3, :3], T[..., :3, 3]

    # Angular part is the SO(3) logarithm
    w = so3.log(R)
    angle = jnp.linalg.norm(w, axis=-1, keepdims=True)

    # Use a small epsilon for safe division
    eps = jnp.finfo(T.dtype).eps

    # Linear part requires the inverse of the V matrix from the exp map.
    K = so3.skew_symmetric(w)

    # Check for small angles
    is_small_angle = angle < 1e-6

    half_angle = angle / 2.0

    # Coefficient for the K^2 term, C = (1/theta^2) * (1 - (A/2B))
    # where A and B are the coefficients from the exp map.
    # This simplifies to 1/theta^2 * (1 - (theta * sin(theta)) / (2 * (1-cos(theta))))
    # A more stable form is (1 - theta * cot(theta/2) / 2) / theta^2

    # cot(x) = cos(x)/sin(x)
    cot_half_angle = jnp.cos(half_angle) / (jnp.sin(half_angle) + eps)

    # For small angles, this coefficient C -> 1/12
    C = jnp.where(is_small_angle, 1.0 / 12.0, (1.0 - half_angle * cot_half_angle) / (angle * angle + eps))

    I = jnp.eye(3, dtype=T.dtype)
    I = jnp.broadcast_to(I, K.shape)

    # V_inv = I - 0.5*K + C*K^2
    V_inv = I - 0.5 * K + C * jnp.matmul(K, K)

    v = jnp.einsum("...ij,...j->...i", V_inv, t)

    return jnp.concatenate([v, w], axis=-1)


def multiply(T1: Array, T2: Array) -> Array:
    """
    Multiply two SE(3) transformation matrices.

    Args:
        T1: (..., 4, 4) first transformation matrix
        T2: (..., 4, 4) second transformation matrix

    Returns:
        (..., 4, 4) result of T1 @ T2
    """
    return jnp.matmul(T1, T2)


def inverse(T: Array) -> Array:
    """
    Compute inverse of SE(3) transformation matrix.

    Uses the block structure for efficient computation:
    T^-1 = [[R^T, -R^T @ t], [0, 1]]

    Args:
        T: (..., 4, 4) transformation matrix

    Returns:
        (..., 4, 4) inverse transformation matrix
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    # Rotation inverse (transpose)
    R_inv = jnp.swapaxes(R, -1, -2)

    # Translation inverse
    t_inv = -jnp.einsum("...ij,...j->...i", R_inv, t)

    return from_position_and_rotation(t_inv, R_inv)


def apply(T: Array, points: Array) -> Array:
    """
    Apply SE(3) transformation to points.

    Args:
        T: (..., 4, 4) transformation matrix
        points: (..., 3) or (..., N, 3) points to transform

    Returns:
        (..., 3) or (..., N, 3) transformed points
    """
    # Convert points to homogeneous coordinates
    # This works for both single vector (..., 3) and batch of vectors (..., N, 3)
    ones = jnp.ones_like(points[..., 0:1])
    points_h = jnp.concatenate([points, ones], axis=-1)

    # Apply transformation using einsum for flexible broadcasting
    transformed_h = jnp.einsum("...ij,...j->...i", T, points_h)

    # Convert back to Cartesian coordinates by dividing by the homogeneous coordinate
    # The homogeneous coordinate should always be 1 for SE(3) transforms
    return transformed_h[..., :3]


def get_position(T: Array) -> Array:
    """
    Extract position from SE(3) transformation matrix.

    Args:
        T: (..., 4, 4) transformation matrix

    Returns:
        (..., 3) position vector
    """
    return T[..., :3, 3]


def get_rotation(T: Array) -> Array:
    """
    Extract rotation matrix from SE(3) transformation matrix.

    Args:
        T: (..., 4, 4) transformation matrix

    Returns:
        (..., 3, 3) rotation matrix
    """
    return T[..., :3, :3]


def adjoint(T: Array) -> Array:
    """
    Compute the adjoint matrix of SE(3) transformation.

    The adjoint matrix is used to transform twists between coordinate frames.

    Args:
        T: (..., 4, 4) transformation matrix

    Returns:
        (..., 6, 6) adjoint matrix
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    # Skew-symmetric matrix of translation
    t_skew = so3.skew_symmetric(t)

    # Build adjoint matrix
    zeros = jnp.zeros_like(R)

    # Adjoint matrix is [[R, [t]_x R], [0, R]]
    top = jnp.concatenate([R, jnp.matmul(t_skew, R)], axis=-1)
    bottom = jnp.concatenate([zeros, R], axis=-1)

    return jnp.concatenate([top, bottom], axis=-2)
