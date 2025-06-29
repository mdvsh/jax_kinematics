"""SO(3) and so(3) Lie group operations in JAX.

This module implements the mathematical foundation for 3D rotations using
rotation matrices and axis-angle representations. All functions are pure,
JIT-able, and operate on JAX arrays.
"""

import jax
import jax.numpy as jnp
from typing import Union

Array = jax.Array


def exp(log_r: Array) -> Array:
    """
    SO(3) exponential map: convert axis-angle vector to rotation matrix.
    
    Implements Rodrigues' formula to convert a 3D axis-angle vector (so(3))
    to a rotation matrix (SO(3)). This is fundamental for applying joint motion.
    
    Args:
        log_r: (..., 3) array of axis-angle vectors
        
    Returns:
        (..., 3, 3) array of rotation matrices
    """
    # Compute angle (magnitude of axis-angle vector)
    angle = jnp.linalg.norm(log_r, axis=-1, keepdims=True)
    
    # Handle near-zero angles for numerical stability
    small_angle = angle < 1e-8
    
    # For small angles, use Taylor expansion
    # For larger angles, use full Rodrigues formula
    cos_angle = jnp.where(small_angle, 1.0 - 0.5 * angle**2, jnp.cos(angle))
    sin_angle = jnp.where(small_angle, angle - angle**3 / 6.0, jnp.sin(angle))
    
    # Normalized axis (handle zero angle case)
    axis = jnp.where(angle > 1e-8, log_r / angle, log_r)
    
    # Cross-product matrix (skew-symmetric matrix)
    K = skew_symmetric(axis)
    
    # Rodrigues formula: R = I + sin(θ) * K + (1 - cos(θ)) * K²
    I = jnp.eye(3, dtype=log_r.dtype)
    I = jnp.broadcast_to(I, log_r.shape[:-1] + (3, 3))
    
    R = (I + 
         sin_angle[..., None] * K + 
         (1.0 - cos_angle)[..., None] * jnp.matmul(K, K))
    
    return R


def log(R: Array) -> Array:
    """
    SO(3) logarithm map: convert rotation matrix to axis-angle vector.
    
    This is the inverse of exp(). Crucial for calculating orientation errors
    in a principled way for IK.
    
    Args:
        R: (..., 3, 3) array of rotation matrices
        
    Returns:
        (..., 3) array of axis-angle vectors
    """
    # Compute trace
    trace = jnp.trace(R, axis1=-2, axis2=-1)
    
    # Compute angle
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)  # Numerical stability
    angle = jnp.arccos(cos_angle)
    
    # Handle different cases based on angle
    small_angle = angle < 1e-8
    near_pi = jnp.abs(angle - jnp.pi) < 1e-8
    
    # For small angles, use approximation
    sin_angle = jnp.where(small_angle, 1.0 - angle**2 / 6.0, jnp.sin(angle))
    
    # Extract axis from skew-symmetric part
    # For general case: axis = [R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]] / (2 * sin(angle))
    skew_part = jnp.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0], 
        R[..., 1, 0] - R[..., 0, 1]
    ], axis=-1)
    
    # For small angles
    axis_small = skew_part / 2.0
    
    # For general case
    axis_general = skew_part / (2.0 * sin_angle[..., None])
    
    # For angles near π, use eigenvector method
    # Find eigenvector corresponding to eigenvalue 1
    B = (R + jnp.eye(3)) / 2.0
    # Use the column with largest diagonal element
    diag_vals = jnp.diagonal(B, axis1=-2, axis2=-1)
    max_idx = jnp.argmax(diag_vals, axis=-1)
    
    # Extract the corresponding column
    axis_pi = jnp.take_along_axis(B, max_idx[..., None, None], axis=-1)[..., 0]
    axis_pi = axis_pi / jnp.linalg.norm(axis_pi, axis=-1, keepdims=True)
    
    # Combine cases
    axis = jnp.where(
        small_angle[..., None], 
        axis_small,
        jnp.where(
            near_pi[..., None],
            axis_pi,
            axis_general
        )
    )
    
    # Return axis-angle vector
    return angle[..., None] * axis


def multiply(R1: Array, R2: Array) -> Array:
    """
    Multiply two rotation matrices.
    
    Args:
        R1: (..., 3, 3) first rotation matrix
        R2: (..., 3, 3) second rotation matrix
        
    Returns:
        (..., 3, 3) result of R1 @ R2
    """
    return jnp.matmul(R1, R2)


def inverse(R: Array) -> Array:
    """
    Compute inverse of rotation matrix.
    
    For rotation matrices, the inverse is simply the transpose.
    
    Args:
        R: (..., 3, 3) rotation matrix
        
    Returns:
        (..., 3, 3) inverse rotation matrix
    """
    return jnp.swapaxes(R, -1, -2)


def apply(R: Array, v: Array) -> Array:
    """
    Apply rotation to vector(s).
    
    Args:
        R: (..., 3, 3) rotation matrix
        v: (..., 3) or (..., N, 3) vector(s) to rotate
        
    Returns:
        (..., 3) or (..., N, 3) rotated vector(s)
    """
    if v.ndim == R.ndim - 1:  # Single vector case
        return jnp.einsum('...ij,...j->...i', R, v)
    else:  # Multiple vectors case
        return jnp.einsum('...ij,...nj->...ni', R, v)


def skew_symmetric(v: Array) -> Array:
    """
    Convert 3D vector to skew-symmetric matrix.
    
    Args:
        v: (..., 3) vector
        
    Returns:
        (..., 3, 3) skew-symmetric matrix
    """
    zeros = jnp.zeros(v.shape[:-1], dtype=v.dtype)
    
    return jnp.stack([
        jnp.stack([zeros, -v[..., 2], v[..., 1]], axis=-1),
        jnp.stack([v[..., 2], zeros, -v[..., 0]], axis=-1),
        jnp.stack([-v[..., 1], v[..., 0], zeros], axis=-1)
    ], axis=-2)


def from_quaternion(quaternions: Array) -> Array:
    """
    Convert quaternions to rotation matrices.
    
    Args:
        quaternions: (..., 4) array of quaternions in (w, x, y, z) format
        
    Returns:
        (..., 3, 3) array of rotation matrices
    """
    # Normalize quaternions for numerical stability
    quaternions = quaternions / jnp.linalg.norm(quaternions, axis=-1, keepdims=True)
    
    # Unpack quaternion components - preserving batch dimensions
    w, x, y, z = jnp.moveaxis(quaternions, -1, 0)
    
    # Compute the components efficiently
    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    # Build matrix in one go for better compilation
    matrix = jnp.stack([
        jnp.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], axis=-1),
        jnp.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], axis=-1),
        jnp.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], axis=-1)
    ], axis=-2)
    
    return matrix


def to_quaternion(matrix: Array) -> Array:
    """
    Convert rotation matrices to quaternions (w, x, y, z).
    Batch-safe and JIT-friendly implementation.
    
    Args:
        matrix: (..., 3, 3) array of rotation matrices
        
    Returns:
        (..., 4) array of quaternions in (w, x, y, z) format
    """
    # Extract matrix elements
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]
    
    # Calculate trace for the entire batch
    trace = m00 + m11 + m22
    
    # Use dtype-adaptive epsilon
    eps = jnp.finfo(matrix.dtype).eps
    
    # Compute four candidate quaternions for the entire batch
    q0 = jnp.stack([
        trace + 1.0,
        m21 - m12,
        m02 - m20,
        m10 - m01
    ], axis=-1) * 0.5  # Scale by 0.5 for correct gradients
    
    q1 = jnp.stack([
        m21 - m12,
        m00 - m11 - m22 + 1.0,
        m01 + m10,
        m02 + m20
    ], axis=-1) * 0.5  # Scale by 0.5
    
    q2 = jnp.stack([
        m02 - m20,
        m01 + m10,
        m11 - m00 - m22 + 1.0,
        m12 + m21
    ], axis=-1) * 0.5  # Scale by 0.5
    
    q3 = jnp.stack([
        m10 - m01,
        m02 + m20,
        m12 + m21,
        m22 - m00 - m11 + 1.0
    ], axis=-1) * 0.5  # Scale by 0.5
    
    # Scale factors for each case
    s0 = 1.0 / jnp.sqrt(jnp.maximum(1.0 + trace, eps))
    s1 = 1.0 / jnp.sqrt(jnp.maximum(1.0 + m00 - m11 - m22, eps))
    s2 = 1.0 / jnp.sqrt(jnp.maximum(1.0 + m11 - m00 - m22, eps))
    s3 = 1.0 / jnp.sqrt(jnp.maximum(1.0 + m22 - m00 - m11, eps))
    
    # Apply scales
    q0 = q0 * s0[..., None]
    q1 = q1 * s1[..., None]
    q2 = q2 * s2[..., None]
    q3 = q3 * s3[..., None]
    
    # Create masks for selecting the right quaternion
    mask0 = (trace > 0)
    mask1 = (~mask0) & (m00 > m11) & (m00 > m22)
    mask2 = (~mask0) & (~mask1) & (m11 > m22)
    mask3 = (~mask0) & (~mask1) & (~mask2)
    
    # Combine quaternions based on masks
    quaternion = (
        jnp.where(mask0[..., None], q0, 0) +
        jnp.where(mask1[..., None], q1, 0) +
        jnp.where(mask2[..., None], q2, 0) +
        jnp.where(mask3[..., None], q3, 0)
    )
    
    # Ensure non-negative scalar part and normalize
    quaternion = jnp.where(quaternion[..., 0:1] < 0, -quaternion, quaternion)
    quaternion = quaternion / jnp.linalg.norm(quaternion, axis=-1, keepdims=True)
    
    return quaternion