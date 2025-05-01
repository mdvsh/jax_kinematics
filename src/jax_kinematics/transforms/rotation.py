"""Rotation conversion utilities in JAX."""

import jax
import jax.numpy as jnp
from typing import Tuple, Union, Optional

# Type aliases
Array = jax.Array
Scalar = Union[float, Array]

def quaternion_to_matrix(quaternions: Array) -> Array:
    """
    Convert quaternions to rotation matrices.
    
    Args:
        quaternions: (..., 4) array of quaternions in (w, x, y, z) format
        
    Returns:
        (..., 3, 3) array of rotation matrices
    """
    # Normalize quaternions for numerical stability
    quaternions = normalize_quaternions(quaternions)
    
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

def normalize_quaternions(quaternions: Array) -> Array:
    """Normalize quaternions to unit length."""
    return quaternions / jnp.linalg.norm(quaternions, axis=-1, keepdims=True)

def matrix_to_quaternion(matrix: Array) -> Array:
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
    quaternion = normalize_quaternions(quaternion)
    
    return quaternion