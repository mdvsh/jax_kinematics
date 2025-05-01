"""Tests for the transforms module."""

import jax
import jax.numpy as jnp
import pytest
import numpy as np
import hypothesis
from hypothesis import strategies as st
from hypothesis import given, settings

from jax_kinematics.transforms import rotation, transform

# Use hypothesis profile for CI
hypothesis.settings.register_profile("ci", max_examples=10, deadline=None)

# Basic tests
def test_quaternion_to_matrix_identity():
    """Test quaternion_to_matrix with identity quaternion."""
    identity_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    matrix = rotation.quaternion_to_matrix(identity_quat)
    expected = jnp.eye(3)
    np.testing.assert_allclose(matrix, expected, rtol=1e-6, atol=1e-6)

def test_matrix_to_quaternion_identity():
    """Test matrix_to_quaternion with identity matrix."""
    identity_matrix = jnp.eye(3)
    quat = rotation.matrix_to_quaternion(identity_matrix)
    expected = jnp.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(quat, expected, rtol=1e-6, atol=1e-6)

def test_transform_compose():
    """Test composition of transforms."""
    # Translation + rotation transforms
    t1 = transform.Transform3d.from_pos_quat(
        pos=jnp.array([1.0, 0.0, 0.0]),
        quat=jnp.array([1.0, 0.0, 0.0, 0.0])
    )
    
    # 90° rotation around Z
    t2 = transform.Transform3d.from_pos_quat(
        pos=jnp.array([0.0, 1.0, 0.0]),
        quat=jnp.array([0.7071068, 0.0, 0.0, 0.7071068])
    )
    
    # Compose transforms
    result = t1.compose(t2)
    
    # Test point transformation
    point = jnp.array([1.0, 0.0, 0.0])
    transformed = result.transform_points(point)
    
    # Expected result after applying both transforms
    expected = jnp.array([[1.0, 2.0, 0.0]])
    np.testing.assert_allclose(transformed, expected, rtol=1e-6, atol=1e-6)

# JIT tests
def test_quaternion_to_matrix_jit():
    """Test quaternion_to_matrix with JIT."""
    jitted_func = jax.jit(rotation.quaternion_to_matrix)
    quat = jnp.array([0.7071068, 0.0, 0.7071068, 0.0])  # 90° around Y
    matrix = jitted_func(quat)
    expected = jnp.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    np.testing.assert_allclose(matrix, expected, rtol=1e-6, atol=1e-6)

def test_matrix_to_quaternion_jit():
    """Test matrix_to_quaternion with JIT."""
    jitted_func = jax.jit(rotation.matrix_to_quaternion)
    matrix = jnp.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    quat = jitted_func(matrix)
    expected = jnp.array([0.7071068, 0.0, 0.7071068, 0.0])  # 90° around Y
    np.testing.assert_allclose(quat, expected, rtol=1e-6, atol=1e-6)

# Batched tests
def test_transform_points_batched():
    """Test transform_points with batched inputs."""
    # Create batch of 10 transforms
    batch_size = 10
    positions = jnp.tile(jnp.array([1.0, 2.0, 3.0]), (batch_size, 1))
    quats = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (batch_size, 1))
    
    transforms = transform.Transform3d.from_pos_quat(positions, quats)
    
    # Create points (2x3)
    points = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    # Transform points
    transformed = transforms.transform_points(points)
    
    # Should have shape (batch_size, 2, 3)
    assert transformed.shape == (batch_size, 2, 3)
    
    # All batches should produce the same result in this case
    for i in range(batch_size):
        expected = points + positions[i]
        np.testing.assert_allclose(transformed[i], expected, rtol=1e-6, atol=1e-6)

# Property-based tests with hypothesis - explicit key handling
@given(st.integers(min_value=0, max_value=100))
@settings(deadline=None)
def test_quaternion_roundtrip(seed):
    """Test quaternion -> matrix -> quaternion roundtrip with explicit key."""
    key = jax.random.PRNGKey(seed)
    # Generate random quaternion
    quat = jax.random.uniform(key, (4,), minval=-1.0, maxval=1.0)
    # Normalize input quaternion
    quat = rotation.normalize_quaternions(quat)
    
    # Convert to matrix and back
    matrix = rotation.quaternion_to_matrix(quat)
    quat2 = rotation.matrix_to_quaternion(matrix)
    
    # Ensure quaternions represent the same rotation
    # Need to handle q and -q representing the same rotation
    dot_product = jnp.abs(jnp.sum(quat * quat2))
    assert dot_product > 0.999  # Close to 1 means same rotation

@given(st.integers(min_value=0, max_value=100))
@settings(deadline=None)
def test_transform_inverse_property(seed):
    """Test that T * T^-1 = Identity with explicit key generation."""
    # Generate random transforms with explicit keys
    master_key = jax.random.PRNGKey(seed)
    key1, key2, key3 = jax.random.split(master_key, 3)
    
    batch_size = 5
    num_points = 10
    
    # Generate random transforms
    positions = jax.random.uniform(key1, (batch_size, 3), minval=-5.0, maxval=5.0)
    quats_raw = jax.random.uniform(key2, (batch_size, 4), minval=-1.0, maxval=1.0)
    quats = rotation.normalize_quaternions(quats_raw)
    
    # Create transforms
    transforms = transform.Transform3d.from_pos_quat(positions, quats)
    inverse_transforms = transforms.inverse()
    
    # Generate random points
    points = jax.random.uniform(key3, (num_points, 3), minval=-10.0, maxval=10.0)
    
    # Apply transform and then inverse, should get original points
    transformed = transforms.transform_points(points)
    back_to_original = inverse_transforms.transform_points(transformed)
    
    # Check that we get back the original points
    # Reshape for comparison since transform_points always returns with batch dim
    original_batched = jnp.broadcast_to(points[None], (batch_size, num_points, 3))
    np.testing.assert_allclose(back_to_original, original_batched, rtol=1e-5, atol=1e-5)