"""Tests for the transforms module."""

import hypothesis
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from jax_kinematics.transforms import se3, so3

# Use hypothesis profile for CI
hypothesis.settings.register_profile("ci", max_examples=10, deadline=None)


# Basic tests
def test_quaternion_to_matrix_identity():
    """Test quaternion_to_matrix with identity quaternion."""
    identity_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    matrix = so3.from_quaternion(identity_quat)
    expected = jnp.eye(3)
    np.testing.assert_allclose(matrix, expected, rtol=1e-6, atol=1e-6)


def test_matrix_to_quaternion_identity():
    """Test matrix_to_quaternion with identity matrix."""
    identity_matrix = jnp.eye(3)
    quat = so3.to_quaternion(identity_matrix)
    expected = jnp.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(quat, expected, rtol=1e-6, atol=1e-6)


def test_transform_compose():
    """Test composition of transforms."""
    # Create transforms using SE(3) functions
    # Translation by [1, 0, 0]
    t1 = se3.from_position_and_rotation(jnp.array([1.0, 0.0, 0.0]), jnp.eye(3))

    # Translation by [0, 1, 0] + 90° rotation around Z
    R_z90 = so3.from_quaternion(jnp.array([0.7071068, 0.0, 0.0, 0.7071068]))
    t2 = se3.from_position_and_rotation(jnp.array([0.0, 1.0, 0.0]), R_z90)

    # Compose transforms
    result = se3.multiply(t1, t2)

    # Test point transformation
    point = jnp.array([1.0, 0.0, 0.0])
    transformed = se3.apply(result, point)

    # Expected result after applying both transforms
    expected = jnp.array([1.0, 2.0, 0.0])
    np.testing.assert_allclose(transformed, expected, rtol=1e-6, atol=1e-6)


# JIT tests
def test_quaternion_to_matrix_jit():
    """Test quaternion_to_matrix with JIT."""
    jitted_func = jax.jit(so3.from_quaternion)
    quat = jnp.array([0.7071068, 0.0, 0.7071068, 0.0])  # 90° around Y
    matrix = jitted_func(quat)
    expected = jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(matrix, expected, rtol=1e-6, atol=1e-6)


def test_matrix_to_quaternion_jit():
    """Test matrix_to_quaternion with JIT."""
    jitted_func = jax.jit(so3.to_quaternion)
    matrix = jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    quat = jitted_func(matrix)
    expected = jnp.array([0.7071068, 0.0, 0.7071068, 0.0])  # 90° around Y
    np.testing.assert_allclose(quat, expected, rtol=1e-6, atol=1e-6)


# Batched tests
def test_transform_points_batched():
    """Test transform_points with batched inputs."""
    # Create batch of 10 transforms
    batch_size = 10
    positions = jnp.tile(jnp.array([1.0, 2.0, 3.0]), (batch_size, 1))
    rotations = jnp.tile(jnp.eye(3), (batch_size, 1, 1))

    transforms = se3.from_position_and_rotation(positions, rotations)

    # Create points (2x3)
    points = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # Transform points - need to broadcast for batch application
    transformed = jax.vmap(lambda T: se3.apply(T, points))(transforms)

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
    quat = quat / jnp.linalg.norm(quat)

    # Convert to matrix and back
    matrix = so3.from_quaternion(quat)
    quat2 = so3.to_quaternion(matrix)

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
    quats = quats_raw / jnp.linalg.norm(quats_raw, axis=-1, keepdims=True)

    # Create transforms
    rotations = jax.vmap(so3.from_quaternion)(quats)
    transforms = se3.from_position_and_rotation(positions, rotations)
    inverse_transforms = jax.vmap(se3.inverse)(transforms)

    # Generate random points
    points = jax.random.uniform(key3, (num_points, 3), minval=-10.0, maxval=10.0)

    # Apply transform and then inverse, should get original points
    transformed = jax.vmap(lambda T: se3.apply(T, points))(transforms)
    back_to_original = jax.vmap(lambda T, pts: se3.apply(T, pts))(inverse_transforms, transformed)

    # Check that we get back the original points
    # Reshape for comparison
    original_batched = jnp.broadcast_to(points[None], (batch_size, num_points, 3))
    np.testing.assert_allclose(back_to_original, original_batched, rtol=1e-5, atol=1e-5)


# SO(3) Lie Group Tests
def test_so3_exp_identity():
    """Test SO(3) exp with zero vector gives identity."""
    zero_vec = jnp.zeros(3)
    R = so3.exp(zero_vec)
    expected = jnp.eye(3)
    np.testing.assert_allclose(R, expected, rtol=1e-6, atol=1e-6)


def test_so3_log_identity():
    """Test SO(3) log with identity matrix gives zero vector."""
    I = jnp.eye(3)
    log_r = so3.log(I)
    expected = jnp.zeros(3)
    np.testing.assert_allclose(log_r, expected, rtol=1e-6, atol=1e-6)


def test_so3_exp_log_roundtrip():
    """Test SO(3) exp(log(R)) = R roundtrip."""
    # Test with simple rotation around z-axis
    angle = jnp.pi / 4
    axis_angle = jnp.array([0.0, 0.0, angle])

    R = so3.exp(axis_angle)
    log_r = so3.log(R)
    R2 = so3.exp(log_r)

    np.testing.assert_allclose(R, R2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(axis_angle, log_r, rtol=1e-6, atol=1e-6)


def test_so3_multiply():
    """Test SO(3) multiplication."""
    # 90° rotations around z-axis
    R1 = so3.exp(jnp.array([0.0, 0.0, jnp.pi / 2]))
    R2 = so3.exp(jnp.array([0.0, 0.0, jnp.pi / 2]))

    # Should give 180° rotation
    R_combined = so3.multiply(R1, R2)
    expected = so3.exp(jnp.array([0.0, 0.0, jnp.pi]))

    np.testing.assert_allclose(R_combined, expected, rtol=1e-6, atol=1e-6)


def test_so3_inverse():
    """Test SO(3) inverse."""
    axis_angle = jnp.array([0.1, 0.2, 0.3])
    R = so3.exp(axis_angle)
    R_inv = so3.inverse(R)

    # R * R_inv should be identity
    I = so3.multiply(R, R_inv)
    expected = jnp.eye(3)
    np.testing.assert_allclose(I, expected, rtol=1e-6, atol=1e-6)


def test_so3_apply():
    """Test SO(3) apply function."""
    # 90° rotation around z-axis
    R = so3.exp(jnp.array([0.0, 0.0, jnp.pi / 2]))

    # Apply to x-axis vector
    v = jnp.array([1.0, 0.0, 0.0])
    v_rotated = so3.apply(R, v)

    # Should become y-axis vector
    expected = jnp.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(v_rotated, expected, rtol=1e-6, atol=1e-6)


def test_so3_skew_symmetric():
    """Test skew-symmetric matrix function."""
    v = jnp.array([1.0, 2.0, 3.0])
    K = so3.skew_symmetric(v)

    expected = jnp.array([[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])
    np.testing.assert_allclose(K, expected, rtol=1e-6, atol=1e-6)

    # Should be skew-symmetric
    np.testing.assert_allclose(K, -K.T, rtol=1e-6, atol=1e-6)


def test_so3_batch_operations():
    """Test SO(3) operations work with batched inputs."""
    batch_size = 5
    # Use smaller angles to avoid numerical issues near π
    axis_angles = jax.random.uniform(jax.random.PRNGKey(42), (batch_size, 3), minval=-1.0, maxval=1.0)

    # Test exp/log roundtrip for batch
    R_batch = so3.exp(axis_angles)
    log_r_batch = so3.log(R_batch)

    assert R_batch.shape == (batch_size, 3, 3)
    assert log_r_batch.shape == (batch_size, 3)

    # For small angles, should be close to original
    np.testing.assert_allclose(axis_angles, log_r_batch, rtol=1e-5, atol=1e-5)


def test_so3_jit_compatibility():
    """Test SO(3) functions are JIT compatible."""

    @jax.jit
    def jitted_exp(axis_angle):
        return so3.exp(axis_angle)

    @jax.jit
    def jitted_log(R):
        return so3.log(R)

    axis_angle = jnp.array([0.1, 0.2, 0.3])
    R = jitted_exp(axis_angle)
    log_r = jitted_log(R)

    np.testing.assert_allclose(axis_angle, log_r, rtol=1e-6, atol=1e-6)


# SE(3) Lie Group Tests
def test_se3_from_position_and_rotation():
    """Test SE(3) construction from position and rotation."""
    p = jnp.array([1.0, 2.0, 3.0])
    R = jnp.eye(3)

    T = se3.from_position_and_rotation(p, R)

    expected = jnp.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose(T, expected, rtol=1e-6, atol=1e-6)


def test_se3_exp_identity():
    """Test SE(3) exp with zero twist gives identity."""
    zero_twist = jnp.zeros(6)
    T = se3.exp(zero_twist)
    expected = jnp.eye(4)
    np.testing.assert_allclose(T, expected, rtol=1e-6, atol=1e-6)


def test_se3_log_identity():
    """Test SE(3) log with identity matrix gives zero twist."""
    I = jnp.eye(4)
    twist = se3.log(I)
    expected = jnp.zeros(6)
    np.testing.assert_allclose(twist, expected, rtol=1e-6, atol=1e-6)


def test_se3_exp_log_roundtrip():
    """Test SE(3) exp(log(T)) = T roundtrip."""
    # Create a simple twist
    twist = jnp.array([0.1, 0.2, 0.3, 0.05, 0.1, 0.15])

    T = se3.exp(twist)
    log_twist = se3.log(T)
    T2 = se3.exp(log_twist)

    np.testing.assert_allclose(T, T2, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(twist, log_twist, rtol=1e-4, atol=1e-4)


def test_se3_multiply():
    """Test SE(3) multiplication."""
    # Translation by [1, 0, 0]
    T1 = se3.exp(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    # Translation by [0, 1, 0]
    T2 = se3.exp(jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))

    # Combined should be translation by [1, 1, 0]
    T_combined = se3.multiply(T1, T2)
    expected_pos = jnp.array([1.0, 1.0, 0.0])

    actual_pos = se3.get_position(T_combined)
    np.testing.assert_allclose(actual_pos, expected_pos, rtol=1e-6, atol=1e-6)


def test_se3_inverse():
    """Test SE(3) inverse."""
    twist = jnp.array([0.1, 0.2, 0.3, 0.05, 0.1, 0.15])
    T = se3.exp(twist)
    T_inv = se3.inverse(T)

    # T * T_inv should be identity
    I = se3.multiply(T, T_inv)
    expected = jnp.eye(4)
    np.testing.assert_allclose(I, expected, rtol=1e-6, atol=1e-6)


def test_se3_apply():
    """Test SE(3) apply function."""
    # Pure translation
    T = se3.exp(jnp.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))

    # Apply to origin
    point = jnp.array([0.0, 0.0, 0.0])
    transformed = se3.apply(T, point)

    expected = jnp.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(transformed, expected, rtol=1e-6, atol=1e-6)


def test_se3_apply_multiple_points():
    """Test SE(3) apply function with multiple points."""
    # Pure translation
    T = se3.exp(jnp.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))

    # Multiple points
    points = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    transformed = se3.apply(T, points)
    expected = points + jnp.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(transformed, expected, rtol=1e-6, atol=1e-6)


def test_se3_get_position_rotation():
    """Test SE(3) position and rotation extraction."""
    p = jnp.array([1.0, 2.0, 3.0])
    R = so3.exp(jnp.array([0.1, 0.2, 0.3]))

    T = se3.from_position_and_rotation(p, R)

    extracted_p = se3.get_position(T)
    extracted_R = se3.get_rotation(T)

    np.testing.assert_allclose(extracted_p, p, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(extracted_R, R, rtol=1e-6, atol=1e-6)


def test_se3_adjoint():
    """Test SE(3) adjoint computation."""
    twist = jnp.array([0.1, 0.2, 0.3, 0.05, 0.1, 0.15])
    T = se3.exp(twist)

    Ad_T = se3.adjoint(T)

    # Adjoint should be 6x6
    assert Ad_T.shape == (6, 6)

    # Test a simpler adjoint property: verify structure
    # The adjoint should have the right block structure
    R = se3.get_rotation(T)
    t = se3.get_position(T)
    t_skew = so3.skew_symmetric(t)

    # Check top-left block is R
    np.testing.assert_allclose(Ad_T[:3, :3], R, rtol=1e-6, atol=1e-6)

    # Check bottom-right block is R
    np.testing.assert_allclose(Ad_T[3:, 3:], R, rtol=1e-6, atol=1e-6)

    # Check bottom-left is zeros
    np.testing.assert_allclose(Ad_T[3:, :3], jnp.zeros((3, 3)), rtol=1e-6, atol=1e-6)

    # Check top-right is [t]_× @ R
    expected_top_right = jnp.matmul(t_skew, R)
    np.testing.assert_allclose(Ad_T[:3, 3:], expected_top_right, rtol=1e-6, atol=1e-6)


def test_se3_batch_operations():
    """Test SE(3) operations work with batched inputs."""
    batch_size = 3
    # Use smaller angles for better numerical stability
    twists = jax.random.uniform(jax.random.PRNGKey(123), (batch_size, 6), minval=-0.2, maxval=0.2)

    # Test exp/log roundtrip for batch
    T_batch = se3.exp(twists)
    log_twists = se3.log(T_batch)

    assert T_batch.shape == (batch_size, 4, 4)
    assert log_twists.shape == (batch_size, 6)

    np.testing.assert_allclose(twists, log_twists, rtol=1e-4, atol=1e-4)


def test_se3_jit_compatibility():
    """Test SE(3) functions are JIT compatible."""

    @jax.jit
    def jitted_exp(twist):
        return se3.exp(twist)

    @jax.jit
    def jitted_log(T):
        return se3.log(T)

    twist = jnp.array([0.1, 0.2, 0.3, 0.05, 0.1, 0.15])
    T = jitted_exp(twist)
    log_twist = jitted_log(T)

    np.testing.assert_allclose(twist, log_twist, rtol=1e-4, atol=1e-4)


def test_se3_pure_rotation():
    """Test SE(3) with pure rotation (no translation)."""
    # Pure rotation twist (zero linear velocity)
    twist = jnp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3])

    T = se3.exp(twist)

    # Position should be zero
    pos = se3.get_position(T)
    np.testing.assert_allclose(pos, jnp.zeros(3), rtol=1e-6, atol=1e-6)

    # Rotation should match SO(3) exp
    R = se3.get_rotation(T)
    R_expected = so3.exp(jnp.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(R, R_expected, rtol=1e-6, atol=1e-6)


def test_se3_pure_translation():
    """Test SE(3) with pure translation (no rotation)."""
    # Pure translation twist (zero angular velocity)
    twist = jnp.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])

    T = se3.exp(twist)

    # Rotation should be identity
    R = se3.get_rotation(T)
    np.testing.assert_allclose(R, jnp.eye(3), rtol=1e-6, atol=1e-6)

    # Position should match linear velocity
    pos = se3.get_position(T)
    np.testing.assert_allclose(pos, jnp.array([1.0, 2.0, 3.0]), rtol=1e-6, atol=1e-6)
