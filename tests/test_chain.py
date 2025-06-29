"""Tests for forward kinematics and Jacobian computation."""

import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from jax_kinematics.chain import forward_kinematics, jacobian
from jax_kinematics.io import load_urdf


def test_fk_panda():
    """Test forward kinematics on Panda robot with known configuration."""
    # Load the Panda URDF
    urdf_path = Path(__file__).parent / "fixtures" / "panda_arm.urdf"
    robot = load_urdf(str(urdf_path))

    # Test with zero configuration (all joints at zero)
    q_zero = jnp.zeros(7)
    poses_zero = forward_kinematics(robot, q_zero)

    # Verify we get poses for all links
    assert len(poses_zero) == len(robot.link_names)
    for link_name in robot.link_names:
        assert link_name in poses_zero
        assert poses_zero[link_name].shape == (4, 4)

        # Verify valid SE(3) matrix
        T = poses_zero[link_name]
        # Check bottom row is [0, 0, 0, 1]
        np.testing.assert_allclose(T[3, :], jnp.array([0, 0, 0, 1]), rtol=1e-6, atol=1e-6)
        # Check rotation part is orthogonal
        R = T[:3, :3]
        should_be_identity = jnp.matmul(R, R.T)
        np.testing.assert_allclose(should_be_identity, jnp.eye(3), rtol=1e-4, atol=1e-4)

    # Test with non-zero configuration
    q_nonzero = jnp.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    poses_nonzero = forward_kinematics(robot, q_nonzero)

    # Verify poses are different from zero configuration
    for link_name in robot.link_names:
        assert link_name in poses_nonzero
        T_nonzero = poses_nonzero[link_name]
        T_zero = poses_zero[link_name]

        # At least some links should have different poses
        if link_name != robot.link_names[0]:  # Skip base link
            # Check that transforms are different (not identical)
            diff_norm = jnp.linalg.norm(T_nonzero - T_zero)
            # Allow for some links to be the same if they're not affected by early joints
            # But the end-effector should definitely be different
            if "link7" in link_name or "link6" in link_name:
                assert diff_norm > 1e-6, f"Link {link_name} pose should change with joint motion"


def test_fk_jit_compatibility():
    """Test that forward kinematics is JIT-compilable."""
    # Load the Panda URDF
    urdf_path = Path(__file__).parent / "fixtures" / "panda_arm.urdf"
    robot = load_urdf(str(urdf_path))

    # Create a JIT-compiled version
    @jax.jit
    def jit_fk(q):
        from jax_kinematics.chain import forward_kinematics_world

        return forward_kinematics_world(robot, q)

    # Test with sample configuration
    q = jnp.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])

    # Should work without errors
    world_transforms = jit_fk(q)
    assert world_transforms.shape == (len(robot.link_names), 4, 4)

    # Verify transforms are valid SE(3)
    for i in range(len(robot.link_names)):
        T = world_transforms[i]
        # Check bottom row
        np.testing.assert_allclose(T[3, :], jnp.array([0, 0, 0, 1]), rtol=1e-6, atol=1e-6)
        # Check rotation orthogonality
        R = T[:3, :3]
        should_be_identity = jnp.matmul(R, R.T)
        np.testing.assert_allclose(should_be_identity, jnp.eye(3), rtol=1e-4, atol=1e-4)


def test_jacobian_panda():
    """Test Jacobian computation for Panda end-effector."""
    # Load the Panda URDF
    urdf_path = Path(__file__).parent / "fixtures" / "panda_arm.urdf"
    robot = load_urdf(str(urdf_path))

    # Find end-effector link (highest numbered link)
    end_effector_candidates = [name for name in robot.link_names if "link" in name]
    end_effector_link = max(end_effector_candidates, key=lambda x: int(x.split("link")[-1]))

    # Test Jacobian at zero configuration
    q_zero = jnp.zeros(7)
    J_zero = jacobian(robot, q_zero, end_effector_link)

    # Verify shape
    assert J_zero.shape == (6, 7), f"Expected (6, 7) Jacobian, got {J_zero.shape}"

    # Test Jacobian at non-zero configuration
    q_nonzero = jnp.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    J_nonzero = jacobian(robot, q_nonzero, end_effector_link)

    # Verify shape
    assert J_nonzero.shape == (6, 7)

    # Jacobians should be different at different configurations
    diff_norm = jnp.linalg.norm(J_nonzero - J_zero)
    assert diff_norm > 1e-6, "Jacobian should change with configuration"

    # Test Jacobian for different links
    for link_name in robot.link_names[1:4]:  # Test a few links
        J_link = jacobian(robot, q_zero, link_name)
        assert J_link.shape == (6, 7)


def test_jacobian_jit_compatibility():
    """Test that Jacobian computation is JIT-compilable."""
    # Load the Panda URDF
    urdf_path = Path(__file__).parent / "fixtures" / "panda_arm.urdf"
    robot = load_urdf(str(urdf_path))

    # Find end-effector link
    end_effector_candidates = [name for name in robot.link_names if "link" in name]
    end_effector_link = max(end_effector_candidates, key=lambda x: int(x.split("link")[-1]))

    # Create JIT-compiled Jacobian function
    @jax.jit
    def jit_jacobian(q):
        return jacobian(robot, q, end_effector_link)

    # Test with sample configuration
    q = jnp.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])

    # Should work without errors
    J = jit_jacobian(q)
    assert J.shape == (6, 7)

    # Verify Jacobian entries are reasonable (not all zeros)
    assert jnp.sum(jnp.abs(J)) > 1e-6, "Jacobian should have non-zero entries"


def test_jacobian_numerical_verification():
    """Verify Jacobian using numerical differentiation."""
    # Load the Panda URDF
    urdf_path = Path(__file__).parent / "fixtures" / "panda_arm.urdf"
    robot = load_urdf(str(urdf_path))

    # Find end-effector link
    end_effector_candidates = [name for name in robot.link_names if "link" in name]
    end_effector_link = max(end_effector_candidates, key=lambda x: int(x.split("link")[-1]))

    # Test configuration
    q = jnp.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])

    # Compute analytical Jacobian
    J_analytical = jacobian(robot, q, end_effector_link)

    # Compute numerical Jacobian
    def get_pose_log(q_test):
        from jax_kinematics.transforms import se3

        poses = forward_kinematics(robot, q_test)
        return se3.log(poses[end_effector_link])

    J_numerical = jax.jacrev(get_pose_log)(q)

    # They should be very close
    np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-10, atol=1e-10)


def test_jacobian_random_configs():
    """Property test: Jacobian is finite and varies across random configurations."""
    # Load the Panda URDF
    urdf_path = Path(__file__).parent / "fixtures" / "panda_arm.urdf"
    robot = load_urdf(str(urdf_path))

    # Find end-effector link
    end_effector_candidates = [name for name in robot.link_names if "link" in name]
    end_effector_link = max(end_effector_candidates, key=lambda x: int(x.split("link")[-1]))

    # Generate random configurations
    key = jrandom.PRNGKey(0)  # Deterministic for CI/caching
    n_samples = 10
    q_samples = jrandom.uniform(
        key,
        shape=(n_samples, len(robot.joint_names)),
        minval=-jnp.pi,
        maxval=jnp.pi,
    )

    # Compute Jacobians for all configurations
    Js = [jacobian(robot, q_samples[i], end_effector_link) for i in range(n_samples)]

    # Verify all Jacobians are finite
    for i, J in enumerate(Js):
        assert jnp.isfinite(J).all(), f"Jacobian {i} contains NaN/Inf"

    # Verify correct shape
    for J in Js:
        assert J.shape == (6, len(robot.joint_names))

    # Verify Jacobians vary across different configurations
    tol = 1e-6
    varied = False
    for J_a, J_b in itertools.combinations(Js, 2):
        if jnp.linalg.norm(J_a - J_b) > tol:
            varied = True
            break

    assert varied, "Jacobian did not change across random configurations"


def test_invalid_link_name():
    """Test error handling for invalid link names."""
    # Load the Panda URDF
    urdf_path = Path(__file__).parent / "fixtures" / "panda_arm.urdf"
    robot = load_urdf(str(urdf_path))

    q = jnp.zeros(7)

    # Should raise ValueError for non-existent link
    with pytest.raises(ValueError, match="Link 'nonexistent_link' not found"):
        jacobian(robot, q, "nonexistent_link")
