"""Tests for URDF parser functionality."""

import pytest
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from jax_kinematics.io import load_urdf
from jax_kinematics.core import RobotModel


def test_load_panda_urdf():
    """Test loading Panda URDF and verify RobotModel structure."""
    # Load the Panda URDF
    urdf_path = Path(__file__).parent / "fixtures" / "panda_arm.urdf"
    robot = load_urdf(str(urdf_path))
    
    # Verify it's a RobotModel instance
    assert isinstance(robot, RobotModel)
    
    # Verify number of links (9 links: panda_link0 through panda_link8)
    assert len(robot.link_names) == 9
    expected_links = [
        "panda_link0", "panda_link1", "panda_link2", "panda_link3",
        "panda_link4", "panda_link5", "panda_link6", "panda_link7", "panda_link8"
    ]
    # Check all 9 links are present (link8 might be at different position due to traversal)
    for i in range(9):
        assert f"panda_link{i}" in robot.link_names
    
    # Verify number of actuated joints (7 revolute joints, not counting the fixed joint8)
    assert len(robot.joint_names) == 7
    expected_joints = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7"
    ]
    for joint in expected_joints:
        assert joint in robot.joint_names
    
    # Verify array shapes
    num_links = len(robot.link_names)
    assert robot.parent_indices.shape == (num_links,)
    assert robot.joint_transforms.shape == (num_links, 4, 4)
    assert robot.joint_axes.shape == (num_links, 6)
    
    # Verify root link parents itself (should be index 0)
    assert robot.parent_indices[0] == 0
    
    # Verify parent-child relationships make sense
    # All parent indices should be valid (< num_links)
    assert jnp.all(robot.parent_indices < num_links)
    assert jnp.all(robot.parent_indices >= 0)
    
    # Verify joint transforms are valid SE(3) matrices
    for i in range(num_links):
        T = robot.joint_transforms[i]
        # Check bottom row is [0, 0, 0, 1]
        np.testing.assert_allclose(T[3, :], jnp.array([0, 0, 0, 1]), rtol=1e-6, atol=1e-6)
        # Check rotation part is approximately orthogonal
        R = T[:3, :3]
        should_be_identity = jnp.matmul(R, R.T)
        np.testing.assert_allclose(should_be_identity, jnp.eye(3), rtol=1e-5, atol=1e-5)
    
    # Verify joint axes are reasonable
    # Should have some non-zero axes for revolute joints
    num_nonzero_axes = jnp.sum(jnp.linalg.norm(robot.joint_axes, axis=1) > 1e-6)
    assert num_nonzero_axes >= 7  # At least 7 revolute joints should have non-zero axes
    
    # Check that revolute joint axes are in the angular part (last 3 components)
    # and have unit length in the angular part
    for i in range(num_links):
        axis = robot.joint_axes[i]
        angular_part = axis[3:]
        angular_norm = jnp.linalg.norm(angular_part)
        
        if angular_norm > 1e-6:  # This is a revolute joint
            # Linear part should be zero for pure revolute joints
            linear_part = axis[:3]
            linear_norm = jnp.linalg.norm(linear_part)
            assert linear_norm < 1e-6, f"Link {i} has non-zero linear part: {linear_part}"
            
            # Angular part should be approximately unit length
            np.testing.assert_allclose(angular_norm, 1.0, rtol=1e-5, atol=1e-5)


def test_robot_model_is_pytree():
    """Test that RobotModel is a valid JAX PyTree."""
    import jax
    
    # Load the Panda URDF
    urdf_path = Path(__file__).parent / "fixtures" / "panda_arm.urdf"
    robot = load_urdf(str(urdf_path))
    
    # Test that it can be used with JAX tree operations
    flat_robot, tree_def = jax.tree_util.tree_flatten(robot)
    reconstructed_robot = jax.tree_util.tree_unflatten(tree_def, flat_robot)
    
    # Verify the reconstruction preserves the data
    assert reconstructed_robot.link_names == robot.link_names
    assert reconstructed_robot.joint_names == robot.joint_names
    np.testing.assert_array_equal(reconstructed_robot.parent_indices, robot.parent_indices)
    np.testing.assert_array_equal(reconstructed_robot.joint_transforms, robot.joint_transforms)
    np.testing.assert_array_equal(reconstructed_robot.joint_axes, robot.joint_axes)


def test_robot_model_jit_compatibility():
    """Test that RobotModel can be used in JIT-compiled functions."""
    import jax
    
    # Load the Panda URDF
    urdf_path = Path(__file__).parent / "fixtures" / "panda_arm.urdf"
    robot = load_urdf(str(urdf_path))
    
    def get_num_links(robot_model):
        return robot_model.parent_indices.shape[0]
    
    def get_first_transform(robot_model):
        return robot_model.joint_transforms[0]
    
    # JIT compile functions that only use array fields
    jit_get_num_links = jax.jit(get_num_links)
    jit_get_first_transform = jax.jit(get_first_transform)
    
    # These should work without errors
    num_links = jit_get_num_links(robot)
    first_transform = jit_get_first_transform(robot)
    
    assert num_links == len(robot.link_names)
    assert first_transform.shape == (4, 4)


def test_panda_specific_structure():
    """Test Panda-specific structural properties."""
    urdf_path = Path(__file__).parent / "fixtures" / "panda_arm.urdf"
    robot = load_urdf(str(urdf_path))
    
    # Panda should have panda_link0 as root
    assert robot.link_names[0] == "panda_link0"
    
    # All joints should be revolute (no prismatic joints in Panda)
    for i in range(len(robot.link_names)):
        axis = robot.joint_axes[i]
        linear_part = axis[:3]
        angular_part = axis[3:]
        
        # Either it's a zero axis (fixed joint) or a pure rotational axis
        if jnp.linalg.norm(axis) > 1e-6:
            # Should be pure rotation (no translation component)
            assert jnp.linalg.norm(linear_part) < 1e-6
            assert jnp.linalg.norm(angular_part) > 1e-6


def test_load_kuka_iiwa_urdf():
    """Test loading KUKA iiwa URDF and verify RobotModel structure."""
    # Load the KUKA iiwa URDF
    urdf_path = Path(__file__).parent / "fixtures" / "kuka_iiwa.urdf"
    robot = load_urdf(str(urdf_path))
    
    # Verify it's a RobotModel instance
    assert isinstance(robot, RobotModel)
    
    # Verify number of links (8 links: lbr_iiwa_link_0 through lbr_iiwa_link_7)
    assert len(robot.link_names) == 8
    expected_links = [
        "lbr_iiwa_link_0", "lbr_iiwa_link_1", "lbr_iiwa_link_2", "lbr_iiwa_link_3",
        "lbr_iiwa_link_4", "lbr_iiwa_link_5", "lbr_iiwa_link_6", "lbr_iiwa_link_7"
    ]
    # Check all 8 links are present
    for i in range(8):
        assert f"lbr_iiwa_link_{i}" in robot.link_names
    
    # Verify number of actuated joints (7 revolute joints)
    assert len(robot.joint_names) == 7
    expected_joints = [
        "lbr_iiwa_joint_1", "lbr_iiwa_joint_2", "lbr_iiwa_joint_3", "lbr_iiwa_joint_4",
        "lbr_iiwa_joint_5", "lbr_iiwa_joint_6", "lbr_iiwa_joint_7"
    ]
    for joint in expected_joints:
        assert joint in robot.joint_names
    
    # Verify array shapes
    num_links = len(robot.link_names)
    assert robot.parent_indices.shape == (num_links,)
    assert robot.joint_transforms.shape == (num_links, 4, 4)
    assert robot.joint_axes.shape == (num_links, 6)
    
    # Verify root link parents itself (should be index 0)
    assert robot.parent_indices[0] == 0
    
    # Verify parent-child relationships make sense
    assert jnp.all(robot.parent_indices < num_links)
    assert jnp.all(robot.parent_indices >= 0)
    
    # Verify joint transforms are valid SE(3) matrices
    for i in range(num_links):
        T = robot.joint_transforms[i]
        # Check bottom row is [0, 0, 0, 1]
        np.testing.assert_allclose(T[3, :], jnp.array([0, 0, 0, 1]), rtol=1e-6, atol=1e-6)
        # Check rotation part is approximately orthogonal
        R = T[:3, :3]
        should_be_identity = jnp.matmul(R, R.T)
        np.testing.assert_allclose(should_be_identity, jnp.eye(3), rtol=1e-5, atol=1e-5)
    
    # Verify joint axes are reasonable
    # Should have exactly 7 non-zero axes for revolute joints
    num_nonzero_axes = jnp.sum(jnp.linalg.norm(robot.joint_axes, axis=1) > 1e-6)
    assert num_nonzero_axes == 7  # Exactly 7 revolute joints should have non-zero axes


def test_kuka_iiwa_specific_structure():
    """Test KUKA iiwa-specific structural properties."""
    urdf_path = Path(__file__).parent / "fixtures" / "kuka_iiwa.urdf"
    robot = load_urdf(str(urdf_path))
    
    # KUKA iiwa should have lbr_iiwa_link_0 as root
    assert robot.link_names[0] == "lbr_iiwa_link_0"
    
    # All joints should be revolute (no prismatic joints in KUKA iiwa)
    revolute_joint_count = 0
    for i in range(len(robot.link_names)):
        axis = robot.joint_axes[i]
        linear_part = axis[:3]
        angular_part = axis[3:]
        
        # Either it's a zero axis (no joint) or a pure rotational axis
        if jnp.linalg.norm(axis) > 1e-6:
            revolute_joint_count += 1
            # Should be pure rotation (no translation component)
            assert jnp.linalg.norm(linear_part) < 1e-6
            assert jnp.linalg.norm(angular_part) > 1e-6
            # Angular part should be approximately unit length
            np.testing.assert_allclose(jnp.linalg.norm(angular_part), 1.0, rtol=1e-5, atol=1e-5)
    
    # Should have exactly 7 revolute joints
    assert revolute_joint_count == 7


def test_multi_robot_compatibility():
    """Test that both robot models can be loaded and used together."""
    # Load both robots
    panda_path = Path(__file__).parent / "fixtures" / "panda_arm.urdf"
    kuka_path = Path(__file__).parent / "fixtures" / "kuka_iiwa.urdf"
    
    panda_robot = load_urdf(str(panda_path))
    kuka_robot = load_urdf(str(kuka_path))
    
    # Both should be RobotModel instances
    assert isinstance(panda_robot, RobotModel)
    assert isinstance(kuka_robot, RobotModel)
    
    # They should have different structures
    assert len(panda_robot.link_names) == 9  # Panda has 9 links
    assert len(kuka_robot.link_names) == 8   # KUKA has 8 links
    
    assert len(panda_robot.joint_names) == 7  # Both have 7 DOF
    assert len(kuka_robot.joint_names) == 7
    
    # Different naming conventions
    assert "panda" in panda_robot.link_names[0]
    assert "lbr_iiwa" in kuka_robot.link_names[0]
    
    # Both should work with JAX operations
    import jax
    
    def get_joint_count(robot_model):
        return robot_model.parent_indices.shape[0]
    
    jit_get_joint_count = jax.jit(get_joint_count)
    
    panda_count = jit_get_joint_count(panda_robot)
    kuka_count = jit_get_joint_count(kuka_robot)
    
    assert panda_count == 9
    assert kuka_count == 8