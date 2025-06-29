"""Core kinematics algorithms: Forward Kinematics and Jacobian computation.
This module implements the heart of the jax_kinematics library - high-performance
forward kinematics and Jacobian calculations using JAX primitives and automatic
differentiation.
"""

from typing import Dict

import jax
import jax.numpy as jnp
from jax import Array

from .core import RobotModel
from .transforms import se3


def forward_kinematics(robot: RobotModel, q: Array) -> Dict[str, Array]:
    """Compute forward kinematics for all links in the robot.

    Args:
        robot: RobotModel containing the robot's kinematic structure
        q: Joint angles array of shape (num_dof,) for actuated joints only

    Returns:
        Dictionary mapping link names to their 4x4 SE(3) world poses
    """
    # Get world transforms for all links
    world_transforms = forward_kinematics_world(robot, q)

    # Return dictionary mapping link names to transforms
    return {name: world_transforms[i] for i, name in enumerate(robot.link_names)}


def forward_kinematics_world(robot: RobotModel, q: Array) -> Array:
    """Internal FK function returning array of world transforms.

    This function is optimized for internal use and automatic differentiation.

    Args:
        robot: RobotModel containing the robot's kinematic structure
        q: Joint angles array of shape (num_dof,) for actuated joints only

    Returns:
        Array of shape (num_links, 4, 4) with world poses for all links
    """
    num_links = len(robot.link_names)

    # Create a full vector of joint values for all links, initialized to zero.
    q_full = jnp.zeros(num_links)

    # Use the new map to correctly and robustly scatter the actuated `q` values.
    # This is JIT-friendly and correct.
    q_full = q_full.at[robot.actuated_joint_to_link_idx].set(q)

    # Initialize world transforms with identity matrices
    world_transforms = jnp.identity(4)[None].repeat(num_links, axis=0)

    def scan_body(carry, i):
        """Processes link `i` using its parent's world pose from `carry`."""
        T_world_to_parent = carry[robot.parent_indices[i]]

        # Transform from parent link to child link
        T_joint_motion = se3.exp(robot.joint_axes[i] * q_full[i])
        T_parent_to_child = robot.joint_transforms[i] @ T_joint_motion

        # This link's world pose
        T_world_to_child = T_world_to_parent @ T_parent_to_child

        # Update the array of transforms
        carry = carry.at[i].set(T_world_to_child)
        return carry, None

    # The `scan` must iterate from 1 to num_links-1 because the root (0) is the base case.
    # The initial carry is correct for the root's children.
    final_transforms, _ = jax.lax.scan(scan_body, world_transforms, jnp.arange(1, num_links))

    return final_transforms


def jacobian(robot: RobotModel, q: Array, link_name: str) -> Array:
    """Compute the 6D spatial Jacobian of a link w.r.t. joint angles.

    Uses JAX automatic differentiation to compute the Jacobian efficiently.

    Args:
        robot: RobotModel containing the robot's kinematic structure
        q: Joint angles array of shape (num_dof,) for actuated joints only
        link_name: Name of the target link

    Returns:
        6x(num_dof) Jacobian matrix relating joint velocities to spatial velocity
    """
    # Find target link index
    try:
        link_idx = robot.link_names.index(link_name)
    except ValueError:
        raise ValueError(f"Link '{link_name}' not found in robot model")

    def get_pose_twist(joint_angles: Array) -> Array:
        """Closure to compute SE(3) pose and convert to se(3) twist."""
        # Get world transforms for all links
        world_transforms = forward_kinematics_world(robot, joint_angles)

        # Extract target link's pose
        T_target = world_transforms[link_idx]

        # Convert SE(3) matrix to se(3) twist vector for proper differentiation
        return se3.log(T_target)

    J = jax.jacrev(get_pose_twist)(q)  # (6, num_dof)
    # Replace NaNs/Infs with 0 so downstream math stays finite
    return jnp.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
