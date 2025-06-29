"""URDF parser for loading robot models into JAX-native data structures.

This module provides functionality to parse URDF files and convert them
into RobotModel PyTree structures for high-performance computation.
"""

from collections import deque
from typing import Dict

import jax.numpy as jnp
import numpy as np
from lxml import etree

from jax_kinematics.core.robot_model import RobotModel
from jax_kinematics.transforms import se3


def load_urdf(urdf_path: str) -> RobotModel:
    """Load a URDF file and convert it to a RobotModel PyTree.

    Args:
        urdf_path: Path to the URDF file to load.

    Returns:
        RobotModel: A JAX-native robot representation.
    """
    # Parse the URDF XML file
    tree = etree.parse(urdf_path)
    root = tree.getroot()

    # First pass: Build topology mappings
    link_map: Dict[str, int] = {}
    child_to_parent_map: Dict[str, str] = {}
    all_links = {link.get("name") for link in root.findall(".//link")}

    # Collect joints and build parent-child relationships
    joints_info = []
    for joint in root.findall(".//joint"):
        parent_name = joint.find("parent").get("link")
        child_name = joint.find("child").get("link")
        child_to_parent_map[child_name] = parent_name
        joints_info.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": parent_name,
                "child": child_name,
                "elem": joint,
            }
        )

    # Find root link (not a child of any joint)
    root_link = list(all_links - set(child_to_parent_map.keys()))[0]

    # Order links using breadth-first traversal from root
    ordered_links = []
    queue = deque([root_link])
    visited = {root_link}

    while queue:
        link_name = queue.popleft()
        ordered_links.append(link_name)
        # Find children of current link and sort for deterministic order
        children = [j["child"] for j in joints_info if j["parent"] == link_name]
        for child in sorted(children):
            if child not in visited:
                visited.add(child)
                queue.append(child)

    # Create link mapping based on traversal order
    link_map = {name: i for i, name in enumerate(ordered_links)}

    # Create joint mapping for actuated joints only (sorted for deterministic order)
    actuated_joint_names = tuple(sorted([j["name"] for j in joints_info if j["type"] != "fixed"]))

    # Second pass: Populate data arrays
    parent_indices_list = []
    joint_transforms_list = []
    joint_axes_list = []
    actuated_joint_to_link_idx_list = []

    # Create joint lookup by child link for efficient access
    joint_by_child = {j["child"]: j for j in joints_info}

    for i, link_name in enumerate(ordered_links):
        if link_name == root_link:
            parent_indices_list.append(i)  # Root parents itself
            joint_transforms_list.append(jnp.eye(4))
            joint_axes_list.append(jnp.zeros(6))
        else:
            parent_name = child_to_parent_map[link_name]
            parent_indices_list.append(link_map[parent_name])

            joint_info = joint_by_child[link_name]
            joint_elem, joint_type = joint_info["elem"], joint_info["type"]

            # Parse origin transform
            origin = joint_elem.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            R = _rpy_to_rotation_matrix(rpy)
            joint_transforms_list.append(se3.from_position_and_rotation(jnp.array(xyz), jnp.array(R)))

            # Parse joint axis with normalization
            axis_elem = joint_elem.find("axis")
            axis_xyz = (
                np.fromstring(axis_elem.get("xyz", "1 0 0"), sep=" ")
                if axis_elem is not None
                else np.array([1.0, 0.0, 0.0])
            )
            axis_xyz_norm = axis_xyz / (np.linalg.norm(axis_xyz) + 1e-8)

            if joint_type in ["revolute", "continuous"]:
                joint_axes_list.append(jnp.concatenate([jnp.zeros(3), jnp.array(axis_xyz_norm)]))
            elif joint_type == "prismatic":
                joint_axes_list.append(jnp.concatenate([jnp.array(axis_xyz_norm), jnp.zeros(3)]))
            else:  # fixed, floating, etc.
                joint_axes_list.append(jnp.zeros(6))

    # Create the actuated_joint_to_link_idx map
    for joint_name in actuated_joint_names:
        # Find which link is the child of this joint
        child_link_name = [j["child"] for j in joints_info if j["name"] == joint_name][0]
        # Get the index of that link
        link_idx = link_map[child_link_name]
        actuated_joint_to_link_idx_list.append(link_idx)

    return RobotModel(
        link_names=tuple(ordered_links),
        joint_names=actuated_joint_names,
        parent_indices=jnp.array(parent_indices_list, dtype=jnp.int32),
        joint_transforms=jnp.stack(joint_transforms_list),
        joint_axes=jnp.stack(joint_axes_list),
        actuated_joint_to_link_idx=jnp.array(actuated_joint_to_link_idx_list, dtype=jnp.int32),
    )


def _rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw angles to rotation matrix (ZYX extrinsic).

    Args:
        rpy: Array of [roll, pitch, yaw] angles in radians.

    Returns:
        3x3 rotation matrix.
    """
    roll, pitch, yaw = rpy

    # Individual rotation matrices
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    # Combined rotation: R = R_z * R_y * R_x
    return Rz @ Ry @ Rx
