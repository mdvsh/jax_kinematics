"""URDF parser for loading robot models into JAX-native data structures.

This module provides functionality to parse URDF files and convert them
into RobotModel PyTree structures for high-performance computation.
"""

import jax.numpy as jnp
from lxml import etree
from typing import Dict, List, Tuple
import numpy as np
from collections import deque

from jax_kinematics.core.robot_model import RobotModel
from jax_kinematics.transforms import se3, so3


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
    joint_map: Dict[str, int] = {}
    child_to_parent_map: Dict[str, str] = {}
    all_links = set()
    child_links = set()
    
    # Collect all links
    for link in root.findall('.//link'):
        link_name = link.get('name')
        all_links.add(link_name)
    
    # Collect joints and build parent-child relationships
    joints_info = []
    for joint in root.findall('.//joint'):
        joint_name = joint.get('name')
        joint_type = joint.get('type')
        
        parent_elem = joint.find('parent')
        child_elem = joint.find('child')
        
        if parent_elem is not None and child_elem is not None:
            parent_name = parent_elem.get('link')
            child_name = child_elem.get('link')
            
            child_to_parent_map[child_name] = parent_name
            child_links.add(child_name)
            
            joints_info.append({
                'name': joint_name,
                'type': joint_type,
                'parent': parent_name,
                'child': child_name,
                'joint_elem': joint
            })
    
    # Find root link (not a child of any joint)
    root_links = all_links - child_links
    if len(root_links) != 1:
        raise ValueError(f"Expected exactly one root link, found: {root_links}")
    root_link = list(root_links)[0]
    
    # Order links using breadth-first traversal from root
    ordered_links = []
    queue = deque([root_link])
    visited = set()
    
    while queue:
        current_link = queue.popleft()
        if current_link in visited:
            continue
            
        visited.add(current_link)
        ordered_links.append(current_link)
        
        # Find children of current link
        for joint_info in joints_info:
            if joint_info['parent'] == current_link:
                child = joint_info['child']
                if child not in visited:
                    queue.append(child)
    
    # Create link mapping based on traversal order
    for i, link_name in enumerate(ordered_links):
        link_map[link_name] = i
    
    # Create joint mapping for actuated joints only
    actuated_joint_names = []
    for joint_info in joints_info:
        if joint_info['type'] not in ['fixed']:
            actuated_joint_names.append(joint_info['name'])
    
    for i, joint_name in enumerate(actuated_joint_names):
        joint_map[joint_name] = i
    
    # Second pass: Populate data arrays
    num_links = len(ordered_links)
    parent_indices_list = []
    joint_transforms_list = []
    joint_axes_list = []
    
    # Create joint lookup by child link
    joint_by_child = {}
    for joint_info in joints_info:
        joint_by_child[joint_info['child']] = joint_info
    
    for i, link_name in enumerate(ordered_links):
        # Parent index
        if link_name == root_link:
            parent_indices_list.append(i)  # Root parents itself
        else:
            parent_name = child_to_parent_map[link_name]
            parent_idx = link_map[parent_name]
            parent_indices_list.append(parent_idx)
        
        # Joint transform and axis
        if link_name in joint_by_child:
            joint_info = joint_by_child[link_name]
            joint_elem = joint_info['joint_elem']
            joint_type = joint_info['type']
            
            # Parse origin transform
            origin_elem = joint_elem.find('origin')
            if origin_elem is not None:
                xyz_str = origin_elem.get('xyz', '0 0 0')
                rpy_str = origin_elem.get('rpy', '0 0 0')
                
                xyz = np.array([float(x) for x in xyz_str.split()])
                rpy = np.array([float(x) for x in rpy_str.split()])
                
                # Convert RPY to rotation matrix
                R = _rpy_to_rotation_matrix(rpy)
                transform = se3.from_position_and_rotation(jnp.array(xyz), jnp.array(R))
            else:
                transform = jnp.eye(4)
                
            joint_transforms_list.append(transform)
            
            # Parse joint axis
            if joint_type == 'fixed':
                axis = jnp.zeros(6)
            else:
                axis_elem = joint_elem.find('axis')
                if axis_elem is not None:
                    axis_xyz_str = axis_elem.get('xyz', '0 0 1')
                    axis_xyz = jnp.array([float(x) for x in axis_xyz_str.split()])
                else:
                    axis_xyz = jnp.array([0.0, 0.0, 1.0])  # Default Z axis
                
                if joint_type == 'revolute' or joint_type == 'continuous':
                    # Revolute: [0, 0, 0, wx, wy, wz]
                    axis = jnp.concatenate([jnp.zeros(3), axis_xyz])
                elif joint_type == 'prismatic':
                    # Prismatic: [vx, vy, vz, 0, 0, 0]
                    axis = jnp.concatenate([axis_xyz, jnp.zeros(3)])
                else:
                    axis = jnp.zeros(6)
                    
            joint_axes_list.append(axis)
        else:
            # Root link has identity transform and zero axis
            joint_transforms_list.append(jnp.eye(4))
            joint_axes_list.append(jnp.zeros(6))
    
    # Convert lists to JAX arrays
    parent_indices = jnp.array(parent_indices_list, dtype=jnp.int32)
    joint_transforms = jnp.stack(joint_transforms_list)
    joint_axes = jnp.stack(joint_axes_list)
    
    return RobotModel(
        link_names=tuple(ordered_links),
        joint_names=tuple(actuated_joint_names),
        parent_indices=parent_indices,
        joint_transforms=joint_transforms,
        joint_axes=joint_axes
    )


def _rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw angles to rotation matrix.
    
    Args:
        rpy: Array of [roll, pitch, yaw] angles in radians.
        
    Returns:
        3x3 rotation matrix.
    """
    roll, pitch, yaw = rpy
    
    # Individual rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: R = R_z * R_y * R_x
    return R_z @ R_y @ R_x