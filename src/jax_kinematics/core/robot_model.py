"""RobotModel PyTree data structure for JAX-native robot representation.

This module defines the core data structure for representing robots in a 
stateless, immutable format that is fully compatible with JAX transformations.
"""

import jax
from jax import Array
from flax import struct
from typing import Tuple


@struct.dataclass
class RobotModel:
    """Immutable PyTree representation of a robot's kinematic structure.
    
    This dataclass represents a robot as a flattened tree structure using
    integer indices for parent-child relationships. All kinematic data is
    stored in JAX arrays for high-performance computation.
    
    Attributes:
        link_names: Tuple of all link names. Index corresponds to link ID.
                    Marked as a static field for JIT compilation.
        joint_names: Tuple of all actuated (non-fixed) joint names.
                     Marked as a static field for JIT compilation.
        parent_indices: Array of shape (num_links,) where parent_indices[i] 
                       is the parent link index of link i. Root link parents itself.
        joint_transforms: Array of shape (num_links, 4, 4) containing SE(3) 
                         transformations from each link to its parent.
        joint_axes: Array of shape (num_links, 6) containing 6D se(3) twist 
                   vectors for each joint. [vx,vy,vz,wx,wy,wz] format.
    """
    link_names: Tuple[str, ...] = struct.field(pytree_node=False)
    joint_names: Tuple[str, ...] = struct.field(pytree_node=False)
    parent_indices: Array
    joint_transforms: Array
    joint_axes: Array