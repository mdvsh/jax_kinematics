"""
JAX-based transforms library for robotics and computer vision.

This module provides mathematically rigorous, JIT-compilable implementations of:
- SO(3) rotations (so3 module) 
- SE(3) rigid body transforms (se3 module)

All functions are pure, stateless, and designed for high-performance computation.
"""

# Core Lie group modules
from . import so3
from . import se3

__all__ = [
    "so3",
    "se3",
]
