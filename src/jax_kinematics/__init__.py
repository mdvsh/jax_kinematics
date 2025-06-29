"""
JAX Kinematics: A high-performance kinematics library for robotics.

This library provides mathematically rigorous, JIT-compilable implementations
of spatial transformations and kinematics computations using JAX.
"""

import jax
jax.config.update("jax_enable_x64", True)

# Import core modules
from . import transforms
from . import core
from . import io

__version__ = "0.1.0"
__all__ = ["transforms", "core", "io"]                 
