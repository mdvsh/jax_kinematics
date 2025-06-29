"""I/O utilities for loading robot models from various file formats.

This module provides functions for parsing standard robotics file formats
and converting them to JAX-native data structures.
"""

from .urdf_parser import load_urdf

__all__ = ["load_urdf"]
