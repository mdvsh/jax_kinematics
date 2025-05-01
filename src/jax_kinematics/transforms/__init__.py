from importlib import import_module

from .rotation import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    normalize_quaternions,
)

# re-export the sub-module itself
transform = import_module(".transform", package=__name__)
Transform3d = transform.Transform3d

__all__ = [
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "normalize_quaternions",
    "transform",
    "Transform3d",
]
