# Changelog


## [0.1.0] - 2025-06-28

### Added
- `RobotModel` PyTree dataclass with flattened tree representation using integer indices
- URDF parser with two-pass topology building and SE(3) transform parsing
- Parser tests with multi-robot validation
  - Tested on Panda and KUKA iiwa robots
- `core/` and `io/` module structure

- `so3.py` with SO(3) exp/log, multiply, inverse, apply, quaternion conversion
- `se3.py` with SE(3) exp/log, multiply, inverse, apply, adjoint  
- Numerical stability for SE(3) log/exp using Taylor series for small angles
- 31 comprehensive tests, all JIT-compatible
- Pure functional API for transforms

### Removed
- `rotation.py` merged into `so3.py`
- `transform.py` replaced with pure functional approach
- `Transform3d` class replaced with direct matrix operations