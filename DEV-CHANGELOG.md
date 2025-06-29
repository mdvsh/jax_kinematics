# Changelog

## [Unreleased]

## [0.1.0] - 2024-12-29

### Added
- `so3.py` with SO(3) exp/log, multiply, inverse, apply, quaternion conversion
- `se3.py` with SE(3) exp/log, multiply, inverse, apply, adjoint  
- Numerical stability for SE(3) log/exp using Taylor series for small angles
- 31 comprehensive tests, all JIT-compatible
- Pure functional API for transforms

### Removed
- `rotation.py` merged into `so3.py`
- `transform.py` replaced with pure functional approach
- `Transform3d` class replaced with direct matrix operations