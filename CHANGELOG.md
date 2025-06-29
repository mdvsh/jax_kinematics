# Changelog


## [0.1.1] - 2025-06-28

### Added
- `chain.py` with Forward Kinematics and Jacobian computation
  - Works on any `RobotModel`, returns per-link SE(3) poses
  - Fully JIT-compilable (`forward_kinematics_world`, `jacobian`)
- New Jacobian test-suite:
  - Deterministic “zero + non-zero” Panda checks
  - JIT-compatibility test
  - Numerical derivative verification against `jax.jacrev`
  - Property test over random joint configurations (finite-value & variation guard)

### Changed
- **`so3.log()`**: replaced small-angle branch with first-order series  
  for numerically stable gradients at θ ≈ 0
- **`chain.jacobian()`**
  - Removed superfluous transpose; output is now `(6, n_dof)`
  - Added `jnp.nan_to_num` post-processing to zero out singular NaN/Inf entries

### Fixed
- Jacobian NaNs and shape mismatch at Panda zero pose (all tests green)
- Rare overflow in `se3.log()` for cotangent term near θ ≈ 0


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