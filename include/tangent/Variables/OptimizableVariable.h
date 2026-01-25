#pragma once

#include <Eigen/Core>

namespace Tangent {

/**
 * @brief Base class for optimizable variables.
 *
 * Variables store state that can be updated via a minimal-dimension perturbation.
 * This enables manifold optimization where internal representation differs from
 * the tangent space (e.g., quaternion vs axis-angle for rotations).
 *
 * ## Required Interface
 *
 * Derived classes must provide:
 * - `static const size_t dimension` - Tangent space dimension
 * - `void update(const Eigen::Matrix<scalar_type, dimension, 1>& dx)` - Apply
 *   perturbation
 *
 * The `update()` convention:
 * - Euclidean spaces: `value += dx`
 * - Lie groups (SE3, SO3): `value = value * exp(dx)` (right multiplication)
 *
 * ## Autodiff Support (Optional)
 *
 * To enable automatic differentiation with AutoDiffErrorTerm, implement these
 * free functions in the Tangent namespace alongside your variable:
 *
 * - `liftToJet<T, N>(const Variable& var, int offset)` - Lift to Jet space
 *   with seeded derivatives starting at the given offset index.
 * - `getValue(const Variable& var)` - Extract raw value for residual-only path.
 *
 * See SE3.h for an example implementation.
 *
 * @tparam ScalarType The floating point type (typically float or double)
 * @tparam Dimension The dimension of the minimal perturbation vector
 */
template <typename ScalarType, size_t Dimension>
class OptimizableVariable {
 public:
  typedef ScalarType scalar_type;
  /// The dimension of the minimal perturbation vector.
  static const size_t dimension = Dimension;

  /// This function will modify a perturbation to ensure that
  /// exp(dx)^(-1) = exp(-dx)
  void ensureUpdateIsRevertible(Eigen::Matrix<double, Dimension, 1> &dx) {}
};

}  // namespace Tangent
