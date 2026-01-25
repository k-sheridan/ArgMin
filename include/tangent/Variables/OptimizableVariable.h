#pragma once

#include <Eigen/Core>

namespace Tangent {

/**
 * @brief Base class for optimizable variables.
 *
 * Derive from this class to create a new variable type. Derived classes must:
 * - Call `update(dx)` to apply a perturbation vector
 * - Optionally override `ensureUpdateIsRevertible(dx)` if the variable has
 *   constraints (see InverseDepth.h for an example)
 *
 * Update convention:
 * - Euclidean: `value += dx`
 * - Lie groups: `value = value * exp(dx)`
 *
 * For autodiff support, also implement `getValue()` and `liftToJet()` free
 * functions. See SE3.h for an example.
 *
 * @tparam ScalarType Floating point type (float or double)
 * @tparam Dimension Tangent space dimension (size of dx vector)
 */
template <typename ScalarType, size_t Dimension>
class OptimizableVariableBase {
 public:
  typedef ScalarType scalar_type;
  static const size_t dimension = Dimension;

  /// Clamps dx so that update(-dx) correctly reverses update(dx).
  /// Override this if your variable has constraints (e.g., must stay positive).
  /// Called by the optimizer before applying updates.
  void ensureUpdateIsRevertible(Eigen::Matrix<double, Dimension, 1> &dx) {}
};

}  // namespace Tangent
