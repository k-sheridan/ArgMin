#pragma once

#include <Eigen/Core>

namespace Tangent {

/**
 * @brief Base class defining the interface for optimizable variables.
 *
 * Optimizable variables can be updated with a minimal-dimension perturbation vector
 * while internally storing a different representation.
 *
 * Derived classes must provide:
 * - A static `dimension` constant specifying the perturbation dimension
 * - An `update(delta)` method that applies the perturbation on-manifold
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
