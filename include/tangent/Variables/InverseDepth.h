#pragma once

#include <limits>

#include "tangent/Differentiation/Jet.h"
#include "tangent/Variables/OptimizableVariable.h"

namespace Tangent {

/**
 * Stores the inverse depth of an observed point.
 *
 * The inverse depth is always a member of (0, infinity].
 */
class InverseDepth : public OptimizableVariableBase<double, 1> {
 public:
  double value;

  InverseDepth() = default;

  InverseDepth(double dinv) : value(dinv) {}

  void update(const Eigen::Matrix<double, 1, 1> &dx) {
    // The inverse depth must come in as a valid
    assert(value >= 0 && value <= std::numeric_limits<double>::max());
    value += dx(0, 0);

    if (value < std::numeric_limits<double>::min()) {
      value = std::numeric_limits<double>::min();
    }
  }

  void ensureUpdateIsRevertible(Eigen::Matrix<double, 1, 1> &dx) {
    if (value + dx(0) < 0) {
      dx(0) = std::numeric_limits<double>::min() - value;
    }
  }
};

// ============================================================================
// Autodiff Support
// ============================================================================

/// Extract raw value for residual-only computation.
inline double getValue(const InverseDepth &dinv) { return dinv.value; }

/**
 * @brief Lift InverseDepth to Jet space for automatic differentiation.
 */
template <typename T, int N>
Jet<T, N> liftToJet(const InverseDepth &dinv, int offset) {
  return Jet<T, N>(static_cast<T>(dinv.value), offset);
}

}  // namespace Tangent
