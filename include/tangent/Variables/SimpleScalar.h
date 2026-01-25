#pragma once

#include "tangent/Differentiation/Jet.h"
#include "tangent/Variables/OptimizableVariable.h"

namespace Tangent {

class SimpleScalar : public OptimizableVariableBase<double, 1> {
 public:
  double value;

  SimpleScalar() = default;

  SimpleScalar(double val) : value(val) {}

  void update(const Eigen::Matrix<double, 1, 1> &dx) { value += dx(0, 0); }
};

// ============================================================================
// Autodiff Support
// ============================================================================

/// Extract raw value for residual-only computation.
inline double getValue(const SimpleScalar &scalar) { return scalar.value; }

/**
 * @brief Generic getValue for scalar-like derived types.
 *
 * Enables autodiff for classes that derive from SimpleScalar.
 */
template <typename T>
auto getValue(const T &var)
    -> std::enable_if_t<std::is_base_of_v<SimpleScalar, T> &&
                            !std::is_same_v<T, SimpleScalar>,
                        double> {
  return var.value;
}

/**
 * @brief Lift SimpleScalar to Jet space for automatic differentiation.
 */
template <typename T, int N>
Jet<T, N> liftToJet(const SimpleScalar &scalar, int offset) {
  return Jet<T, N>(static_cast<T>(scalar.value), offset);
}

/**
 * @brief Generic liftToJet for scalar-like derived types.
 *
 * Enables autodiff for classes that derive from SimpleScalar.
 */
template <typename T, int N, typename Variable>
auto liftToJet(const Variable &var, int offset)
    -> std::enable_if_t<std::is_base_of_v<SimpleScalar, Variable> &&
                            !std::is_same_v<Variable, SimpleScalar>,
                        Jet<T, N>> {
  return Jet<T, N>(static_cast<T>(var.value), offset);
}

}  // namespace Tangent
