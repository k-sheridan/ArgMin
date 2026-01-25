#pragma once

#include <sophus/so3.hpp>

#include "tangent/Differentiation/Jet.h"
#include "tangent/Differentiation/JetTraits.h"
#include "tangent/Variables/OptimizableVariable.h"

namespace Tangent {

/**
 * Sophus SO3 transformation.
 *
 * The order of the delta vector is [wx,wy,wz].
 */
class SO3 : public OptimizableVariableBase<double, 3> {
 public:
  Sophus::SO3<double> value;

  SO3() = default;

  SO3(const Sophus::SO3<double> &so3) : value(so3) {}

  void update(const Eigen::Matrix<double, 3, 1> &dx) {
    value *= Sophus::SO3d::exp(dx);
  }
};

// ============================================================================
// Autodiff Support
// ============================================================================

/// Extract raw value for residual-only computation.
inline const Sophus::SO3d &getValue(const SO3 &so3) { return so3.value; }

/**
 * @brief Lift SO3 to Jet space for automatic differentiation.
 * Parameterization: omega(3) matching update() convention: R_new = R * exp(dx)
 */
template <typename T, int N>
Sophus::SO3<Jet<T, N>> liftToJet(const SO3 &so3, int offset) {
  using JetT = Jet<T, N>;

  Eigen::Quaternion<JetT> q;
  q.w() = JetT(so3.value.unit_quaternion().w());
  q.x() = JetT(so3.value.unit_quaternion().x());
  q.y() = JetT(so3.value.unit_quaternion().y());
  q.z() = JetT(so3.value.unit_quaternion().z());

  Sophus::SO3<JetT> result(q);

  // Seed perturbation omega(3)
  Eigen::Matrix<JetT, 3, 1> omega;
  for (int i = 0; i < 3; ++i) {
    omega(i) = JetT(T(0), offset + i);
  }

  result = result * Sophus::SO3<JetT>::exp(omega);

  return result;
}

}  // namespace Tangent
