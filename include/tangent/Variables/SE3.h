#pragma once

#include <sophus/se3.hpp>

#include "tangent/Differentiation/Jet.h"
#include "tangent/Differentiation/JetTraits.h"
#include "tangent/Variables/OptimizableVariable.h"

namespace Tangent {

/**
 * Sophus SE3 transformation.
 *
 * The update is applied as a so3 + translation update not an se3 update.
 *
 * The order of the delta vector is [rotation, translation].
 */
class SE3 : public OptimizableVariableBase<double, 6> {
 public:
  Sophus::SE3<double> value;

  SE3() = default;

  SE3(const Sophus::SE3<double> &se3) : value(se3) {}

  SE3(Sophus::SE3<double> se3) : value(se3) {}

  void update(const Eigen::Matrix<double, 6, 1> &dx) {
    value.so3() *= Sophus::SO3d::exp(dx.block<3, 1>(0, 0));
    value.translation() += dx.block<3, 1>(3, 0);
  }
};

// ============================================================================
// Autodiff Support
// ============================================================================

/// Extract raw value for residual-only computation.
inline const Sophus::SE3d &getValue(const SE3 &se3) { return se3.value; }

/**
 * @brief Lift SE3 to Jet space for automatic differentiation.
 * Parameterization: [omega(3), v(3)] matching update() convention.
 */
template <typename T, int N>
Sophus::SE3<Jet<T, N>> liftToJet(const SE3 &se3, int offset) {
  using JetT = Jet<T, N>;

  Eigen::Quaternion<JetT> q;
  q.w() = JetT(se3.value.unit_quaternion().w());
  q.x() = JetT(se3.value.unit_quaternion().x());
  q.y() = JetT(se3.value.unit_quaternion().y());
  q.z() = JetT(se3.value.unit_quaternion().z());

  Eigen::Matrix<JetT, 3, 1> t;
  t(0) = JetT(se3.value.translation()(0));
  t(1) = JetT(se3.value.translation()(1));
  t(2) = JetT(se3.value.translation()(2));

  Sophus::SE3<JetT> result(q, t);

  // Seed perturbation
  Eigen::Matrix<JetT, 6, 1> delta;
  for (int i = 0; i < 6; ++i) {
    delta(i) = JetT(T(0), offset + i);
  }

  result.so3() =
      result.so3() * Sophus::SO3<JetT>::exp(delta.template head<3>());
  result.translation() += delta.template tail<3>();

  return result;
}

}  // namespace Tangent
