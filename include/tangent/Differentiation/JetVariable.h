#pragma once

#include "tangent/Differentiation/Jet.h"
#include "tangent/Differentiation/JetTraits.h"
#include "tangent/Variables/InverseDepth.h"
#include "tangent/Variables/SE3.h"
#include "tangent/Variables/SimpleScalar.h"
#include "tangent/Variables/SO3.h"

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

namespace Tangent {

// ============================================================================
// Sophus Jet Type Aliases
// ============================================================================
// These aliases define Sophus types parameterized by Jet scalar type.
// Sophus is designed to work with any scalar type that satisfies Eigen's
// NumTraits requirements, which our Jet type does via JetTraits.h.

template <typename T, int N>
using SE3Jet = Sophus::SE3<Jet<T, N>>;

template <typename T, int N>
using SO3Jet = Sophus::SO3<Jet<T, N>>;

// ============================================================================
// Variable Lifting Functions
// ============================================================================

/**
 * @brief Lift an SE3 variable to Jet space with derivative seeding.
 *
 * The tangent space of SE3 is parameterized as [omega, v] where:
 * - omega (3D): rotation perturbation in Lie algebra so(3)
 * - v (3D): translation perturbation
 *
 * The update model follows SE3::update() in SE3.h:
 * - R_new = R * exp(omega)
 * - t_new = t + v
 *
 * We use Sophus::SE3<Jet> directly, applying a small perturbation via the
 * exponential map to capture the Jacobian structure.
 *
 * @tparam T Underlying scalar type
 * @tparam N Total number of derivatives
 * @param se3 The SE3 variable to lift
 * @param derivativeStartIndex The index at which to start seeding derivatives
 *        (indices 0-2 for rotation, 3-5 for translation)
 * @return A Sophus::SE3<Jet> with seeded derivatives
 */
template <typename T, int N>
SE3Jet<T, N> liftSE3(const SE3& se3, int derivativeStartIndex) {
  using JetT = Jet<T, N>;

  // Convert base SE3 to Jet-valued SE3 (no derivatives yet)
  Eigen::Quaternion<JetT> q_jet;
  q_jet.w() = JetT(se3.value.unit_quaternion().w());
  q_jet.x() = JetT(se3.value.unit_quaternion().x());
  q_jet.y() = JetT(se3.value.unit_quaternion().y());
  q_jet.z() = JetT(se3.value.unit_quaternion().z());

  Eigen::Matrix<JetT, 3, 1> t_jet;
  t_jet(0) = JetT(se3.value.translation()(0));
  t_jet(1) = JetT(se3.value.translation()(1));
  t_jet(2) = JetT(se3.value.translation()(2));

  SE3Jet<T, N> se3_jet(q_jet, t_jet);

  // Create seeded perturbation vector [omega(3), v(3)]
  Eigen::Matrix<JetT, 6, 1> delta;
  for (int i = 0; i < 6; ++i) {
    delta(i) = JetT(T(0), derivativeStartIndex + i);
  }

  // Apply perturbation following SE3::update() convention:
  // R_new = R * exp(omega), t_new = t + v
  Sophus::SO3<JetT> so3_perturb = Sophus::SO3<JetT>::exp(delta.template head<3>());
  se3_jet.so3() = se3_jet.so3() * so3_perturb;
  se3_jet.translation() += delta.template tail<3>();

  return se3_jet;
}

/**
 * @brief Lift an SO3 variable to Jet space with derivative seeding.
 *
 * The tangent space of SO3 is parameterized as omega (3D rotation in so(3)).
 *
 * The update model follows SO3::update() in SO3.h:
 * - R_new = R * exp(omega)
 *
 * We use Sophus::SO3<Jet> directly.
 *
 * @tparam T Underlying scalar type
 * @tparam N Total number of derivatives
 * @param so3 The SO3 variable to lift
 * @param derivativeStartIndex The index at which to start seeding derivatives
 * @return A Sophus::SO3<Jet> with seeded derivatives
 */
template <typename T, int N>
SO3Jet<T, N> liftSO3(const SO3& so3, int derivativeStartIndex) {
  using JetT = Jet<T, N>;

  // Convert base SO3 to Jet-valued SO3 (no derivatives yet)
  Eigen::Quaternion<JetT> q_jet;
  q_jet.w() = JetT(so3.value.unit_quaternion().w());
  q_jet.x() = JetT(so3.value.unit_quaternion().x());
  q_jet.y() = JetT(so3.value.unit_quaternion().y());
  q_jet.z() = JetT(so3.value.unit_quaternion().z());

  SO3Jet<T, N> so3_jet(q_jet);

  // Create seeded perturbation omega
  Eigen::Matrix<JetT, 3, 1> omega;
  for (int i = 0; i < 3; ++i) {
    omega(i) = JetT(T(0), derivativeStartIndex + i);
  }

  // Apply perturbation: R_new = R * exp(omega)
  Sophus::SO3<JetT> so3_perturb = Sophus::SO3<JetT>::exp(omega);
  so3_jet = so3_jet * so3_perturb;

  return so3_jet;
}

/**
 * @brief Lift an InverseDepth variable to Jet space.
 *
 * InverseDepth is a scalar variable, so lifting is straightforward.
 *
 * @tparam T Underlying scalar type
 * @tparam N Total number of derivatives
 * @param dinv The InverseDepth variable to lift
 * @param derivativeStartIndex The index for the derivative seed
 * @return A Jet representing the inverse depth with seeded derivative
 */
template <typename T, int N>
Jet<T, N> liftInverseDepth(const InverseDepth& dinv, int derivativeStartIndex) {
  return Jet<T, N>(static_cast<T>(dinv.value), derivativeStartIndex);
}

/**
 * @brief Lift a SimpleScalar variable to Jet space.
 *
 * SimpleScalar is a scalar variable, so lifting is straightforward.
 *
 * @tparam T Underlying scalar type
 * @tparam N Total number of derivatives
 * @param scalar The SimpleScalar variable to lift
 * @param derivativeStartIndex The index for the derivative seed
 * @return A Jet representing the scalar with seeded derivative
 */
template <typename T, int N>
Jet<T, N> liftSimpleScalar(const SimpleScalar& scalar,
                           int derivativeStartIndex) {
  return Jet<T, N>(static_cast<T>(scalar.value), derivativeStartIndex);
}

// ============================================================================
// Value Extraction Helpers (for double computation path)
// ============================================================================

/**
 * @brief Extract the raw SE3 value from an SE3 variable.
 */
inline const Sophus::SE3d& getValue(const SE3& se3) { return se3.value; }

/**
 * @brief Extract the raw SO3 value from an SO3 variable.
 */
inline const Sophus::SO3d& getValue(const SO3& so3) { return so3.value; }

/**
 * @brief Extract the raw double value from an InverseDepth variable.
 */
inline double getValue(const InverseDepth& dinv) { return dinv.value; }

/**
 * @brief Extract the raw double value from a SimpleScalar variable.
 */
inline double getValue(const SimpleScalar& scalar) { return scalar.value; }

// ============================================================================
// Type Traits for Variable Lifting
// ============================================================================

/**
 * @brief Traits class for determining the lifted type of a variable.
 *
 * This provides a mapping from variable types to their Jet-lifted equivalents.
 * SE3 and SO3 lift to Sophus types parameterized by Jet.
 * Scalar types lift directly to Jet.
 */
template <typename Variable, typename T, int N>
struct LiftedVariableType;

template <typename T, int N>
struct LiftedVariableType<SE3, T, N> {
  using type = SE3Jet<T, N>;  // Sophus::SE3<Jet<T, N>>
};

template <typename T, int N>
struct LiftedVariableType<SO3, T, N> {
  using type = SO3Jet<T, N>;  // Sophus::SO3<Jet<T, N>>
};

template <typename T, int N>
struct LiftedVariableType<InverseDepth, T, N> {
  using type = Jet<T, N>;
};

template <typename T, int N>
struct LiftedVariableType<SimpleScalar, T, N> {
  using type = Jet<T, N>;
};

/**
 * @brief Convenience alias for the lifted type of a variable.
 */
template <typename Variable, typename T, int N>
using LiftedType = typename LiftedVariableType<Variable, T, N>::type;

// ============================================================================
// Generic Lifting Dispatch
// ============================================================================

namespace internal {

// Primary template - will be specialized
template <typename Variable>
struct VariableLiftCategory {
  static constexpr int value = 0;  // Unknown
};

template <>
struct VariableLiftCategory<SE3> {
  static constexpr int value = 1;  // SE3
};

template <>
struct VariableLiftCategory<SO3> {
  static constexpr int value = 2;  // SO3
};

template <>
struct VariableLiftCategory<InverseDepth> {
  static constexpr int value = 3;  // InverseDepth
};

template <>
struct VariableLiftCategory<SimpleScalar> {
  static constexpr int value = 4;  // SimpleScalar
};

// Implementation for each category
template <int Category>
struct LiftVariableImpl;

template <>
struct LiftVariableImpl<1> {  // SE3
  template <typename T, int N>
  static SE3Jet<T, N> lift(const SE3& var, int offset) {
    return liftSE3<T, N>(var, offset);
  }
};

template <>
struct LiftVariableImpl<2> {  // SO3
  template <typename T, int N>
  static SO3Jet<T, N> lift(const SO3& var, int offset) {
    return liftSO3<T, N>(var, offset);
  }
};

template <>
struct LiftVariableImpl<3> {  // InverseDepth
  template <typename T, int N>
  static auto lift(const InverseDepth& var, int offset) {
    return liftInverseDepth<T, N>(var, offset);
  }
};

template <>
struct LiftVariableImpl<4> {  // SimpleScalar (and derived types)
  template <typename T, int N, typename Variable>
  static auto lift(const Variable& var, int offset) {
    return Jet<T, N>(static_cast<T>(var.value), offset);
  }
};

template <>
struct LiftVariableImpl<0> {  // Unknown - fallback for scalar-like types
  template <typename T, int N, typename Variable>
  static auto lift(const Variable& var, int offset) {
    static_assert(Variable::dimension == 1,
                  "Unknown variable type must have dimension 1");
    return Jet<T, N>(static_cast<T>(var.value), offset);
  }
};

// Trait to detect if a type derives from SimpleScalar
template <typename T>
struct is_scalar_variable {
  static constexpr bool value =
      std::is_base_of_v<SimpleScalar, T> || (T::dimension == 1);
};

// Get the lift category, treating SimpleScalar-derived types as SimpleScalar
template <typename Variable>
constexpr int getLiftCategory() {
  if constexpr (std::is_same_v<Variable, SE3>) {
    return 1;
  } else if constexpr (std::is_same_v<Variable, SO3>) {
    return 2;
  } else if constexpr (std::is_same_v<Variable, InverseDepth>) {
    return 3;
  } else if constexpr (std::is_base_of_v<SimpleScalar, Variable>) {
    return 4;  // Treat derived types as SimpleScalar
  } else if constexpr (Variable::dimension == 1) {
    return 4;  // Other scalar-like types
  } else {
    return 0;  // Unknown
  }
}

}  // namespace internal

/**
 * @brief Lift a variable to Jet space (generic dispatch).
 *
 * Automatically selects the correct lifting function based on the variable
 * type. Supports:
 * - SE3 -> LiftedSE3
 * - SO3 -> LiftedSO3
 * - InverseDepth -> Jet
 * - SimpleScalar (and derived types) -> Jet
 * - Any scalar-like type with dimension=1 and value member -> Jet
 */
template <typename T, int N, typename Variable>
auto liftVariable(const Variable& var, int derivativeStartIndex) {
  constexpr int category = internal::getLiftCategory<Variable>();
  return internal::LiftVariableImpl<category>::template lift<T, N>(
      var, derivativeStartIndex);
}

/**
 * @brief Generic getValue for any scalar-like variable.
 *
 * Handles types that inherit from SimpleScalar or have a 'value' member.
 */
template <typename Variable>
auto getValue(const Variable& var)
    -> std::enable_if_t<internal::is_scalar_variable<Variable>::value &&
                            !std::is_same_v<Variable, SE3> &&
                            !std::is_same_v<Variable, SO3> &&
                            !std::is_same_v<Variable, InverseDepth> &&
                            !std::is_same_v<Variable, SimpleScalar>,
                        double> {
  return var.value;
}

}  // namespace Tangent
