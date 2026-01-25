#pragma once

#include "tangent/Differentiation/Jet.h"
#include "tangent/Differentiation/JetTraits.h"
#include "tangent/ErrorTerms/ErrorTermBase.h"
#include "tangent/Types/MetaHelpers.h"

#include <tuple>
#include <type_traits>
#include <utility>

namespace Tangent {

namespace internal {

/**
 * @brief Compute total dimension of a variable pack at compile time.
 */
template <typename... Vars>
struct TotalDimension;

template <>
struct TotalDimension<> {
  static constexpr int value = 0;
};

template <typename First, typename... Rest>
struct TotalDimension<First, Rest...> {
  static constexpr int value =
      static_cast<int>(First::dimension) + TotalDimension<Rest...>::value;
};

/**
 * @brief Compute cumulative dimension offset for variable at index I.
 */
template <int I, typename... Vars>
struct DimensionOffset;

template <typename First, typename... Rest>
struct DimensionOffset<0, First, Rest...> {
  static constexpr int value = 0;
};

template <int I, typename First, typename... Rest>
struct DimensionOffset<I, First, Rest...> {
  static constexpr int value =
      static_cast<int>(First::dimension) + DimensionOffset<I - 1, Rest...>::value;
};

}  // namespace internal

/**
 * @brief Base class for error terms with automatic differentiation.
 *
 * Users derive from this class using CRTP and implement a templated
 * `computeError()` method that works with any scalar type. The base class
 * automatically computes Jacobians using dual numbers (Jets) when
 * `relinearize` is true.
 *
 * @tparam Derived The CRTP derived class type
 * @tparam ScalarType The underlying scalar type (typically double)
 * @tparam ResidualDimension The dimension of the residual vector
 * @tparam IndependentVariables... The variable types this error term depends on
 *
 * Usage example:
 * @code
 * class MyReprojectionError
 *     : public AutoDiffErrorTerm<MyReprojectionError, double, 2,
 *                                 SE3, SE3, InverseDepth> {
 * public:
 *     Eigen::Vector2d measurement;
 *     Eigen::Vector2d bearing;
 *
 *     MyReprojectionError(VariableKey<SE3> hostKey,
 *                         VariableKey<SE3> targetKey,
 *                         VariableKey<InverseDepth> depthKey,
 *                         Eigen::Vector2d z,
 *                         Eigen::Vector2d b)
 *         : measurement(z), bearing(b) {
 *         std::get<0>(this->variableKeys) = hostKey;
 *         std::get<1>(this->variableKeys) = targetKey;
 *         std::get<2>(this->variableKeys) = depthKey;
 *     }
 *
 *     // This single templated method works for both double and Jet types.
 *     // The template parameters are the "lifted" types of each variable:
 *     // - SE3 -> LiftedSE3<T, N> when T=Jet, or Sophus::SE3d when T=double
 *     // - InverseDepth -> Jet<T, N> when T=Jet, or double when T=double
 *     template <typename T, typename Host, typename Target, typename Depth>
 *     Eigen::Matrix<T, 2, 1> computeError(const Host& host,
 *                                          const Target& target,
 *                                          const Depth& depth) const {
 *         // Implement error computation using the lifted variables
 *         // The same code works for both value and autodiff evaluation
 *     }
 * };
 * @endcode
 *
 * The derived class must implement:
 * @code
 * template <typename T, typename... LiftedVars>
 * Eigen::Matrix<T, ResidualDimension, 1> computeError(
 *     const LiftedVars&... vars) const;
 * @endcode
 *
 * Where:
 * - T is either `ScalarType` (for residual-only) or `JetType` (for autodiff)
 * - LiftedVars are the lifted representations of each variable
 */
template <typename Derived, typename ScalarType, int ResidualDimension,
          typename... IndependentVariables>
class AutoDiffErrorTerm
    : public ErrorTermBase<Scalar<ScalarType>, Dimension<ResidualDimension>,
                           VariableGroup<IndependentVariables...>> {
 public:
  using Base = ErrorTermBase<Scalar<ScalarType>, Dimension<ResidualDimension>,
                             VariableGroup<IndependentVariables...>>;

  // Inherit member variables from base
  using Base::information;
  using Base::linearizationValid;
  using Base::residual;
  using Base::variableJacobians;
  using Base::variableKeys;
  using Base::variablePointers;

  // Total dimension of all variables combined
  static constexpr int TotalDim =
      internal::TotalDimension<IndependentVariables...>::value;

  // Jet type for automatic differentiation
  using JetType = Jet<ScalarType, TotalDim>;

  // Number of variables
  static constexpr size_t NumVariables = sizeof...(IndependentVariables);

  /**
   * @brief Evaluate the error term, computing residual and optionally
   * Jacobians.
   *
   * When relinearize is false, only the residual is computed using double
   * arithmetic (fast path). When relinearize is true, Jets are used to
   * automatically compute both residual and Jacobians.
   *
   * @param variables The variable container
   * @param relinearize If true, compute Jacobians via autodiff
   */
  template <typename... Variables>
  void evaluate(VariableContainer<Variables...>& variables, bool relinearize) {
    if (relinearize) {
      // Compute with Jets to get both residual and Jacobians
      auto jetResidual = computeWithJets(std::index_sequence_for<IndependentVariables...>{});

      // Extract residual values
      for (int i = 0; i < ResidualDimension; ++i) {
        residual(i) = jetResidual(i).a;
      }

      // Extract Jacobians for each variable
      extractAllJacobians(jetResidual,
                          std::index_sequence_for<IndependentVariables...>{});

      linearizationValid = true;
    } else {
      // Fast path: compute residual only with double arithmetic
      residual = computeWithDoubles(std::index_sequence_for<IndependentVariables...>{});
      linearizationValid = false;
    }

    // Default information matrix to identity (user can override in derived
    // class)
    if (information.isZero()) {
      information.setIdentity();
    }
  }

 protected:
  /**
   * @brief Get the dimension offset for variable at index I.
   */
  template <size_t I>
  static constexpr int getVariableOffset() {
    return internal::DimensionOffset<static_cast<int>(I),
                                     IndependentVariables...>::value;
  }

  /**
   * @brief Get the dimension of variable at index I.
   */
  template <size_t I>
  static constexpr int getVariableDimension() {
    using VarType =
        typename std::tuple_element<I,
                                    std::tuple<IndependentVariables...>>::type;
    return static_cast<int>(VarType::dimension);
  }

 private:
  /**
   * @brief Cast to derived class.
   */
  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  /**
   * @brief Compute residual using double arithmetic (no autodiff).
   */
  template <size_t... Is>
  Eigen::Matrix<ScalarType, ResidualDimension, 1> computeWithDoubles(
      std::index_sequence<Is...>) {
    return derived().template computeError<ScalarType>(
        getValue(*std::get<Is>(variablePointers))...);
  }

  /**
   * @brief Lift a single variable to Jet space.
   */
  template <size_t I>
  auto liftVariableAtIndex() const {
    constexpr int offset = getVariableOffset<I>();
    return liftToJet<ScalarType, TotalDim>(*std::get<I>(variablePointers),
                                           offset);
  }

  /**
   * @brief Compute residual using Jets (with autodiff).
   */
  template <size_t... Is>
  Eigen::Matrix<JetType, ResidualDimension, 1> computeWithJets(
      std::index_sequence<Is...>) {
    return derived().template computeError<JetType>(liftVariableAtIndex<Is>()...);
  }

  /**
   * @brief Extract Jacobian for a single variable from Jet residual.
   */
  template <size_t I>
  void extractJacobianForVariable(
      const Eigen::Matrix<JetType, ResidualDimension, 1>& jetResidual) {
    using VarType =
        typename std::tuple_element<I,
                                    std::tuple<IndependentVariables...>>::type;
    constexpr int varDim = static_cast<int>(VarType::dimension);
    constexpr int offset = getVariableOffset<I>();

    auto& J = std::get<I>(variableJacobians);
    for (int row = 0; row < ResidualDimension; ++row) {
      for (int col = 0; col < varDim; ++col) {
        J(row, col) = jetResidual(row).v(offset + col);
      }
    }
  }

  /**
   * @brief Extract Jacobians for all variables.
   */
  template <size_t... Is>
  void extractAllJacobians(
      const Eigen::Matrix<JetType, ResidualDimension, 1>& jetResidual,
      std::index_sequence<Is...>) {
    (extractJacobianForVariable<Is>(jetResidual), ...);
  }
};

}  // namespace Tangent
