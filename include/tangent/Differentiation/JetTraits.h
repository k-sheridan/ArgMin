#pragma once

#include "tangent/Differentiation/Jet.h"

#include <Eigen/Core>
#include <limits>

namespace Eigen {

/**
 * @brief NumTraits specialization for Jet<T, N> to enable Eigen compatibility.
 *
 * This allows Eigen matrices to use Jet<T, N> as their scalar type, enabling
 * automatic differentiation through matrix operations.
 *
 * Example:
 * @code
 * using JetType = Tangent::Jet<double, 6>;
 * Eigen::Matrix<JetType, 3, 1> v;
 * Eigen::Matrix<JetType, 3, 3> M;
 * auto result = M * v;  // Works with automatic differentiation
 * @endcode
 */
template <typename T, int N>
struct NumTraits<Tangent::Jet<T, N>> : NumTraits<T> {
  using Real = Tangent::Jet<T, N>;
  using NonInteger = Tangent::Jet<T, N>;
  using Nested = Tangent::Jet<T, N>;
  using Literal = Tangent::Jet<T, N>;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = NumTraits<T>::AddCost,
    MulCost = NumTraits<T>::MulCost + N * NumTraits<T>::AddCost
  };

  static Tangent::Jet<T, N> epsilon() {
    return Tangent::Jet<T, N>(NumTraits<T>::epsilon());
  }

  static Tangent::Jet<T, N> dummy_precision() {
    return Tangent::Jet<T, N>(NumTraits<T>::dummy_precision());
  }

  static Tangent::Jet<T, N> lowest() {
    return Tangent::Jet<T, N>(NumTraits<T>::lowest());
  }

  static Tangent::Jet<T, N> highest() {
    return Tangent::Jet<T, N>(NumTraits<T>::highest());
  }

  static Tangent::Jet<T, N> infinity() {
    return Tangent::Jet<T, N>(NumTraits<T>::infinity());
  }

  static Tangent::Jet<T, N> quiet_NaN() {
    return Tangent::Jet<T, N>(NumTraits<T>::quiet_NaN());
  }

  static int digits10() { return NumTraits<T>::digits10(); }
};

/**
 * @brief ScalarBinaryOpTraits for Jet<T, N> with T.
 *
 * Enables mixed operations between Jet and the underlying scalar type T
 * in Eigen expressions.
 */
template <typename T, int N, typename BinaryOp>
struct ScalarBinaryOpTraits<Tangent::Jet<T, N>, T, BinaryOp> {
  using ReturnType = Tangent::Jet<T, N>;
};

template <typename T, int N, typename BinaryOp>
struct ScalarBinaryOpTraits<T, Tangent::Jet<T, N>, BinaryOp> {
  using ReturnType = Tangent::Jet<T, N>;
};

/**
 * @brief ScalarBinaryOpTraits for Jet<T, N> with int.
 *
 * Enables operations with integer literals in Eigen expressions.
 */
template <typename T, int N, typename BinaryOp>
struct ScalarBinaryOpTraits<Tangent::Jet<T, N>, int, BinaryOp> {
  using ReturnType = Tangent::Jet<T, N>;
};

template <typename T, int N, typename BinaryOp>
struct ScalarBinaryOpTraits<int, Tangent::Jet<T, N>, BinaryOp> {
  using ReturnType = Tangent::Jet<T, N>;
};

}  // namespace Eigen

namespace Tangent {

// ============================================================================
// Utility functions for working with Jet matrices
// ============================================================================

/**
 * @brief Create a Jet matrix from a double matrix with derivative seeding.
 *
 * Each element of the resulting matrix will have its derivative seeded
 * sequentially starting from startDerivativeIndex.
 *
 * @tparam T Underlying scalar type
 * @tparam N Total number of derivatives
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @param values The input matrix of scalar values
 * @param startDerivativeIndex The index at which to start seeding derivatives
 * @return A matrix of Jets with seeded derivatives
 */
template <typename T, int N, int Rows, int Cols>
Eigen::Matrix<Jet<T, N>, Rows, Cols> seedJetMatrix(
    const Eigen::Matrix<T, Rows, Cols>& values, int startDerivativeIndex) {
  Eigen::Matrix<Jet<T, N>, Rows, Cols> result;
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      int idx = startDerivativeIndex + i * Cols + j;
      if (idx >= 0 && idx < N) {
        result(i, j) = Jet<T, N>(values(i, j), idx);
      } else {
        result(i, j) = Jet<T, N>(values(i, j));
      }
    }
  }
  return result;
}

/**
 * @brief Create a Jet vector from a double vector with derivative seeding.
 *
 * @tparam T Underlying scalar type
 * @tparam N Total number of derivatives
 * @tparam Rows Number of rows
 * @param values The input vector of scalar values
 * @param startDerivativeIndex The index at which to start seeding derivatives
 * @return A vector of Jets with seeded derivatives
 */
template <typename T, int N, int Rows>
Eigen::Matrix<Jet<T, N>, Rows, 1> seedJetVector(
    const Eigen::Matrix<T, Rows, 1>& values, int startDerivativeIndex) {
  return seedJetMatrix<T, N, Rows, 1>(values, startDerivativeIndex);
}

/**
 * @brief Convert a double matrix to a Jet matrix without derivative seeding.
 *
 * The resulting Jets will have zero derivatives.
 *
 * @tparam T Underlying scalar type
 * @tparam N Total number of derivatives
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @param values The input matrix of scalar values
 * @return A matrix of Jets with zero derivatives
 */
template <typename T, int N, int Rows, int Cols>
Eigen::Matrix<Jet<T, N>, Rows, Cols> toJetMatrix(
    const Eigen::Matrix<T, Rows, Cols>& values) {
  Eigen::Matrix<Jet<T, N>, Rows, Cols> result;
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      result(i, j) = Jet<T, N>(values(i, j));
    }
  }
  return result;
}

/**
 * @brief Convert a double vector to a Jet vector without derivative seeding.
 *
 * @tparam T Underlying scalar type
 * @tparam N Total number of derivatives
 * @tparam Rows Number of rows
 * @param values The input vector of scalar values
 * @return A vector of Jets with zero derivatives
 */
template <typename T, int N, int Rows>
Eigen::Matrix<Jet<T, N>, Rows, 1> toJetVector(
    const Eigen::Matrix<T, Rows, 1>& values) {
  return toJetMatrix<T, N, Rows, 1>(values);
}

/**
 * @brief Extract the scalar values from a Jet matrix.
 *
 * @tparam T Underlying scalar type
 * @tparam N Total number of derivatives
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @param jets The input matrix of Jets
 * @return A matrix of scalar values
 */
template <typename T, int N, int Rows, int Cols>
Eigen::Matrix<T, Rows, Cols> extractValues(
    const Eigen::Matrix<Jet<T, N>, Rows, Cols>& jets) {
  Eigen::Matrix<T, Rows, Cols> result;
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      result(i, j) = jets(i, j).a;
    }
  }
  return result;
}

/**
 * @brief Extract the scalar values from a Jet vector.
 *
 * @tparam T Underlying scalar type
 * @tparam N Total number of derivatives
 * @tparam Rows Number of rows
 * @param jets The input vector of Jets
 * @return A vector of scalar values
 */
template <typename T, int N, int Rows>
Eigen::Matrix<T, Rows, 1> extractValues(
    const Eigen::Matrix<Jet<T, N>, Rows, 1>& jets) {
  return extractValues<T, N, Rows, 1>(jets);
}

/**
 * @brief Extract the full Jacobian from a Jet vector (residual).
 *
 * The Jacobian J has dimensions (ResidualDim x N), where each row i
 * contains the gradient of residual[i] with respect to all N variables.
 *
 * @tparam T Underlying scalar type
 * @tparam N Total number of derivatives (columns of Jacobian)
 * @tparam ResidualDim Dimension of residual (rows of Jacobian)
 * @param jetResidual The residual vector as Jets
 * @return The full Jacobian matrix
 */
template <typename T, int N, int ResidualDim>
Eigen::Matrix<T, ResidualDim, N> extractFullJacobian(
    const Eigen::Matrix<Jet<T, N>, ResidualDim, 1>& jetResidual) {
  Eigen::Matrix<T, ResidualDim, N> J;
  for (int i = 0; i < ResidualDim; ++i) {
    J.row(i) = jetResidual(i).v.transpose();
  }
  return J;
}

/**
 * @brief Extract a block of the Jacobian for a specific variable.
 *
 * @tparam T Underlying scalar type
 * @tparam N Total number of derivatives
 * @tparam ResidualDim Dimension of residual
 * @tparam VarDim Dimension of the specific variable
 * @param jetResidual The residual vector as Jets
 * @param varOffset The column offset in the full Jacobian for this variable
 * @return The Jacobian block for this variable
 */
template <typename T, int N, int ResidualDim, int VarDim>
Eigen::Matrix<T, ResidualDim, VarDim> extractJacobianBlock(
    const Eigen::Matrix<Jet<T, N>, ResidualDim, 1>& jetResidual, int varOffset) {
  Eigen::Matrix<T, ResidualDim, VarDim> J;
  for (int i = 0; i < ResidualDim; ++i) {
    for (int j = 0; j < VarDim; ++j) {
      J(i, j) = jetResidual(i).v(varOffset + j);
    }
  }
  return J;
}

}  // namespace Tangent
