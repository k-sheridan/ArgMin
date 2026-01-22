#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "tangent/Differentiation/Jet.h"
#include "tangent/Differentiation/JetTraits.h"
#include "tangent/Differentiation/JetVariable.h"
#include "tangent/Differentiation/NumericalDifferentiator.h"
#include "tangent/ErrorTerms/AutoDiffErrorTerm.h"
#include "tangent/ErrorTerms/ErrorTermValidator.h"
#include "tangent/Optimization/OptimizerContainers.h"
#include "tangent/Variables/InverseDepth.h"
#include "tangent/Variables/SE3.h"
#include "tangent/Variables/SimpleScalar.h"
#include "tangent/Variables/SO3.h"

#include "TestAutoDiffErrorTerms.h"
#include "TestUtils.h"

using namespace Tangent;
using namespace Tangent::Test;

// =============================================================================
// Jet Arithmetic Tests
// =============================================================================

TEST(JetTest, DefaultConstructor) {
  Jet<double, 3> j;
  EXPECT_DOUBLE_EQ(j.a, 0.0);
  EXPECT_TRUE(j.v.isZero());
}

TEST(JetTest, ValueConstructor) {
  Jet<double, 3> j(5.0);
  EXPECT_DOUBLE_EQ(j.a, 5.0);
  EXPECT_TRUE(j.v.isZero());
}

TEST(JetTest, SeedConstructor) {
  Jet<double, 3> j(5.0, 1);
  EXPECT_DOUBLE_EQ(j.a, 5.0);
  EXPECT_DOUBLE_EQ(j.v(0), 0.0);
  EXPECT_DOUBLE_EQ(j.v(1), 1.0);
  EXPECT_DOUBLE_EQ(j.v(2), 0.0);
}

TEST(JetTest, Addition) {
  Jet<double, 2> x(3.0, 0);
  Jet<double, 2> y(4.0, 1);

  auto sum = x + y;
  EXPECT_DOUBLE_EQ(sum.a, 7.0);
  EXPECT_DOUBLE_EQ(sum.v(0), 1.0);
  EXPECT_DOUBLE_EQ(sum.v(1), 1.0);
}

TEST(JetTest, Subtraction) {
  Jet<double, 2> x(3.0, 0);
  Jet<double, 2> y(4.0, 1);

  auto diff = x - y;
  EXPECT_DOUBLE_EQ(diff.a, -1.0);
  EXPECT_DOUBLE_EQ(diff.v(0), 1.0);
  EXPECT_DOUBLE_EQ(diff.v(1), -1.0);
}

TEST(JetTest, Multiplication) {
  Jet<double, 2> x(3.0, 0);
  Jet<double, 2> y(4.0, 1);

  auto prod = x * y;
  EXPECT_DOUBLE_EQ(prod.a, 12.0);
  EXPECT_DOUBLE_EQ(prod.v(0), 4.0);  // d(xy)/dx = y
  EXPECT_DOUBLE_EQ(prod.v(1), 3.0);  // d(xy)/dy = x
}

TEST(JetTest, Division) {
  Jet<double, 2> x(6.0, 0);
  Jet<double, 2> y(2.0, 1);

  auto quot = x / y;
  EXPECT_DOUBLE_EQ(quot.a, 3.0);
  EXPECT_DOUBLE_EQ(quot.v(0), 0.5);   // d(x/y)/dx = 1/y = 0.5
  EXPECT_DOUBLE_EQ(quot.v(1), -1.5);  // d(x/y)/dy = -x/y^2 = -6/4
}

TEST(JetTest, ChainRule) {
  // f(x) = x^2, df/dx = 2x
  Jet<double, 1> x(3.0, 0);
  auto f = x * x;
  EXPECT_DOUBLE_EQ(f.a, 9.0);
  EXPECT_DOUBLE_EQ(f.v(0), 6.0);  // 2 * 3
}

TEST(JetTest, ComplexExpression) {
  // f(x, y) = (x + y) * (x - y) = x^2 - y^2
  // df/dx = 2x, df/dy = -2y
  Jet<double, 2> x(3.0, 0);
  Jet<double, 2> y(2.0, 1);

  auto f = (x + y) * (x - y);
  EXPECT_DOUBLE_EQ(f.a, 5.0);   // 9 - 4
  EXPECT_DOUBLE_EQ(f.v(0), 6.0);  // 2 * 3
  EXPECT_DOUBLE_EQ(f.v(1), -4.0); // -2 * 2
}

TEST(JetTest, ScalarOperations) {
  Jet<double, 2> x(3.0, 0);

  auto sum = x + 2.0;
  EXPECT_DOUBLE_EQ(sum.a, 5.0);
  EXPECT_DOUBLE_EQ(sum.v(0), 1.0);

  auto prod = x * 2.0;
  EXPECT_DOUBLE_EQ(prod.a, 6.0);
  EXPECT_DOUBLE_EQ(prod.v(0), 2.0);

  auto leftProd = 2.0 * x;
  EXPECT_DOUBLE_EQ(leftProd.a, 6.0);
  EXPECT_DOUBLE_EQ(leftProd.v(0), 2.0);
}

// =============================================================================
// Jet Math Function Tests
// =============================================================================

TEST(JetMathTest, Sin) {
  Jet<double, 1> x(M_PI / 4, 0);
  auto s = sin(x);
  EXPECT_NEAR(s.a, std::sqrt(2.0) / 2.0, 1e-10);
  EXPECT_NEAR(s.v(0), std::sqrt(2.0) / 2.0, 1e-10);  // cos(pi/4)
}

TEST(JetMathTest, Cos) {
  Jet<double, 1> x(M_PI / 3, 0);
  auto c = cos(x);
  EXPECT_NEAR(c.a, 0.5, 1e-10);
  EXPECT_NEAR(c.v(0), -std::sqrt(3.0) / 2.0, 1e-10);  // -sin(pi/3)
}

TEST(JetMathTest, Exp) {
  Jet<double, 1> x(1.0, 0);
  auto e = exp(x);
  EXPECT_NEAR(e.a, std::exp(1.0), 1e-10);
  EXPECT_NEAR(e.v(0), std::exp(1.0), 1e-10);  // d(e^x)/dx = e^x
}

TEST(JetMathTest, Log) {
  Jet<double, 1> x(2.0, 0);
  auto l = log(x);
  EXPECT_NEAR(l.a, std::log(2.0), 1e-10);
  EXPECT_NEAR(l.v(0), 0.5, 1e-10);  // d(ln x)/dx = 1/x
}

TEST(JetMathTest, Sqrt) {
  Jet<double, 1> x(4.0, 0);
  auto s = sqrt(x);
  EXPECT_NEAR(s.a, 2.0, 1e-10);
  EXPECT_NEAR(s.v(0), 0.25, 1e-10);  // d(sqrt(x))/dx = 1/(2*sqrt(x))
}

TEST(JetMathTest, Pow) {
  Jet<double, 1> x(2.0, 0);
  auto p = pow(x, 3.0);
  EXPECT_NEAR(p.a, 8.0, 1e-10);
  EXPECT_NEAR(p.v(0), 12.0, 1e-10);  // d(x^3)/dx = 3*x^2
}

TEST(JetMathTest, Atan2) {
  Jet<double, 2> y(1.0, 0);
  Jet<double, 2> x(1.0, 1);
  auto a = atan2(y, x);
  EXPECT_NEAR(a.a, M_PI / 4, 1e-10);
  // d(atan2(y,x))/dy = x/(x^2+y^2) = 1/2
  EXPECT_NEAR(a.v(0), 0.5, 1e-10);
  // d(atan2(y,x))/dx = -y/(x^2+y^2) = -1/2
  EXPECT_NEAR(a.v(1), -0.5, 1e-10);
}

// =============================================================================
// Eigen Compatibility Tests
// =============================================================================

TEST(JetEigenTest, VectorOperations) {
  using JetT = Jet<double, 3>;
  Eigen::Matrix<JetT, 3, 1> v;
  v << JetT(1.0, 0), JetT(2.0, 1), JetT(3.0, 2);

  auto sum = v(0) + v(1) + v(2);
  EXPECT_DOUBLE_EQ(sum.a, 6.0);
  EXPECT_DOUBLE_EQ(sum.v(0), 1.0);
  EXPECT_DOUBLE_EQ(sum.v(1), 1.0);
  EXPECT_DOUBLE_EQ(sum.v(2), 1.0);
}

TEST(JetEigenTest, MatrixVectorMultiply) {
  using JetT = Jet<double, 3>;
  Eigen::Matrix<JetT, 3, 3> M;
  M.setIdentity();

  Eigen::Matrix<JetT, 3, 1> v;
  v << JetT(1.0, 0), JetT(2.0, 1), JetT(3.0, 2);

  auto result = M * v;
  EXPECT_DOUBLE_EQ(result(0).a, 1.0);
  EXPECT_DOUBLE_EQ(result(1).a, 2.0);
  EXPECT_DOUBLE_EQ(result(2).a, 3.0);
}

TEST(JetEigenTest, CrossProduct) {
  using JetT = Jet<double, 6>;
  Eigen::Matrix<JetT, 3, 1> a, b;

  a << JetT(1.0, 0), JetT(0.0, 1), JetT(0.0, 2);
  b << JetT(0.0, 3), JetT(1.0, 4), JetT(0.0, 5);

  auto c = a.cross(b);

  // [1,0,0] x [0,1,0] = [0,0,1]
  EXPECT_NEAR(c(0).a, 0.0, 1e-10);
  EXPECT_NEAR(c(1).a, 0.0, 1e-10);
  EXPECT_NEAR(c(2).a, 1.0, 1e-10);
}

// =============================================================================
// Variable Lifting Tests
// =============================================================================

TEST(JetVariableTest, SE3LiftingPreservesValue) {
  SE3 pose;
  pose.value = Sophus::SE3d::rotX(0.1) * Sophus::SE3d::trans(1, 2, 3);

  auto lifted = liftSE3<double, 6>(pose, 0);

  // Value should match original rotation matrix
  Eigen::Matrix3d R = pose.value.rotationMatrix();
  auto liftedR = lifted.rotationMatrix();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(liftedR(i, j).a, R(i, j), 1e-10);
    }
  }

  // Value should match original translation
  Eigen::Vector3d t = pose.value.translation();
  auto liftedT = lifted.translation();
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(liftedT(i).a, t(i), 1e-10);
  }
}

TEST(JetVariableTest, SE3LiftingDerivatives) {
  SE3 pose;
  pose.value = Sophus::SE3d::trans(1, 2, 3);  // Identity rotation

  auto lifted = liftSE3<double, 6>(pose, 0);

  // For identity rotation, perturbation in omega should affect rotation as:
  // R * exp(omega) ≈ R * (I + [omega]_x)
  // So dR(0,1)/d(omega_z) = -1, dR(1,0)/d(omega_z) = 1, etc.

  // Check translation derivatives: dt/dv = I
  auto liftedT = lifted.translation();
  EXPECT_NEAR(liftedT(0).v(3), 1.0, 1e-10);  // dt_x/dv_x
  EXPECT_NEAR(liftedT(1).v(4), 1.0, 1e-10);  // dt_y/dv_y
  EXPECT_NEAR(liftedT(2).v(5), 1.0, 1e-10);  // dt_z/dv_z
}

TEST(JetVariableTest, SO3LiftingPreservesValue) {
  SO3 rot;
  rot.value = Sophus::SO3d::exp(Eigen::Vector3d(0.1, 0.2, 0.3));

  auto lifted = liftSO3<double, 3>(rot, 0);

  Eigen::Matrix3d R = rot.value.matrix();
  auto liftedR = lifted.matrix();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(liftedR(i, j).a, R(i, j), 1e-10);
    }
  }
}

TEST(JetVariableTest, InverseDepthLifting) {
  InverseDepth dinv(0.5);

  auto lifted = liftInverseDepth<double, 1>(dinv, 0);

  EXPECT_DOUBLE_EQ(lifted.a, 0.5);
  EXPECT_DOUBLE_EQ(lifted.v(0), 1.0);  // d(dinv)/d(dinv) = 1
}

TEST(JetVariableTest, SimpleScalarLifting) {
  SimpleScalar scalar(42.0);

  auto lifted = liftSimpleScalar<double, 1>(scalar, 0);

  EXPECT_DOUBLE_EQ(lifted.a, 42.0);
  EXPECT_DOUBLE_EQ(lifted.v(0), 1.0);
}

// =============================================================================
// AutoDiff Error Term Tests
// =============================================================================

class AutoDiffTest : public ::testing::Test {
 protected:
  std::mt19937 gen{42};
  std::uniform_real_distribution<> smallValue{-1.0, 1.0};

  void SetUp() override {}
};

TEST_F(AutoDiffTest, DifferenceErrorTermResidual) {
  VariableContainer<SimpleScalar, DifferentSimpleScalar> variables;
  auto k1 = variables.insert(SimpleScalar(5.0));
  auto k2 = variables.insert(DifferentSimpleScalar(3.0));

  DifferenceErrorTermAutoDiff errorTerm(k1, k2);
  errorTerm.updateVariablePointers(variables);
  errorTerm.evaluate(variables, false);

  // Residual should be var2 - var1 = 3 - 5 = -2
  EXPECT_NEAR(errorTerm.residual(0), -2.0, 1e-10);
}

TEST_F(AutoDiffTest, DifferenceErrorTermJacobians) {
  VariableContainer<SimpleScalar, DifferentSimpleScalar> variables;
  auto k1 = variables.insert(SimpleScalar(5.0));
  auto k2 = variables.insert(DifferentSimpleScalar(3.0));

  DifferenceErrorTermAutoDiff errorTerm(k1, k2);
  errorTerm.updateVariablePointers(variables);
  errorTerm.evaluate(variables, true);

  // Jacobian w.r.t var1: d(var2 - var1)/d(var1) = -1
  EXPECT_NEAR(std::get<0>(errorTerm.variableJacobians)(0, 0), -1.0, 1e-10);

  // Jacobian w.r.t var2: d(var2 - var1)/d(var2) = 1
  EXPECT_NEAR(std::get<1>(errorTerm.variableJacobians)(0, 0), 1.0, 1e-10);
}

TEST_F(AutoDiffTest, DifferenceErrorTermMatchesManual) {
  VariableContainer<SimpleScalar, DifferentSimpleScalar> variables;
  auto k1 = variables.insert(SimpleScalar(5.0));
  auto k2 = variables.insert(DifferentSimpleScalar(3.0));

  // AutoDiff version
  DifferenceErrorTermAutoDiff autoDiffTerm(k1, k2);
  autoDiffTerm.updateVariablePointers(variables);
  autoDiffTerm.evaluate(variables, true);

  // Manual version
  DifferenceErrorTerm manualTerm(k1, k2);
  manualTerm.updateVariablePointers(variables);
  manualTerm.evaluate(variables, true);

  // Residuals should match
  EXPECT_NEAR(autoDiffTerm.residual(0), manualTerm.residual(0), 1e-10);

  // Jacobians should match
  EXPECT_NEAR(std::get<0>(autoDiffTerm.variableJacobians)(0, 0),
              std::get<0>(manualTerm.variableJacobians)(0, 0), 1e-10);
  EXPECT_NEAR(std::get<1>(autoDiffTerm.variableJacobians)(0, 0),
              std::get<1>(manualTerm.variableJacobians)(0, 0), 1e-10);
}

TEST_F(AutoDiffTest, DifferenceErrorTermMatchesNumerical) {
  VariableContainer<SimpleScalar, DifferentSimpleScalar> variables;
  auto k1 = variables.insert(SimpleScalar(5.0));
  auto k2 = variables.insert(DifferentSimpleScalar(3.0));

  DifferenceErrorTermAutoDiff errorTerm(k1, k2);
  errorTerm.updateVariablePointers(variables);
  errorTerm.evaluate(variables, true);

  // Numerical differentiation
  auto numericalJacobians = numericallyDifferentiate(errorTerm, variables);

  EXPECT_NEAR(std::get<0>(errorTerm.variableJacobians)(0, 0),
              std::get<0>(numericalJacobians)(0, 0), 1e-5);
  EXPECT_NEAR(std::get<1>(errorTerm.variableJacobians)(0, 0),
              std::get<1>(numericalJacobians)(0, 0), 1e-5);
}

TEST_F(AutoDiffTest, QuadraticErrorTerm) {
  VariableContainer<SimpleScalar> variables;
  auto k = variables.insert(SimpleScalar(5.0));

  QuadraticErrorTermAutoDiff errorTerm(k, 3.0);  // target = 3
  errorTerm.updateVariablePointers(variables);
  errorTerm.evaluate(variables, true);

  // Residual: (5 - 3)^2 = 4
  EXPECT_NEAR(errorTerm.residual(0), 4.0, 1e-10);

  // Jacobian: d((x-3)^2)/dx = 2*(x-3) = 2*(5-3) = 4
  EXPECT_NEAR(std::get<0>(errorTerm.variableJacobians)(0, 0), 4.0, 1e-10);

  // Verify against numerical
  auto numericalJacobians = numericallyDifferentiate(errorTerm, variables);
  EXPECT_NEAR(std::get<0>(errorTerm.variableJacobians)(0, 0),
              std::get<0>(numericalJacobians)(0, 0), 1e-5);
}

// =============================================================================
// Reprojection Error Tests
// =============================================================================

class ReprojectionErrorTest : public ::testing::Test {
 protected:
  VariableContainer<SE3, InverseDepth> variables;
  VariableKey<SE3> hostK, targetK;
  VariableKey<InverseDepth> dinvK;

  void SetUp() override {
    SE3 host, target;
    host.value = Sophus::SE3d::trans(0, 0, 0);
    target.value = Sophus::SE3d::trans(0.5, 0, 0);

    hostK = variables.insert(host);
    targetK = variables.insert(target);
    dinvK = variables.insert(InverseDepth(1.0));
  }
};

TEST_F(ReprojectionErrorTest, AutoDiffResidualMatchesManual) {
  Eigen::Vector2d bearing(0.1, 0.2);

  // Create both error terms with matching measurements
  auto autoDiffTerm = createAutoDiffReprojectionErrorTerm(
      variables, hostK, targetK, dinvK, bearing);
  auto manualTerm = createReprojectionErrorTerm(
      variables, hostK, targetK, dinvK, bearing);

  autoDiffTerm.updateVariablePointers(variables);
  manualTerm.updateVariablePointers(variables);

  autoDiffTerm.evaluate(variables, false);
  manualTerm.evaluate(variables, false);

  // Residuals should be approximately zero (created with correct measurement)
  EXPECT_NEAR(autoDiffTerm.residual.norm(), 0.0, 1e-8);
  EXPECT_NEAR(manualTerm.residual.norm(), 0.0, 1e-8);
}

TEST_F(ReprojectionErrorTest, AutoDiffJacobiansMatchNumerical) {
  Eigen::Vector2d bearing(0.1, 0.2);

  auto errorTerm = createAutoDiffReprojectionErrorTerm(
      variables, hostK, targetK, dinvK, bearing);
  errorTerm.updateVariablePointers(variables);
  errorTerm.evaluate(variables, true);

  // Numerical differentiation
  auto numericalJacobians = numericallyDifferentiate(errorTerm, variables);

  // Host Jacobian (2x6)
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_NEAR(std::get<0>(errorTerm.variableJacobians)(i, j),
                  std::get<0>(numericalJacobians)(i, j), 1e-4)
          << "Host Jacobian mismatch at (" << i << ", " << j << ")";
    }
  }

  // Target Jacobian (2x6)
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_NEAR(std::get<1>(errorTerm.variableJacobians)(i, j),
                  std::get<1>(numericalJacobians)(i, j), 1e-4)
          << "Target Jacobian mismatch at (" << i << ", " << j << ")";
    }
  }

  // InverseDepth Jacobian (2x1)
  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(std::get<2>(errorTerm.variableJacobians)(i, 0),
                std::get<2>(numericalJacobians)(i, 0), 1e-4)
        << "InverseDepth Jacobian mismatch at (" << i << ", 0)";
  }
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST(AutoDiffStressTest, ManyVariables) {
  // Test with maximum typical variable count
  constexpr int N = 13;  // 2 SE3 (6+6) + 1 InverseDepth (1) = 13

  using JetT = Jet<double, N>;
  Eigen::Matrix<JetT, 3, 1> v;
  v << JetT(1.0, 0), JetT(2.0, 6), JetT(3.0, 12);

  auto sum = v(0) + v(1) + v(2);
  EXPECT_DOUBLE_EQ(sum.a, 6.0);
}

TEST(AutoDiffStressTest, RandomDifferenceErrors) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<> dist(-100.0, 100.0);

  for (int trial = 0; trial < 100; ++trial) {
    VariableContainer<SimpleScalar, DifferentSimpleScalar> variables;
    double v1 = dist(gen);
    double v2 = dist(gen);

    auto k1 = variables.insert(SimpleScalar(v1));
    auto k2 = variables.insert(DifferentSimpleScalar(v2));

    DifferenceErrorTermAutoDiff errorTerm(k1, k2);
    errorTerm.updateVariablePointers(variables);
    errorTerm.evaluate(variables, true);

    auto numericalJacobians = numericallyDifferentiate(errorTerm, variables);

    EXPECT_NEAR(std::get<0>(errorTerm.variableJacobians)(0, 0),
                std::get<0>(numericalJacobians)(0, 0), 1e-5);
    EXPECT_NEAR(std::get<1>(errorTerm.variableJacobians)(0, 0),
                std::get<1>(numericalJacobians)(0, 0), 1e-5);
  }
}

// =============================================================================
// Dimension Calculation Tests
// =============================================================================

TEST(AutoDiffDimensionTest, TotalDimension) {
  using namespace Tangent::internal;

  // Single variable
  EXPECT_EQ((TotalDimension<SimpleScalar>::value), 1);
  EXPECT_EQ((TotalDimension<SE3>::value), 6);

  // Multiple variables
  EXPECT_EQ((TotalDimension<SimpleScalar, SimpleScalar>::value), 2);
  EXPECT_EQ((TotalDimension<SE3, SE3, InverseDepth>::value), 13);
}

TEST(AutoDiffDimensionTest, DimensionOffset) {
  using namespace Tangent::internal;

  // SE3, SE3, InverseDepth -> offsets: 0, 6, 12
  EXPECT_EQ((DimensionOffset<0, SE3, SE3, InverseDepth>::value), 0);
  EXPECT_EQ((DimensionOffset<1, SE3, SE3, InverseDepth>::value), 6);
  EXPECT_EQ((DimensionOffset<2, SE3, SE3, InverseDepth>::value), 12);
}

// =============================================================================
// Multi-Point Validation Tests using ErrorTermValidator
// =============================================================================

/**
 * @brief Test fixture for comprehensive multi-point Jacobian validation.
 *
 * Tests autodiff Jacobians against numerical differentiation at many
 * random test points to ensure correctness across the parameter space.
 */
class MultiPointValidationTest : public ::testing::Test {
 protected:
  std::mt19937 gen{12345};
  std::uniform_real_distribution<> smallAngle{-0.5, 0.5};
  std::uniform_real_distribution<> translation{-5.0, 5.0};
  std::uniform_real_distribution<> positiveValue{0.1, 2.0};
  std::uniform_real_distribution<> scalarValue{-10.0, 10.0};
  std::uniform_real_distribution<> bearing{-0.5, 0.5};

  // Generate random SE3 pose
  Sophus::SE3d randomSE3() {
    Eigen::Vector3d omega(smallAngle(gen), smallAngle(gen), smallAngle(gen));
    Eigen::Vector3d trans(translation(gen), translation(gen), translation(gen));
    return Sophus::SE3d(Sophus::SO3d::exp(omega), trans);
  }

  // Generate random SO3 rotation
  Sophus::SO3d randomSO3() {
    Eigen::Vector3d omega(smallAngle(gen), smallAngle(gen), smallAngle(gen));
    return Sophus::SO3d::exp(omega);
  }
};

TEST_F(MultiPointValidationTest, DifferenceErrorMultiplePoints) {
  constexpr int numTrials = 50;

  for (int trial = 0; trial < numTrials; ++trial) {
    VariableContainer<SimpleScalar, DifferentSimpleScalar> variables;
    auto k1 = variables.insert(SimpleScalar(scalarValue(gen)));
    auto k2 = variables.insert(DifferentSimpleScalar(scalarValue(gen)));

    DifferenceErrorTermAutoDiff errorTerm(k1, k2);
    ErrorTermValidator<DifferenceErrorTermAutoDiff> validator(errorTerm);
    validator.threshold = 1e-5;

    EXPECT_TRUE(validator.validate(variables))
        << "Failed at trial " << trial;
  }
}

TEST_F(MultiPointValidationTest, QuadraticErrorMultiplePoints) {
  constexpr int numTrials = 50;

  for (int trial = 0; trial < numTrials; ++trial) {
    VariableContainer<SimpleScalar> variables;
    double value = scalarValue(gen);
    double target = scalarValue(gen);
    auto k = variables.insert(SimpleScalar(value));

    QuadraticErrorTermAutoDiff errorTerm(k, target);
    ErrorTermValidator<QuadraticErrorTermAutoDiff> validator(errorTerm);
    validator.threshold = 1e-5;

    EXPECT_TRUE(validator.validate(variables))
        << "Failed at trial " << trial << " with value=" << value
        << ", target=" << target;
  }
}

TEST_F(MultiPointValidationTest, ReprojectionErrorMultiplePoints) {
  constexpr int numTrials = 30;

  for (int trial = 0; trial < numTrials; ++trial) {
    VariableContainer<SE3, InverseDepth> variables;

    SE3 host, target;
    host.value = randomSE3();
    target.value = randomSE3();

    auto hostK = variables.insert(host);
    auto targetK = variables.insert(target);
    auto dinvK = variables.insert(InverseDepth(positiveValue(gen)));

    Eigen::Vector2d bearingVec(bearing(gen), bearing(gen));

    auto errorTerm = createAutoDiffReprojectionErrorTerm(
        variables, hostK, targetK, dinvK, bearingVec);

    ErrorTermValidator<RelativeReprojectionErrorAutoDiff> validator(errorTerm);
    validator.threshold = 1e-4;

    EXPECT_TRUE(validator.validate(variables))
        << "Failed at trial " << trial;
  }
}

TEST_F(MultiPointValidationTest, SE3PriorErrorMultiplePoints) {
  constexpr int numTrials = 30;

  for (int trial = 0; trial < numTrials; ++trial) {
    VariableContainer<SE3> variables;

    SE3 pose;
    pose.value = randomSE3();
    auto poseK = variables.insert(pose);

    Sophus::SE3d targetPose = randomSE3();

    SE3PriorErrorAutoDiff errorTerm(poseK, targetPose);
    ErrorTermValidator<SE3PriorErrorAutoDiff> validator(errorTerm);
    validator.threshold = 1e-4;

    EXPECT_TRUE(validator.validate(variables))
        << "Failed at trial " << trial;
  }
}

TEST_F(MultiPointValidationTest, SO3ErrorMultiplePoints) {
  constexpr int numTrials = 30;

  for (int trial = 0; trial < numTrials; ++trial) {
    VariableContainer<SO3> variables;

    SO3 rot;
    rot.value = randomSO3();
    auto rotK = variables.insert(rot);

    Sophus::SO3d targetRot = randomSO3();

    SO3ErrorAutoDiff errorTerm(rotK, targetRot);
    ErrorTermValidator<SO3ErrorAutoDiff> validator(errorTerm);
    validator.threshold = 1e-4;

    EXPECT_TRUE(validator.validate(variables))
        << "Failed at trial " << trial;
  }
}

TEST_F(MultiPointValidationTest, ComplexNonlinearErrorMultiplePoints) {
  constexpr int numTrials = 30;

  for (int trial = 0; trial < numTrials; ++trial) {
    VariableContainer<SE3, InverseDepth, SimpleScalar> variables;

    SE3 sensorPose, targetPose;
    sensorPose.value = randomSE3();
    targetPose.value = randomSE3();

    auto sensorK = variables.insert(sensorPose);
    auto targetK = variables.insert(targetPose);
    auto scaleK = variables.insert(InverseDepth(positiveValue(gen)));
    auto weightK = variables.insert(SimpleScalar(positiveValue(gen)));

    Eigen::Vector3d targetPoint(translation(gen), translation(gen),
                                translation(gen));

    auto errorTerm = createComplexNonlinearErrorTerm(
        variables, sensorK, targetK, scaleK, weightK, targetPoint);

    ErrorTermValidator<ComplexNonlinearErrorAutoDiff> validator(errorTerm);
    validator.threshold = 1e-4;

    EXPECT_TRUE(validator.validate(variables))
        << "Failed at trial " << trial;
  }
}

// =============================================================================
// Comprehensive Jet Operation Tests
// =============================================================================

/**
 * @brief Test all Jet math functions at multiple test points.
 */
class JetMathComprehensiveTest : public ::testing::Test {
 protected:
  std::mt19937 gen{54321};

  // Test a Jet function against numerical derivative
  template <typename Func>
  void testJetFunction(Func func, double minVal, double maxVal, int numPoints) {
    std::uniform_real_distribution<> dist(minVal, maxVal);
    const double delta = 1e-7;

    for (int i = 0; i < numPoints; ++i) {
      double x = dist(gen);
      Jet<double, 1> jetX(x, 0);

      auto result = func(jetX);

      // Numerical derivative
      double fPlus = func(Jet<double, 1>(x + delta)).a;
      double fMinus = func(Jet<double, 1>(x - delta)).a;
      double numericalDeriv = (fPlus - fMinus) / (2 * delta);

      EXPECT_NEAR(result.v(0), numericalDeriv, 1e-5)
          << "Failed for x=" << x;
    }
  }
};

TEST_F(JetMathComprehensiveTest, SinMultiplePoints) {
  testJetFunction([](auto x) { return sin(x); }, -M_PI, M_PI, 50);
}

TEST_F(JetMathComprehensiveTest, CosMultiplePoints) {
  testJetFunction([](auto x) { return cos(x); }, -M_PI, M_PI, 50);
}

TEST_F(JetMathComprehensiveTest, TanMultiplePoints) {
  testJetFunction([](auto x) { return tan(x); }, -1.4, 1.4, 50);
}

TEST_F(JetMathComprehensiveTest, ExpMultiplePoints) {
  testJetFunction([](auto x) { return exp(x); }, -3.0, 3.0, 50);
}

TEST_F(JetMathComprehensiveTest, LogMultiplePoints) {
  testJetFunction([](auto x) { return log(x); }, 0.1, 10.0, 50);
}

TEST_F(JetMathComprehensiveTest, SqrtMultiplePoints) {
  testJetFunction([](auto x) { return sqrt(x); }, 0.01, 10.0, 50);
}

TEST_F(JetMathComprehensiveTest, AsinMultiplePoints) {
  testJetFunction([](auto x) { return asin(x); }, -0.99, 0.99, 50);
}

TEST_F(JetMathComprehensiveTest, AcosMultiplePoints) {
  testJetFunction([](auto x) { return acos(x); }, -0.99, 0.99, 50);
}

TEST_F(JetMathComprehensiveTest, AtanMultiplePoints) {
  testJetFunction([](auto x) { return atan(x); }, -10.0, 10.0, 50);
}

TEST_F(JetMathComprehensiveTest, SinhMultiplePoints) {
  testJetFunction([](auto x) { return sinh(x); }, -3.0, 3.0, 50);
}

TEST_F(JetMathComprehensiveTest, CoshMultiplePoints) {
  testJetFunction([](auto x) { return cosh(x); }, -3.0, 3.0, 50);
}

TEST_F(JetMathComprehensiveTest, TanhMultiplePoints) {
  testJetFunction([](auto x) { return tanh(x); }, -3.0, 3.0, 50);
}

TEST_F(JetMathComprehensiveTest, PowMultiplePoints) {
  testJetFunction([](auto x) { return pow(x, 2.5); }, 0.1, 5.0, 50);
}

TEST_F(JetMathComprehensiveTest, AbsMultiplePoints) {
  // Test positive values
  testJetFunction([](auto x) { return abs(x); }, 0.1, 10.0, 25);
  // Test negative values
  testJetFunction([](auto x) { return abs(x); }, -10.0, -0.1, 25);
}

TEST_F(JetMathComprehensiveTest, Atan2MultiplePoints) {
  std::uniform_real_distribution<> dist(-5.0, 5.0);
  const double delta = 1e-7;

  for (int i = 0; i < 50; ++i) {
    double y = dist(gen);
    double x = dist(gen);

    // Skip near-zero x to avoid instability
    if (std::abs(x) < 0.1) continue;

    Jet<double, 2> jetY(y, 0);
    Jet<double, 2> jetX(x, 1);

    auto result = atan2(jetY, jetX);

    // Numerical derivatives
    double fPlusY = std::atan2(y + delta, x);
    double fMinusY = std::atan2(y - delta, x);
    double derivY = (fPlusY - fMinusY) / (2 * delta);

    double fPlusX = std::atan2(y, x + delta);
    double fMinusX = std::atan2(y, x - delta);
    double derivX = (fPlusX - fMinusX) / (2 * delta);

    EXPECT_NEAR(result.v(0), derivY, 1e-5)
        << "Failed for y=" << y << ", x=" << x;
    EXPECT_NEAR(result.v(1), derivX, 1e-5)
        << "Failed for y=" << y << ", x=" << x;
  }
}

// =============================================================================
// Eigen + Jet Integration Tests
// =============================================================================

TEST(JetEigenIntegrationTest, MatrixMultiplication) {
  using JetT = Jet<double, 4>;

  Eigen::Matrix<JetT, 2, 2> A;
  A << JetT(1.0, 0), JetT(2.0, 1), JetT(3.0, 2), JetT(4.0, 3);

  Eigen::Matrix<JetT, 2, 1> x;
  x << JetT(5.0), JetT(6.0);  // Constants (no derivatives)

  auto result = A * x;

  // result(0) = 1*5 + 2*6 = 17
  // result(1) = 3*5 + 4*6 = 39
  EXPECT_NEAR(result(0).a, 17.0, 1e-10);
  EXPECT_NEAR(result(1).a, 39.0, 1e-10);

  // d(result(0))/dA(0,0) = x(0) = 5
  EXPECT_NEAR(result(0).v(0), 5.0, 1e-10);
  // d(result(0))/dA(0,1) = x(1) = 6
  EXPECT_NEAR(result(0).v(1), 6.0, 1e-10);
}

TEST(JetEigenIntegrationTest, DotProduct) {
  using JetT = Jet<double, 6>;

  Eigen::Matrix<JetT, 3, 1> a, b;
  a << JetT(1.0, 0), JetT(2.0, 1), JetT(3.0, 2);
  b << JetT(4.0, 3), JetT(5.0, 4), JetT(6.0, 5);

  JetT dot = a.dot(b);

  // dot = 1*4 + 2*5 + 3*6 = 32
  EXPECT_NEAR(dot.a, 32.0, 1e-10);

  // d(dot)/d(a(i)) = b(i)
  EXPECT_NEAR(dot.v(0), 4.0, 1e-10);
  EXPECT_NEAR(dot.v(1), 5.0, 1e-10);
  EXPECT_NEAR(dot.v(2), 6.0, 1e-10);

  // d(dot)/d(b(i)) = a(i)
  EXPECT_NEAR(dot.v(3), 1.0, 1e-10);
  EXPECT_NEAR(dot.v(4), 2.0, 1e-10);
  EXPECT_NEAR(dot.v(5), 3.0, 1e-10);
}

TEST(JetEigenIntegrationTest, Norm) {
  using JetT = Jet<double, 3>;

  Eigen::Matrix<JetT, 3, 1> v;
  v << JetT(3.0, 0), JetT(4.0, 1), JetT(0.0, 2);

  // Manual norm computation (norm() may not work directly with Jets)
  JetT normSq = v(0) * v(0) + v(1) * v(1) + v(2) * v(2);
  JetT norm = sqrt(normSq);

  EXPECT_NEAR(norm.a, 5.0, 1e-10);

  // d(norm)/d(v(i)) = v(i) / norm
  EXPECT_NEAR(norm.v(0), 3.0 / 5.0, 1e-10);
  EXPECT_NEAR(norm.v(1), 4.0 / 5.0, 1e-10);
  EXPECT_NEAR(norm.v(2), 0.0 / 5.0, 1e-10);
}

// =============================================================================
// Sophus + Jet Integration Tests
// =============================================================================

TEST(JetSophusIntegrationTest, SE3PointTransformation) {
  SE3 pose;
  pose.value = Sophus::SE3d::rotZ(0.1) * Sophus::SE3d::trans(1, 2, 3);

  auto liftedPose = liftSE3<double, 6>(pose, 0);

  using JetT = Jet<double, 6>;
  Eigen::Matrix<JetT, 3, 1> point;
  point << JetT(1.0), JetT(0.0), JetT(0.0);  // Point with no derivatives

  auto transformed = liftedPose * point;

  // The transformed point should have derivatives w.r.t. pose parameters
  // Check that derivatives are non-zero (transformation depends on pose)
  bool hasNonZeroDerivs = false;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 6; ++j) {
      if (std::abs(transformed(i).v(j)) > 1e-10) {
        hasNonZeroDerivs = true;
      }
    }
  }
  EXPECT_TRUE(hasNonZeroDerivs);
}

TEST(JetSophusIntegrationTest, SE3Inverse) {
  SE3 pose;
  pose.value = Sophus::SE3d::rotY(0.2) * Sophus::SE3d::trans(1, 2, 3);

  auto liftedPose = liftSE3<double, 6>(pose, 0);
  auto liftedInverse = liftedPose.inverse();

  // T * T^-1 should be identity
  auto product = liftedPose * liftedInverse;

  // Check rotation is identity
  auto R = product.rotationMatrix();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double expected = (i == j) ? 1.0 : 0.0;
      EXPECT_NEAR(R(i, j).a, expected, 1e-10);
    }
  }

  // Check translation is zero
  auto t = product.translation();
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(t(i).a, 0.0, 1e-10);
  }
}

TEST(JetSophusIntegrationTest, SO3LogMap) {
  SO3 rot;
  rot.value = Sophus::SO3d::exp(Eigen::Vector3d(0.1, 0.2, 0.3));

  auto liftedRot = liftSO3<double, 3>(rot, 0);
  auto logVec = liftedRot.log();

  // The log should approximately equal the original rotation vector
  EXPECT_NEAR(logVec(0).a, 0.1, 1e-10);
  EXPECT_NEAR(logVec(1).a, 0.2, 1e-10);
  EXPECT_NEAR(logVec(2).a, 0.3, 1e-10);

  // At identity, d(log)/d(omega) should be approximately identity
  // For small rotations, this should be close
}
