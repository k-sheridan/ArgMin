#pragma once

#include "tangent/ErrorTerms/AutoDiffErrorTerm.h"
#include "tangent/Variables/InverseDepth.h"
#include "tangent/Variables/SE3.h"
#include "tangent/Variables/SimpleScalar.h"
#include "tangent/Variables/SO3.h"

#include "TestUtils.h"

namespace Tangent::Test {

// =============================================================================
// AutoDiff version of DifferenceErrorTerm
// =============================================================================

/**
 * @brief AutoDiff version of the simple difference error term.
 *
 * Computes the difference between two scalar variables: var2 - var1.
 * This is the simplest possible autodiff error term for testing.
 */
class DifferenceErrorTermAutoDiff
    : public AutoDiffErrorTerm<DifferenceErrorTermAutoDiff, double, 1,
                               SimpleScalar, DifferentSimpleScalar> {
 public:
  DifferenceErrorTermAutoDiff(VariableKey<SimpleScalar> key1,
                              VariableKey<DifferentSimpleScalar> key2) {
    std::get<0>(variableKeys) = key1;
    std::get<1>(variableKeys) = key2;
    information.setIdentity();
  }

  /**
   * @brief Compute the difference error.
   *
   * @tparam T Scalar type (double for value, JetType for autodiff)
   * @tparam Scalar1 Type of first variable (double or Jet)
   * @tparam Scalar2 Type of second variable (double or Jet)
   */
  template <typename T, typename Scalar1, typename Scalar2>
  Eigen::Matrix<T, 1, 1> computeError(const Scalar1& var1,
                                      const Scalar2& var2) const {
    Eigen::Matrix<T, 1, 1> error;
    error(0) = var2 - var1;
    return error;
  }
};

// =============================================================================
// AutoDiff version of RelativeReprojectionError
// =============================================================================

/**
 * @brief AutoDiff version of the relative reprojection error.
 *
 * Models the reprojection of a 3D point from a host frame to a target frame.
 * The point is parameterized by its inverse depth in the host frame and
 * a bearing vector (normalized image coordinates).
 *
 * Residual = measurement - project(T_target^-1 * T_host * unproject(bearing,
 * dinv))
 */
class RelativeReprojectionErrorAutoDiff
    : public AutoDiffErrorTerm<RelativeReprojectionErrorAutoDiff, double, 2,
                               SE3, SE3, InverseDepth> {
 public:
  Eigen::Vector2d bearing;  // Bearing in host frame
  Eigen::Vector2d z;        // Measurement in target frame

  RelativeReprojectionErrorAutoDiff(VariableKey<SE3> hostFrame,
                                    VariableKey<SE3> targetFrame,
                                    VariableKey<InverseDepth> dinv,
                                    Eigen::Vector2d bearingMeasurement,
                                    Eigen::Vector2d bearingInHost) {
    std::get<0>(variableKeys) = hostFrame;
    std::get<1>(variableKeys) = targetFrame;
    std::get<2>(variableKeys) = dinv;
    z = bearingMeasurement;
    bearing = bearingInHost;
    information.setIdentity();
  }

  /**
   * @brief Compute the reprojection error.
   *
   * This single templated method works for both:
   * - T = double (residual-only computation, fast path)
   * - T = Jet<double, N> (autodiff with Jacobians)
   *
   * Both paths use Sophus directly - Sophus::SE3d for double and
   * Sophus::SE3<Jet> for autodiff.
   *
   * @tparam T Scalar type
   * @tparam Host Type of host pose (Sophus::SE3<T>)
   * @tparam Target Type of target pose (Sophus::SE3<T>)
   * @tparam Depth Type of inverse depth (double or Jet)
   */
  template <typename T, typename Host, typename Target, typename Depth>
  Eigen::Matrix<T, 2, 1> computeError(const Host& host, const Target& target,
                                      const Depth& dinv) const {
    // Unproject point in host frame: p = [bx/d, by/d, 1/d]
    Eigen::Matrix<T, 3, 1> pointInHost;
    pointInHost(0) = T(bearing(0)) / dinv;
    pointInHost(1) = T(bearing(1)) / dinv;
    pointInHost(2) = T(1.0) / dinv;

    // Transform to world then to target: p_target = T_target^-1 * T_host * p
    // Both Sophus::SE3d and Sophus::SE3<Jet> support operator* for points
    Eigen::Matrix<T, 3, 1> pointInTarget = target.inverse() * (host * pointInHost);

    // Error = measurement - projection
    Eigen::Matrix<T, 2, 1> error;
    error(0) = T(z(0)) - pointInTarget(0) / pointInTarget(2);
    error(1) = T(z(1)) - pointInTarget(1) / pointInTarget(2);

    return error;
  }
};

/**
 * @brief Helper to create a RelativeReprojectionErrorAutoDiff with correct
 * measurement.
 *
 * Computes the expected measurement from the current variable values so that
 * the initial error is zero.
 */
template <typename VariableContainerType>
RelativeReprojectionErrorAutoDiff createAutoDiffReprojectionErrorTerm(
    VariableContainerType& variableContainer, VariableKey<SE3> hostK,
    VariableKey<SE3> targetK, VariableKey<InverseDepth> dinvK,
    Eigen::Vector2d bearing) {
  const auto& dinv = variableContainer.at(dinvK).value;
  Eigen::Vector3d p =
      (variableContainer.at(targetK).value.inverse() *
       variableContainer.at(hostK).value *
       Eigen::Vector3d(bearing(0) / dinv, bearing(1) / dinv, 1 / dinv));
  Eigen::Vector2d z(p(0) / p(2), p(1) / p(2));
  return RelativeReprojectionErrorAutoDiff(hostK, targetK, dinvK, z, bearing);
}

// =============================================================================
// Simple quadratic error term for testing
// =============================================================================

/**
 * @brief A simple quadratic error term for testing: error = (x - target)^2
 *
 * This has a known analytical Jacobian: J = 2 * (x - target)
 */
class QuadraticErrorTermAutoDiff
    : public AutoDiffErrorTerm<QuadraticErrorTermAutoDiff, double, 1,
                               SimpleScalar> {
 public:
  double target;

  QuadraticErrorTermAutoDiff(VariableKey<SimpleScalar> key, double targetValue)
      : target(targetValue) {
    std::get<0>(variableKeys) = key;
    information.setIdentity();
  }

  template <typename T, typename Scalar>
  Eigen::Matrix<T, 1, 1> computeError(const Scalar& x) const {
    Eigen::Matrix<T, 1, 1> error;
    T diff = x - T(target);
    error(0) = diff * diff;
    return error;
  }
};

// =============================================================================
// SE3 only error term (pose prior)
// =============================================================================

/**
 * @brief A pose prior error term that penalizes deviation from a target pose.
 *
 * Error = log(T_target^-1 * T)
 * For small perturbations, this is approximately the 6-vector [omega, v].
 */
class SE3PriorErrorAutoDiff
    : public AutoDiffErrorTerm<SE3PriorErrorAutoDiff, double, 6, SE3> {
 public:
  Sophus::SE3d target;

  SE3PriorErrorAutoDiff(VariableKey<SE3> key, const Sophus::SE3d& targetPose)
      : target(targetPose) {
    std::get<0>(variableKeys) = key;
    information.setIdentity();
  }

  template <typename T, typename Pose>
  Eigen::Matrix<T, 6, 1> computeError(const Pose& pose) const {
    // Compute diff = target^-1 * pose and return log(diff)
    // Cast target to T scalar type (same pattern as other error terms)
    auto diff = target.template cast<T>().inverse() * pose;
    return diff.log();
  }
};

// =============================================================================
// Complex error term exercising many Jet operations
// =============================================================================

/**
 * @brief A complex error term designed to exercise the full Jet autodiff system.
 *
 * This error term combines multiple mathematical operations to verify that
 * the Jet implementation correctly handles:
 * - Basic arithmetic: +, -, *, /
 * - Trigonometric functions: sin, cos, atan2
 * - Exponential/logarithmic: exp, log, sqrt
 * - Absolute value: abs
 * - Eigen matrix operations: dot product, cross product, norm
 * - Sophus operations: SE3 multiplication, rotation, translation
 *
 * The error models a synthetic "range-bearing" sensor with nonlinear effects:
 * - Range measurement with log scaling
 * - Bearing measurement with atan2
 * - Orientation error using rotation matrix operations
 *
 * Residual dimension: 5
 * - [0]: log(range) error with exponential weighting
 * - [1]: azimuth angle error (atan2)
 * - [2]: elevation angle error (atan2)
 * - [3]: rotation trace error (uses cos implicitly via trace)
 * - [4]: combined nonlinear term using sin, cos, sqrt, abs
 */
class ComplexNonlinearErrorAutoDiff
    : public AutoDiffErrorTerm<ComplexNonlinearErrorAutoDiff, double, 5,
                               SE3, SE3, InverseDepth, SimpleScalar> {
 public:
  Eigen::Vector3d targetPoint;     // Target point in world frame
  double rangeRef;                 // Reference range for log scaling
  Eigen::Vector2d bearingRef;      // Reference bearing (azimuth, elevation)
  double rotationTraceRef;         // Reference rotation trace

  ComplexNonlinearErrorAutoDiff(
      VariableKey<SE3> sensorPoseKey,
      VariableKey<SE3> targetPoseKey,
      VariableKey<InverseDepth> scaleKey,
      VariableKey<SimpleScalar> weightKey,
      const Eigen::Vector3d& target,
      double refRange,
      const Eigen::Vector2d& refBearing,
      double refTrace)
      : targetPoint(target),
        rangeRef(refRange),
        bearingRef(refBearing),
        rotationTraceRef(refTrace) {
    std::get<0>(variableKeys) = sensorPoseKey;
    std::get<1>(variableKeys) = targetPoseKey;
    std::get<2>(variableKeys) = scaleKey;
    std::get<3>(variableKeys) = weightKey;
    information.setIdentity();
  }

  /**
   * @brief Compute the complex nonlinear error.
   *
   * This method exercises a wide variety of Jet operations to ensure
   * comprehensive coverage of the autodiff system.
   */
  template <typename T, typename SensorPose, typename TargetPose,
            typename Scale, typename Weight>
  Eigen::Matrix<T, 5, 1> computeError(const SensorPose& sensorPose,
                                       const TargetPose& targetPose,
                                       const Scale& scale,
                                       const Weight& weight) const {
    Eigen::Matrix<T, 5, 1> error;

    // Transform target point to sensor frame
    // Tests: SE3 multiplication, inverse, point transformation
    Eigen::Matrix<T, 3, 1> targetInWorld;
    targetInWorld(0) = T(targetPoint(0));
    targetInWorld(1) = T(targetPoint(1));
    targetInWorld(2) = T(targetPoint(2));

    Eigen::Matrix<T, 3, 1> pointInSensor =
        sensorPose.inverse() * (targetPose * targetInWorld);

    // Compute range with scale factor
    // Tests: sqrt (via norm computation), multiplication, division
    T x = pointInSensor(0);
    T y = pointInSensor(1);
    T z = pointInSensor(2);
    T range = sqrt(x * x + y * y + z * z) / scale;

    // Error 0: Log-scaled range error with exponential weighting
    // Tests: log, exp, abs
    T logRange = log(abs(range) + T(0.001));  // Add small epsilon for stability
    T logRangeRef = T(std::log(rangeRef + 0.001));
    T rangeError = logRange - logRangeRef;
    error(0) = rangeError * exp(-abs(rangeError) * T(0.1));  // Soft saturation

    // Error 1: Azimuth bearing error
    // Tests: atan2
    T azimuth = atan2(y, x);
    error(1) = azimuth - T(bearingRef(0));

    // Error 2: Elevation bearing error
    // Tests: atan2, sqrt
    T xyNorm = sqrt(x * x + y * y + T(1e-10));  // Small epsilon for stability
    T elevation = atan2(z, xyNorm);
    error(2) = elevation - T(bearingRef(1));

    // Error 3: Rotation consistency error (trace of relative rotation)
    // Tests: Sophus rotation matrix access, matrix trace computation
    auto relativeRotation = sensorPose.so3().inverse() * targetPose.so3();
    auto R = relativeRotation.matrix();
    T trace = R(0, 0) + R(1, 1) + R(2, 2);
    error(3) = trace - T(rotationTraceRef);

    // Error 4: Combined nonlinear term
    // Tests: sin, cos, sqrt, abs, multiplication, addition
    // Creates a complex dependency on all variables
    T sinAz = sin(azimuth);
    T cosAz = cos(azimuth);
    T sinEl = sin(elevation);
    T cosEl = cos(elevation);

    // Spherical harmonic-like term
    T sphericalTerm = sinAz * cosEl + cosAz * sinEl * T(0.5);

    // Range-weighted orientation term
    T orientationTerm = sqrt(abs(trace - T(1.0)) + T(0.01));

    // Combine with weight variable
    error(4) = weight * (sphericalTerm * orientationTerm);

    return error;
  }
};

/**
 * @brief Helper to create ComplexNonlinearErrorAutoDiff with zero initial error.
 *
 * Computes reference values from current variable state.
 */
template <typename VariableContainerType>
ComplexNonlinearErrorAutoDiff createComplexNonlinearErrorTerm(
    VariableContainerType& variables,
    VariableKey<SE3> sensorK,
    VariableKey<SE3> targetK,
    VariableKey<InverseDepth> scaleK,
    VariableKey<SimpleScalar> weightK,
    const Eigen::Vector3d& targetPoint) {

  const auto& sensorPose = variables.at(sensorK).value;
  const auto& targetPose = variables.at(targetK).value;
  const auto& scale = variables.at(scaleK).value;

  // Compute point in sensor frame
  Eigen::Vector3d pointInSensor =
      sensorPose.inverse() * (targetPose * targetPoint);

  // Compute reference values
  double x = pointInSensor(0);
  double y = pointInSensor(1);
  double z = pointInSensor(2);
  double range = std::sqrt(x * x + y * y + z * z) / scale;

  double azimuth = std::atan2(y, x);
  double xyNorm = std::sqrt(x * x + y * y + 1e-10);
  double elevation = std::atan2(z, xyNorm);

  auto relRot = sensorPose.so3().inverse() * targetPose.so3();
  auto R = relRot.matrix();
  double trace = R(0, 0) + R(1, 1) + R(2, 2);

  return ComplexNonlinearErrorAutoDiff(
      sensorK, targetK, scaleK, weightK,
      targetPoint, range, Eigen::Vector2d(azimuth, elevation), trace);
}

// =============================================================================
// SO3-only error term for rotation testing
// =============================================================================

/**
 * @brief An SO3 error term exercising rotation-specific Jet operations.
 *
 * Tests SO3 operations: matrix multiplication, inverse, log map, hat operator.
 */
class SO3ErrorAutoDiff
    : public AutoDiffErrorTerm<SO3ErrorAutoDiff, double, 3, SO3> {
 public:
  Sophus::SO3d target;

  SO3ErrorAutoDiff(VariableKey<SO3> key, const Sophus::SO3d& targetRot)
      : target(targetRot) {
    std::get<0>(variableKeys) = key;
    information.setIdentity();
  }

  template <typename T, typename Rotation>
  Eigen::Matrix<T, 3, 1> computeError(const Rotation& rot) const {
    auto diff = target.template cast<T>().inverse() * rot;
    return diff.log();
  }
};

}  // namespace Tangent::Test
