#pragma once

#include <Eigen/Core>
#include <cmath>
#include <ostream>

namespace Tangent {

/**
 * @brief Dual number for automatic differentiation.
 *
 * Jet<T, N> represents a value with N partial derivatives.
 * For a scalar function f(x1, x2, ..., xn), a Jet stores:
 *   - a: the value f
 *   - v: the gradient [df/dx1, df/dx2, ..., df/dxn]
 *
 * Jets propagate derivatives through arithmetic operations and mathematical
 * functions using the chain rule, enabling automatic computation of Jacobians.
 *
 * @tparam T The underlying scalar type (typically double)
 * @tparam N The number of derivatives to track
 *
 * Example usage:
 * @code
 * // Create Jets with derivative seeding
 * Jet<double, 2> x(3.0, 0);  // x = 3, dx/dx = 1, dx/dy = 0
 * Jet<double, 2> y(4.0, 1);  // y = 4, dy/dx = 0, dy/dy = 1
 *
 * auto f = x * y + sin(x);   // f = 12 + sin(3)
 * // f.a = 12.1411...
 * // f.v = [y + cos(x), x] = [4 + cos(3), 3] = [3.01, 3]
 * @endcode
 */
template <typename T, int N>
struct Jet {
  T a;                         // Scalar value
  Eigen::Matrix<T, N, 1> v;    // Derivative vector

  // Default constructor - zero value and derivatives
  Jet() : a(T(0)) { v.setZero(); }

  // Construct from scalar value only (no derivatives)
  explicit Jet(const T& value) : a(value) { v.setZero(); }

  // Construct with value and single seeded derivative
  Jet(const T& value, int derivative_index) : a(value) {
    v.setZero();
    if (derivative_index >= 0 && derivative_index < N) {
      v(derivative_index) = T(1);
    }
  }

  // Construct with value and full derivative vector
  Jet(const T& value, const Eigen::Matrix<T, N, 1>& derivatives)
      : a(value), v(derivatives) {}

  // Copy constructor
  Jet(const Jet& other) = default;

  // Assignment operator
  Jet& operator=(const Jet& other) = default;

  // Assignment from scalar (clears derivatives)
  Jet& operator=(const T& scalar) {
    a = scalar;
    v.setZero();
    return *this;
  }

  // Binary arithmetic operators (Jet op Jet)
  Jet operator+(const Jet& other) const {
    return Jet(a + other.a, v + other.v);
  }

  Jet operator-(const Jet& other) const {
    return Jet(a - other.a, v - other.v);
  }

  Jet operator*(const Jet& other) const {
    // Product rule: d(f*g) = f*dg + g*df
    return Jet(a * other.a, a * other.v + other.a * v);
  }

  Jet operator/(const Jet& other) const {
    // Quotient rule: d(f/g) = (g*df - f*dg) / g^2
    const T inv = T(1) / other.a;
    return Jet(a * inv, (v - a * inv * other.v) * inv);
  }

  // Unary operators
  Jet operator-() const { return Jet(-a, -v); }

  Jet operator+() const { return *this; }

  // Compound assignment operators (Jet op= Jet)
  Jet& operator+=(const Jet& other) {
    a += other.a;
    v += other.v;
    return *this;
  }

  Jet& operator-=(const Jet& other) {
    a -= other.a;
    v -= other.v;
    return *this;
  }

  Jet& operator*=(const Jet& other) {
    *this = *this * other;
    return *this;
  }

  Jet& operator/=(const Jet& other) {
    *this = *this / other;
    return *this;
  }

  // Binary arithmetic operators (Jet op scalar)
  Jet operator+(const T& s) const { return Jet(a + s, v); }

  Jet operator-(const T& s) const { return Jet(a - s, v); }

  Jet operator*(const T& s) const { return Jet(a * s, v * s); }

  Jet operator/(const T& s) const {
    const T inv = T(1) / s;
    return Jet(a * inv, v * inv);
  }

  // Compound assignment operators (Jet op= scalar)
  Jet& operator+=(const T& s) {
    a += s;
    return *this;
  }

  Jet& operator-=(const T& s) {
    a -= s;
    return *this;
  }

  Jet& operator*=(const T& s) {
    a *= s;
    v *= s;
    return *this;
  }

  Jet& operator/=(const T& s) {
    const T inv = T(1) / s;
    a *= inv;
    v *= inv;
    return *this;
  }

  // Comparison operators (compare values only, derivatives ignored)
  bool operator<(const Jet& other) const { return a < other.a; }
  bool operator>(const Jet& other) const { return a > other.a; }
  bool operator<=(const Jet& other) const { return a <= other.a; }
  bool operator>=(const Jet& other) const { return a >= other.a; }
  bool operator==(const Jet& other) const { return a == other.a; }
  bool operator!=(const Jet& other) const { return a != other.a; }

  // Comparison with scalar
  bool operator<(const T& s) const { return a < s; }
  bool operator>(const T& s) const { return a > s; }
  bool operator<=(const T& s) const { return a <= s; }
  bool operator>=(const T& s) const { return a >= s; }
  bool operator==(const T& s) const { return a == s; }
  bool operator!=(const T& s) const { return a != s; }
};

// Left-hand scalar operators (scalar op Jet)
template <typename T, int N>
Jet<T, N> operator+(const T& s, const Jet<T, N>& j) {
  return Jet<T, N>(s + j.a, j.v);
}

template <typename T, int N>
Jet<T, N> operator-(const T& s, const Jet<T, N>& j) {
  return Jet<T, N>(s - j.a, -j.v);
}

template <typename T, int N>
Jet<T, N> operator*(const T& s, const Jet<T, N>& j) {
  return Jet<T, N>(s * j.a, s * j.v);
}

template <typename T, int N>
Jet<T, N> operator/(const T& s, const Jet<T, N>& j) {
  // d(s/f) = -s * df / f^2
  const T inv = T(1) / j.a;
  return Jet<T, N>(s * inv, -s * inv * inv * j.v);
}

// Comparison: scalar vs Jet
template <typename T, int N>
bool operator<(const T& s, const Jet<T, N>& j) { return s < j.a; }

template <typename T, int N>
bool operator>(const T& s, const Jet<T, N>& j) { return s > j.a; }

template <typename T, int N>
bool operator<=(const T& s, const Jet<T, N>& j) { return s <= j.a; }

template <typename T, int N>
bool operator>=(const T& s, const Jet<T, N>& j) { return s >= j.a; }

// ============================================================================
// Mathematical functions (using chain rule)
// ============================================================================

template <typename T, int N>
Jet<T, N> abs(const Jet<T, N>& j) {
  // d|f| = sign(f) * df
  return j.a >= T(0) ? j : -j;
}

template <typename T, int N>
Jet<T, N> sqrt(const Jet<T, N>& j) {
  // d(sqrt(f)) = df / (2 * sqrt(f))
  const T sqrtVal = std::sqrt(j.a);
  return Jet<T, N>(sqrtVal, j.v / (T(2) * sqrtVal));
}

template <typename T, int N>
Jet<T, N> sin(const Jet<T, N>& j) {
  // d(sin(f)) = cos(f) * df
  return Jet<T, N>(std::sin(j.a), std::cos(j.a) * j.v);
}

template <typename T, int N>
Jet<T, N> cos(const Jet<T, N>& j) {
  // d(cos(f)) = -sin(f) * df
  return Jet<T, N>(std::cos(j.a), -std::sin(j.a) * j.v);
}

template <typename T, int N>
Jet<T, N> tan(const Jet<T, N>& j) {
  // d(tan(f)) = df / cos^2(f) = (1 + tan^2(f)) * df
  const T tanVal = std::tan(j.a);
  const T sec2 = T(1) + tanVal * tanVal;
  return Jet<T, N>(tanVal, sec2 * j.v);
}

template <typename T, int N>
Jet<T, N> asin(const Jet<T, N>& j) {
  // d(asin(f)) = df / sqrt(1 - f^2)
  return Jet<T, N>(std::asin(j.a), j.v / std::sqrt(T(1) - j.a * j.a));
}

template <typename T, int N>
Jet<T, N> acos(const Jet<T, N>& j) {
  // d(acos(f)) = -df / sqrt(1 - f^2)
  return Jet<T, N>(std::acos(j.a), -j.v / std::sqrt(T(1) - j.a * j.a));
}

template <typename T, int N>
Jet<T, N> atan(const Jet<T, N>& j) {
  // d(atan(f)) = df / (1 + f^2)
  return Jet<T, N>(std::atan(j.a), j.v / (T(1) + j.a * j.a));
}

template <typename T, int N>
Jet<T, N> atan2(const Jet<T, N>& y, const Jet<T, N>& x) {
  // d(atan2(y,x)) = (x*dy - y*dx) / (x^2 + y^2)
  const T denom = x.a * x.a + y.a * y.a;
  return Jet<T, N>(std::atan2(y.a, x.a), (x.a * y.v - y.a * x.v) / denom);
}

template <typename T, int N>
Jet<T, N> exp(const Jet<T, N>& j) {
  // d(exp(f)) = exp(f) * df
  const T expVal = std::exp(j.a);
  return Jet<T, N>(expVal, expVal * j.v);
}

template <typename T, int N>
Jet<T, N> log(const Jet<T, N>& j) {
  // d(log(f)) = df / f
  return Jet<T, N>(std::log(j.a), j.v / j.a);
}

template <typename T, int N>
Jet<T, N> log10(const Jet<T, N>& j) {
  // d(log10(f)) = df / (f * ln(10))
  const T ln10 = std::log(T(10));
  return Jet<T, N>(std::log10(j.a), j.v / (j.a * ln10));
}

template <typename T, int N>
Jet<T, N> pow(const Jet<T, N>& base, const T& exponent) {
  // d(f^n) = n * f^(n-1) * df
  const T val = std::pow(base.a, exponent);
  return Jet<T, N>(val, exponent * std::pow(base.a, exponent - T(1)) * base.v);
}

template <typename T, int N>
Jet<T, N> pow(const Jet<T, N>& base, const Jet<T, N>& exponent) {
  // d(f^g) = f^g * (g' * ln(f) + g * f' / f)
  const T val = std::pow(base.a, exponent.a);
  const T logBase = std::log(base.a);
  return Jet<T, N>(val,
                   val * (exponent.v * logBase + exponent.a * base.v / base.a));
}

template <typename T, int N>
Jet<T, N> pow(const T& base, const Jet<T, N>& exponent) {
  // d(c^g) = c^g * ln(c) * dg
  const T val = std::pow(base, exponent.a);
  return Jet<T, N>(val, val * std::log(base) * exponent.v);
}

template <typename T, int N>
Jet<T, N> sinh(const Jet<T, N>& j) {
  // d(sinh(f)) = cosh(f) * df
  return Jet<T, N>(std::sinh(j.a), std::cosh(j.a) * j.v);
}

template <typename T, int N>
Jet<T, N> cosh(const Jet<T, N>& j) {
  // d(cosh(f)) = sinh(f) * df
  return Jet<T, N>(std::cosh(j.a), std::sinh(j.a) * j.v);
}

template <typename T, int N>
Jet<T, N> tanh(const Jet<T, N>& j) {
  // d(tanh(f)) = (1 - tanh^2(f)) * df
  const T tanhVal = std::tanh(j.a);
  return Jet<T, N>(tanhVal, (T(1) - tanhVal * tanhVal) * j.v);
}

// floor and ceil (derivatives are zero almost everywhere)
template <typename T, int N>
Jet<T, N> floor(const Jet<T, N>& j) {
  return Jet<T, N>(std::floor(j.a));  // Derivatives are zero
}

template <typename T, int N>
Jet<T, N> ceil(const Jet<T, N>& j) {
  return Jet<T, N>(std::ceil(j.a));  // Derivatives are zero
}

// isfinite, isinf, isnan for compatibility
template <typename T, int N>
bool isfinite(const Jet<T, N>& j) {
  return std::isfinite(j.a);
}

template <typename T, int N>
bool isinf(const Jet<T, N>& j) {
  return std::isinf(j.a);
}

template <typename T, int N>
bool isnan(const Jet<T, N>& j) {
  return std::isnan(j.a);
}

// Output stream operator for debugging
template <typename T, int N>
std::ostream& operator<<(std::ostream& os, const Jet<T, N>& j) {
  os << "[" << j.a << "; " << j.v.transpose() << "]";
  return os;
}

}  // namespace Tangent

// ============================================================================
// ADL hooks for Sophus compatibility
// ============================================================================
// Sophus calls math functions using ADL (Argument Dependent Lookup).
// For Jet<T, N> arguments, ADL will search the Tangent namespace where Jet
// is defined, finding our overloads automatically.
//
// We also need to add `using std::xxx` declarations in the Tangent namespace
// so that when Sophus does `using std::sqrt; sqrt(x)`, it can still find
// our overloads through ADL.

namespace Tangent {

// Bring std math functions into Tangent namespace so that expressions like
// `using std::sqrt; sqrt(jet)` work correctly - they'll find Tangent::sqrt
// through ADL on the Jet argument.
using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::ceil;
using std::cos;
using std::cosh;
using std::exp;
using std::floor;
using std::isfinite;
using std::isinf;
using std::isnan;
using std::log;
using std::log10;
using std::pow;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;

}  // namespace Tangent
