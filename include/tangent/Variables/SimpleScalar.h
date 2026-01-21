#pragma once

#include "tangent/Variables/OptimizableVariable.h"

namespace Tangent
{

class SimpleScalar : public Tangent::OptimizableVariable<double, 1>
{
public:

    double value;

    SimpleScalar() = default;

    SimpleScalar(double val) : value(val)
    {
    }

    void update(const Eigen::Matrix<double, 1, 1> &dx)
    {
        value += dx(0, 0);
    }
};

} // namespace Tangent
