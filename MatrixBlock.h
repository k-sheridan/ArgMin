#pragma once

#include <Eigen/Core>

namespace LittleOptimizer {

template<typename Scalar, size_t rows, size_t cols>
class MatrixBlock : public Eigen::Matrix<Scalar, rows, cols> {

public:

bool zero = true; // tracks whether the sub matrix is zero or not.

};
}