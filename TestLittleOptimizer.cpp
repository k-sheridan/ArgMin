#include "gtest/gtest.h"
#include "LittleOptimizer.h"
#include "SparseBlockMatrix.h"
#include "SE3.h"
#include "InverseDepth.h"

TEST(SparseBlockMatrix, Basic){
    LittleOptimizer::SE3 pose;
    LittleOptimizer::InverseDepth zinv;

    using ScalarType = LittleOptimizer::Scalar<double>;
    using VariableTypes = LittleOptimizer::VariableGroup<LittleOptimizer::SE3, LittleOptimizer::InverseDepth>;

    LittleOptimizer::SparseBlockMatrix::SparseBlockMatrix< ScalarType, VariableTypes > matrix;
    LittleOptimizer::SparseBlockMatrix::SparseBlockMatrix< ScalarType, VariableTypes > matrix2;

    using SE3Key = LittleOptimizer::TypedIndex<LittleOptimizer::SE3>;
    using DinvKey = LittleOptimizer::TypedIndex<LittleOptimizer::InverseDepth>;

    SE3Key k1;
    k1.index = 0;
    DinvKey k2;
    k2.index = 0;

    matrix += matrix2;

    matrix.addRow<LittleOptimizer::SE3>();
    matrix.addRow<LittleOptimizer::InverseDepth>();
    matrix.addColumn<LittleOptimizer::SE3>();
    matrix.addColumn<LittleOptimizer::InverseDepth>();


    std::cout << matrix.rowsOfType<LittleOptimizer::SE3>() << ", " << matrix.columnsOfType<LittleOptimizer::SE3>() << std::endl;

    LittleOptimizer::MatrixBlock<double, 6, 1> matrixBlock = matrix.get(k1, k2);
    LittleOptimizer::MatrixBlock<double, 6, 6> matrixBlock2 = matrix.get(k1, k1);

    matrix.removeColumn(k1);
    matrix.removeColumn(k2);

    matrix.clear();

}

TEST(SparseBlockRow, Basic){
    LittleOptimizer::SE3 pose;
    LittleOptimizer::InverseDepth zinv;

    using ScalarType = LittleOptimizer::Scalar<double>;
    using VariableTypes = LittleOptimizer::VariableGroup<LittleOptimizer::SE3, LittleOptimizer::InverseDepth>;

    LittleOptimizer::SparseBlockMatrix::Row< ScalarType, LittleOptimizer::Dimension<1>, VariableTypes > vector;
    LittleOptimizer::SparseBlockMatrix::Row< ScalarType, LittleOptimizer::Dimension<1>, VariableTypes > vector2;

    using SE3Key = LittleOptimizer::TypedIndex<LittleOptimizer::SE3>;
    using DinvKey = LittleOptimizer::TypedIndex<LittleOptimizer::InverseDepth>;

    SE3Key k1;
    k1.index = 0;
    DinvKey k2;
    k2.index = 0;

    vector.addColumn<LittleOptimizer::SE3>();
    vector.addColumn<LittleOptimizer::InverseDepth>();

    vector2.addColumn<LittleOptimizer::SE3>();
    vector2.addColumn<LittleOptimizer::InverseDepth>();

    vector += vector2;

    EXPECT_EQ(vector.dot(vector2)(0, 0), 0);

    vector.getColumn(k1)(0, 0) = 1;
    vector2.getColumn(k1)(0, 0) = 2;
    vector.getColumn(k2)(0, 0) = 1;
    vector2.getColumn(k2)(0, 0) = 2;

    EXPECT_EQ(vector.dot(vector2)(0, 0), 4);

    vector.removeColumn(k1);
    vector.removeColumn(k2);

    vector.clear();

}

TEST(LittleOptimizer, Basic){
    LittleOptimizer::SE3 pose;
    LittleOptimizer::InverseDepth zinv;

    LittleOptimizer::Optimizer<LittleOptimizer::VariableGroup<LittleOptimizer::SE3, LittleOptimizer::InverseDepth>, LittleOptimizer::ErrorTermGroup<LittleOptimizer::SE3, LittleOptimizer::InverseDepth>> optimizer;
}