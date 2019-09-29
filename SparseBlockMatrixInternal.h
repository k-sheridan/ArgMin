#pragma once

#include <tuple>
#include <utility>
#include <memory>
#include <type_traits>
#include <vector>

#include "MetaHelpers.h"

namespace LittleOptimizer::SparseBlockMatrix
{

template <typename... T>
class Row;

template <typename... Variables>
class SparseBlockMatrix;

namespace internal
{

/// Lowest level. add together the individual matrix blocks.
template <typename T>
bool add_rhs_column_matrices_to_lhs_column_matrices(std::vector<T> &lhsColumnMatrices, std::vector<T> &rhsColumnMatrices)
{
    assert(lhsColumnMatrices.size() == rhsColumnMatrices.size());

    for (size_t idx = 0; idx < lhsColumnMatrices.size(); ++idx)
    {
        auto& lhs = lhsColumnMatrices.at(idx);
        auto& rhs = rhsColumnMatrices.at(idx);

        lhs += rhs;
    }
    return true;
}

template <typename T, size_t... Is, typename... Variables>
bool add_rhs_row_to_lhs_row(T &lhsRow, T &rhsRow, std::integer_sequence<size_t, Is...>)
{
    auto l = {add_rhs_column_matrices_to_lhs_column_matrices(std::get<Is>(lhsRow.columns), std::get<Is>(rhsRow.columns))...};
    (void)l;

    return true;
}

template <typename T, size_t... Is, typename... Variables>
bool add_rhs_row_to_lhs_row(std::vector<T> &lhsRow, std::vector<T> &rhsRow, std::integer_sequence<size_t, Is...>, VariableGroup<Variables...> vars)
{
    assert(lhsRow.size() == rhsRow.size());

    for (size_t idx = 0; idx < lhsRow.size(); ++idx)
    {
        auto l = {add_rhs_column_matrices_to_lhs_column_matrices(std::get<Is>(lhsRow.at(idx).columns), std::get<Is>(rhsRow.at(idx).columns))...};
        (void)l;
    }

    return true;
}

template <typename T, size_t... Is, typename... Variables>
void add_rhs_matrix_to_lhs_matrix(T &lhsMatrix, T &rhsMatrix, std::integer_sequence<size_t, Is...>, VariableGroup<Variables...> vars)
{
    auto l = {add_rhs_row_to_lhs_row(std::get<Is>(lhsMatrix), std::get<Is>(rhsMatrix), std::index_sequence_for<Variables...>{}, vars)...};
    (void)l;
}

// sets the size of a target vector to the size of a reference vector.
template <typename T1>
bool resize_vector(std::vector<T1> &theVector, size_t newSize)
{
    theVector.resize(newSize);
    return true;
}

template <typename TupleOfVectors, size_t... Is>
bool clear_tuple_of_vectors(TupleOfVectors &tupleOfVectors, std::integer_sequence<size_t, Is...>)
{
    auto l = {resize_vector(std::get<Is>(tupleOfVectors), 0)...};
    (void)l;
    return true;
}

template <typename T>
bool set_size(std::pair<T, size_t> &typeSizePair, size_t newSize)
{
    typeSizePair.second = newSize;
    return true;
}

template <typename TupleOfTypeSizePairs, size_t... Is>
bool set_sizes_zero(TupleOfTypeSizePairs &tuple, std::integer_sequence<size_t, Is...>)
{
    auto l = {set_size(std::get<Is>(tuple), 0)...};
    (void)l;
    return true;
}

template <size_t ColTypeIndex, typename T>
bool add_column_to_matrix(std::vector<T> &vectorOfRows)
{

    for (size_t idx = 0; idx < vectorOfRows.size(); ++idx)
    {
        auto &vectorOfMatrixBlocks = std::get<ColTypeIndex>(vectorOfRows.at(idx).columns);

        vectorOfMatrixBlocks.emplace_back();
    }

    return true;
}

template <size_t ColTypeIndex, typename T, size_t... Is>
bool add_column_to_matrix(T &tupleOfVectorOfRows, std::integer_sequence<size_t, Is...>)
{
    auto l = {add_column_to_matrix<ColTypeIndex>(std::get<Is>(tupleOfVectorOfRows))...};
    (void)l;
    return true;
}

template <typename ColType, typename... T>
bool remove_column_from_matrix(std::vector<Row<T...>> &vectorOfRows, TypedIndex<ColType> index)
{

    for (size_t idx = 0; idx < vectorOfRows.size(); ++idx)
    {
        vectorOfRows.at(idx).removeColumn(index);
    }

    return true;
}

template <typename ColType, typename T, size_t... Is>
bool remove_column_from_matrix(T &tupleOfVectorOfRows, TypedIndex<ColType> idx, std::integer_sequence<size_t, Is...>)
{
    auto l = {remove_column_from_matrix(std::get<Is>(tupleOfVectorOfRows), idx)...};
    (void)l;
    return true;
}

template <size_t R1, size_t R2, typename Scalar, size_t C1>
bool dot_row_with_other_row(std::vector<MatrixBlock<Scalar, R1, C1>>& lhsVec, std::vector<MatrixBlock<Scalar, R2, C1>>& rhsVec, MatrixBlock<Scalar, R1, R2>& result) {
    assert(rhsVec.size() == lhsVec.size());

    for (size_t idx = 0; idx < lhsVec.size(); ++idx) {
        auto& lhm = lhsVec.at(idx);
        auto& rhm = rhsVec.at(idx);
        
        result += lhm * rhm.transpose();
    }
    return true;
}

template <size_t R1, size_t R2, typename Scalar_, typename... Variables, size_t... Is>
MatrixBlock<Scalar_, R1, R2> dot_row_with_other_row(Row<Scalar<Scalar_>, Dimension<R1>, VariableGroup<Variables...>>& lhs, Row<Scalar<Scalar_>, Dimension<R2>, VariableGroup<Variables...>>& rhs, std::integer_sequence<size_t, Is...>) {
    
    MatrixBlock<Scalar_, R1, R2> result;
    result.setZero();

    auto l = {dot_row_with_other_row(std::get<Is>(lhs.columns), std::get<Is>(rhs.columns), result)...};
    (void)l;

    return result;
}

}; // namespace internal
} // namespace LittleOptimizer::SparseBlockMatrix