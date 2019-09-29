#pragma once

#include <tuple>
#include <utility>
#include <memory>
#include <type_traits>
#include <vector>

#include "Key.h"
#include "MetaHelpers.h"
#include "MatrixBlock.h"
#include "SparseBlockMatrixInternal.h"

namespace LittleOptimizer::SparseBlockMatrix
{

template <typename Scalar_, typename... Variables>
class SparseBlockMatrix<Scalar<Scalar_>, VariableGroup<Variables...>>
{

private:
    using variable_list = std::tuple<Variables...>;
    static_assert(sizeof...(Variables) > 0); // ensure that there is at least on variable.

    template <typename VariableType>
    using variable_index = LittleOptimizer::internal::Index<VariableType, variable_list>;

    typedef std::tuple<std::pair<Variables, size_t>...> sizes_t;
    sizes_t rowSizes, columnSizes; // Stores the size of the sub matrices for all type combintations.

    template <size_t dimension>
    using row_t = Row<Scalar<Scalar_>, Dimension<dimension>, VariableGroup<Variables...>>;

    typedef std::tuple<std::vector<row_t<Variables::dimension>>...> matrix_t;

    matrix_t matrix;

public:
    SparseBlockMatrix()
    {
        // resize everything to zero.
        this->clear();
    }

    /**
     * Removes all matrices and sets the total matrix size to 0.
     */
    void clear()
    {
        internal::clear_tuple_of_vectors(matrix, std::index_sequence_for<Variables...>{});
        internal::set_sizes_zero(rowSizes, std::index_sequence_for<Variables...>{});
        internal::set_sizes_zero(columnSizes, std::index_sequence_for<Variables...>{});
    }

    /**
     * Gets the sub matrix for the corresponding type keys, and returns it by reference.
     */
    template <typename RowType, typename ColType>
    MatrixBlock<Scalar_, RowType::dimension, ColType::dimension> &get(LittleOptimizer::TypedIndex<RowType> row, LittleOptimizer::TypedIndex<ColType> col)
    {
        return getRow(row).getColumn(col);
    }

    /**
     * Finds the row of the corresponding type key, and returns it by reference.
     */
    template <typename RowType>
    Row<Scalar<Scalar_>, Dimension<RowType::dimension>, VariableGroup<Variables...>> &getRow(LittleOptimizer::TypedIndex<RowType> row)
    {
        return std::get<variable_index<RowType>::value>(matrix).at(row.index);
    }

    /**
     * Computes how many rows there are for a given type.
     */
    template <typename RowType>
    size_t rowsOfType()
    {
        return std::get<std::pair<RowType, size_t>>(rowSizes).second;
    }

    /**
     * Computes how many rows there are for a given type.
     */
    template <typename ColumnType>
    size_t columnsOfType()
    {
        return std::get<std::pair<ColumnType, size_t>>(columnSizes).second;
    }

    /**
     * Adds a row of the correct size to the corresponding vector.
     */
    template <typename RowType>
    TypedIndex<RowType> addRow()
    {
        constexpr auto idx = variable_index<RowType>::value;
        static_assert(std::is_same<typename std::tuple_element<idx, sizes_t>::type, std::pair<RowType, size_t>>::value);
        assert(std::get<idx>(rowSizes).second == std::get<idx>(matrix).size());

        TypedIndex<RowType> index;
        index.index = std::get<idx>(matrix).size();

        std::get<idx>(matrix).emplace_back(columnSizes, std::index_sequence_for<Variables...>{});
        std::get<idx>(rowSizes).second = std::get<idx>(matrix).size(); // update the row count.

        return index;
    }

    // Erases a row from the matrix.
    template <typename RowType>
    void removeRow(TypedIndex<RowType> idx) {

        assert(std::get<variable_index<RowType>::value>(rowSizes).second > 0);

        std::get<variable_index<RowType>::value>(matrix).erase(std::get<variable_index<RowType>::value>(matrix).begin() + idx.index);
        std::get<variable_index<RowType>::value>(rowSizes).second--;
    }

    template <typename ColType>
    TypedIndex<ColType> addColumn()
    {
        constexpr auto idx = variable_index<ColType>::value;
        static_assert(std::is_same<typename std::tuple_element<idx, sizes_t>::type, std::pair<ColType, size_t>>::value);

        internal::add_column_to_matrix<idx>(matrix, std::index_sequence_for<Variables...>{});

        TypedIndex<ColType> index;
        index.index = std::get<idx>(columnSizes).second;
        ++std::get<idx>(columnSizes).second; // After adding columns increment the column sizes at the col type given.

        return index;
    }

    // Erases a column from the matrix.
    template <typename ColType>
    void removeColumn(TypedIndex<ColType> idx) {

        assert(std::get<variable_index<ColType>::value>(columnSizes).second > 0);

        internal::remove_column_from_matrix(matrix, idx, std::index_sequence_for<Variables...>{});
        std::get<variable_index<ColType>::value>(columnSizes).second--;
    }

    /**
     * Adds together in the most efficient way two sparse block matrices.
     */
    SparseBlockMatrix<Scalar<Scalar_>, VariableGroup<Variables...>> &operator+=(SparseBlockMatrix<Scalar<Scalar_>, VariableGroup<Variables...>> &rhs)
    {
        internal::add_rhs_matrix_to_lhs_matrix(matrix, rhs.matrix, std::index_sequence_for<Variables...>{}, VariableGroup<Variables...>());
        return *this;
    }
};


// Sparse Block Row

template <typename Scalar_, size_t Rows_, typename... Variables>
class Row<Scalar<Scalar_>, Dimension<Rows_>, VariableGroup<Variables...>>
{
private:
    using variable_list = std::tuple<Variables...>;
    static_assert(sizeof...(Variables) > 0); // ensure that there is at least on variable.

    template <typename VariableType>
    using variable_index = LittleOptimizer::internal::Index<VariableType, variable_list>;

public:
    typedef std::tuple<std::vector<MatrixBlock<Scalar_, Rows_, Variables::dimension>>...> columns_t;

    columns_t columns;

    Row()
    {
    }

    // Expects: tuple<pair<T1, size_t>, pair<T2, size_t>, ...>
    // Sets up the columns to be a certiain size.
    template <typename TupleOfTypeSizePair, size_t... Is>
    Row(TupleOfTypeSizePair &tuple, std::integer_sequence<size_t, Is...>)
    {
        auto l = {internal::resize_vector(std::get<Is>(columns), std::get<Is>(tuple).second)...};
        (void)l;
    }

    /**
     * Gets the sub matrix for the corresponding type keys, and returns it by reference.
     */
    template <typename ColType>
    MatrixBlock<Scalar_, Rows_, ColType::dimension>& getColumn(LittleOptimizer::TypedIndex<ColType> col) {
        return std::get<variable_index<ColType>::value>(columns).at(col.index);
    }

    // Removes all matrices from vector.
    void clear() {
        internal::clear_tuple_of_vectors(columns, std::index_sequence_for<Variables...>{});
    }

    // Appends a block column to the row.
    template <typename ColType>
    TypedIndex<ColType> addColumn() {
        TypedIndex<ColType> index;
        index.index = std::get<variable_index<ColType>::value>(columns).size();

        std::get<variable_index<ColType>::value>(columns).emplace_back();

        return index;
    }

    // removes a block column from the row
    template <typename ColType>
    void removeColumn(TypedIndex<ColType> idx) {
        std::get<variable_index<ColType>::value>(columns).erase(std::get<variable_index<ColType>::value>(columns).begin() + idx.index);
    }

    /**
     * Computes how many rows there are for a given type.
     */
    template <typename ColumnType>
    size_t columnsOfType() {
        return std::get<variable_index<ColumnType>::value>(columns).size();
    }

    /**
     * Computes the block dot product or two rows.
     */
    template <size_t OtherRowDimension>
    MatrixBlock<Scalar_, Rows_, OtherRowDimension> dot(Row<Scalar<Scalar_>, Dimension<OtherRowDimension>, VariableGroup<Variables...>>& rhsRow) {
        return internal::dot_row_with_other_row(*this, rhsRow, std::index_sequence_for<Variables...>{});
    }

    /**
     * Adds together in the most efficient way two sparse block matrices.
     */
    Row<Scalar<Scalar_>, Dimension<Rows_>, VariableGroup<Variables...>> &operator+=(Row<Scalar<Scalar_>, Dimension<Rows_>, VariableGroup<Variables...>> &rhs)
    {
        internal::add_rhs_row_to_lhs_row(*this, rhs, std::index_sequence_for<Variables...>{});
        return *this;
    }
};

// Sparse Block Vector

template <typename Scalar_, typename... Variables>
using SparseBlockVector = Row<Scalar<Scalar_>, Dimension<1>, VariableGroup<Variables...>>;

} // namespace LittleOptimizer::SparseBlockMatrix