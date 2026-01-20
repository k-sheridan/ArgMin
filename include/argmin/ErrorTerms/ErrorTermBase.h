#pragma once

#include <Eigen/Core>
#include "argmin/Containers/Key.h"
#include "argmin/Optimization/OptimizerContainers.h"
#include "argmin/Types/MetaHelpers.h"

namespace ArgMin
{

template <typename...>
class ErrorTermBase;

/**
 * @brief Base class for error terms that depend on a set of independent variables.
 *
 * Stores the linearized error term (Jacobians and residual) and keys to access
 * the variables. Derived classes must implement an evaluate function:
 *
 * @code
 * void evaluate(VariableContainer<Variables...>& variables, bool relinearize);
 * @endcode
 *
 * The evaluate function must:
 * 1. Compute the residual vector and store it in the `residual` member
 * 2. If relinearize is true, compute the Jacobian of the residual with respect
 *    to each independent variable and store them in `variableJacobians`
 * 3. Set `linearizationValid` to true if linearization succeeded, false otherwise
 *
 * @tparam ScalarType The floating point type (typically float or double)
 * @tparam ResidualDimension The dimension of the residual vector
 * @tparam IndependentVariables... The variable types this error term depends on
 */
template <int ResidualDimension, typename ScalarType, typename... IndependentVariables>
class ErrorTermBase<Scalar<ScalarType>, Dimension<ResidualDimension>, VariableGroup<IndependentVariables...>>
{
public:

    using VariablePointers = std::tuple<IndependentVariables*...>;
    using VariableKeys = std::tuple<VariableKey<IndependentVariables>...>;
    using VariableJacobians = std::tuple<Eigen::Matrix<ScalarType, ResidualDimension, IndependentVariables::dimension>...>;

    // The precision of this error term.
    typedef ScalarType scalar_type;

    /// Compile time acces to the error term's dimension.
    static const int residual_dimension = ResidualDimension;

    /// These jacobians are from the most recent linearization.
    VariableJacobians variableJacobians;
    /// These are the keys used to access the variables over time.
    /// These keys can only be invalidated is the variable is removed or overwritten.
    VariableKeys variableKeys;
    /// These pointers are used to avoid the indirection of the slotmap.
    /// These pointers are invalidated every time a key is added or removed from the slot map.
    VariablePointers variablePointers;

    /// This is the most recent residual computed for the error term.
    Eigen::Matrix<ScalarType, ResidualDimension, 1> residual;

    /// This is the information matrix for this error term.
    Eigen::Matrix<ScalarType, ResidualDimension, ResidualDimension> information;

    /// This flag is used to let the user know if the error term was linearized successfully.
    bool linearizationValid = false;

    /// Extracts the most recent pointer to each of the variables using their key and updates them.
    template <typename... Variables>
    void updateVariablePointers(VariableContainer<Variables...> &variableContainer)
    {
        internal::static_for(variableKeys, [&](auto i, auto &variableKey) {
            auto& variableMap = variableContainer.template getVariableMap<typename std::tuple_element<i, std::tuple<IndependentVariables...>>::type>();
            auto variableIterator = variableMap.at(variableKey);

            assert(variableIterator != variableMap.end());

            std::get<i>(variablePointers) = &(*(variableIterator));
        });
    }

    /// Verifies that the pointers stored are equal to the true variable location.
    template <typename... Variables>
    bool checkVariablePointerConsistency(VariableContainer<Variables...> &variableContainer)
    {
        bool flag = true;
        internal::static_for(variableKeys, [&](auto i, auto &variableKey) {
            auto& variableMap = variableContainer.template getVariableMap<typename std::tuple_element<i, std::tuple<IndependentVariables...>>::type>();
            auto variableIterator = variableMap.at(variableKey);

            assert(variableIterator != variableMap.end());

            // The pointers should match.
            if (std::get<i>(variablePointers) != &(*(variableIterator)))
            {
                flag = false;
            }
        });

        return flag;
    }
};

} // namespace ArgMin