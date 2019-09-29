#pragma once

#include <tuple>
#include <vector>
 
#include "PSDLinearSystem.h"
#include "Key.h"
#include "slot_map.h"

namespace LittleOptimizer {

template <typename... T>
class Optimizer;

template <typename... Variables, typename... ErrorTerms>
class Optimizer<VariableGroup<Variables...>, ErrorTermGroup<ErrorTerms...>> 
{
    std::tuple<slot_map<Variables>...> variableContainers; // Tuple of vectors of variables.

    std::tuple<slot_map<ErrorTerms>...> errorTermContainers; // Tuple of vectors of error terms.

    PSDLinearSystem<Scalar<double>, VariableGroup<Variables...>> linearSystem; // Stores A and b for solving.

    template <typename VariableType>
    VariableKey<VariableType> addVariable(VariableType var) {
        // Set up the key 
        VariableKey<VariableType> key;
        // Add the variable
        key.slotMapKey = std::get<slot_map<VariableType>>(variableContainers).insert(var);

        //TODO Make a spot in the LinearSystems for this variable, and ensure its key is consistent


        return key;
    }

    template <typename ErrorTermType>
    ErrorTermKey<ErrorTermType> addErrorTerm(ErrorTermType errorTerm) {
        // Set up the key 
        ErrorTermKey<ErrorTermType> key;
        // Add error term to its vector.
        key.slotMapKey = std::get<std::vector<ErrorTermType>>(errorTermContainers).insert(errorTerm);

        return key;
    }

};

} // namespace LittleOptimizer