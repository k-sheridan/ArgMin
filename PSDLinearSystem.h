#pragma once

#include "MetaHelpers.h"

namespace LittleOptimizer {

template <typename... T>
class PSDLinearSystem;

template <typename ScalarType, typename... Variables>
class PSDLinearSystem<Scalar<ScalarType>, VariableGroup<Variables...>> {

};

}