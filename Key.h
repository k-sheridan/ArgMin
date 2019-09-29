#pragma once

#include "Types.h"
#include "slot_map.h"

namespace QDVO
{

/**
 * A key class which allows for nested IDs.
 */
class Key
{
public:
    QDVO::ID inner, outer;

    Key(QDVO::ID outerKey, QDVO::ID innerKey) : inner(innerKey), outer(outerKey) {}

    bool operator<(const Key &comp) const
    {
        return outer < comp.outer || (outer == comp.outer && inner < comp.inner);
    }
};

} // namespace QDVO

namespace LittleOptimizer {

template <typename T>
struct VariableKey {
    std::pair<unsigned, unsigned> slotMapKey; // Slot Map key (index, generation).
};

template <typename T>
struct ErrorTermKey {
    std::pair<unsigned, unsigned> slotMapKey; // Slot Map key (index, generation).
};

} // namespace LittleOptimizer