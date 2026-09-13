#pragma once

#include "parameter.h"

#include <vector>

namespace nexmind {

class Module {
public:
    virtual ~Module() = default;
    virtual std::vector<Parameter*> parameters() noexcept = 0;
};

} // namespace nexmind
