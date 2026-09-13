#pragma once

#include "autograd.h"

#include <string>

namespace nexmind {

class Parameter {
public:
    Parameter() = default;
    explicit Parameter(Tensor data, std::string name = {});

    Value& value() noexcept { return value_; }
    const Value& value() const noexcept { return value_; }
    const std::string& name() const noexcept { return value_.name(); }

    void zero_grad() { value_.zero_grad(); }

private:
    Value value_;
};

} // namespace nexmind
