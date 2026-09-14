#pragma once

#include "module.h"

#include <cstddef>

namespace nexmind {

class LayerNorm final : public Module {
public:
    explicit LayerNorm(std::size_t features, double epsilon = 1e-5);

    Value forward(const Value& input) const;
    std::vector<Parameter*> parameters() override;

    Parameter& gamma() noexcept { return gamma_; }
    Parameter& beta() noexcept { return beta_; }
    const Parameter& gamma() const noexcept { return gamma_; }
    const Parameter& beta() const noexcept { return beta_; }

private:
    std::size_t features_;
    double epsilon_;
    Parameter gamma_;
    Parameter beta_;
};

} // namespace nexmind
