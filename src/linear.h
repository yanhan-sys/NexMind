#pragma once

#include "module.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nexmind {

class Linear final : public Module {
public:
    Linear(std::size_t in_features, std::size_t out_features, std::uint64_t seed = 42);

    std::size_t in_features() const noexcept { return in_features_; }
    std::size_t out_features() const noexcept { return out_features_; }

    Value forward(const Value& input) const;

    Parameter& weight() noexcept { return weight_; }
    const Parameter& weight() const noexcept { return weight_; }
    Parameter& bias() noexcept { return bias_; }
    const Parameter& bias() const noexcept { return bias_; }

    std::vector<Parameter*> parameters() noexcept override;

private:
    std::size_t in_features_;
    std::size_t out_features_;
    Parameter weight_;
    Parameter bias_;
};

} // namespace nexmind
