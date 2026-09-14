#pragma once

#include "linear.h"
#include "relu.h"

#include <vector>

namespace nexmind {

class MLP final : public Module {
public:
    MLP(std::size_t in_features, std::size_t hidden_features, std::size_t out_features, std::uint64_t seed = 42);

    Value forward(const Value& input) const;
    std::vector<Parameter*> parameters() override;

    Linear& input_layer() noexcept { return input_layer_; }
    Linear& output_layer() noexcept { return output_layer_; }
    const Linear& input_layer() const noexcept { return input_layer_; }
    const Linear& output_layer() const noexcept { return output_layer_; }

private:
    Linear input_layer_;
    Linear output_layer_;
};

} // namespace nexmind
