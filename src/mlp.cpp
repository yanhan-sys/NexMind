#include "mlp.h"

#include <utility>

namespace nexmind {

MLP::MLP(std::size_t in_features, std::size_t hidden_features, std::size_t out_features, std::uint64_t seed)
    : input_layer_(in_features, hidden_features, seed), output_layer_(hidden_features, out_features, seed + 1) {}

Value MLP::forward(const Value& input) const {
    Value hidden = input_layer_.forward(input);
    Value activated = relu(hidden);
    return output_layer_.forward(activated);
}

std::vector<Parameter*> MLP::parameters() {
    std::vector<Parameter*> result = input_layer_.parameters();
    const std::vector<Parameter*> output_parameters = output_layer_.parameters();
    result.insert(result.end(), output_parameters.begin(), output_parameters.end());
    return result;
}

} // namespace nexmind
