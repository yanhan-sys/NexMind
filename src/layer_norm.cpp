#include "layer_norm.h"

#include <stdexcept>
#include <utility>

namespace nexmind {

LayerNorm::LayerNorm(std::size_t features, double epsilon)
    : features_(features), epsilon_(epsilon), gamma_(Tensor({1, features}, 1.0), "layer_norm_gamma"), beta_(Tensor({1, features}, 0.0), "layer_norm_beta") {
    if (features == 0 || epsilon <= 0.0) {
        throw std::invalid_argument("LayerNorm configuration is invalid");
    }
}

Value LayerNorm::forward(const Value& input) const {
    return layer_norm(input, gamma_.value(), beta_.value(), epsilon_);
}

std::vector<Parameter*> LayerNorm::parameters() {
    return {&gamma_, &beta_};
}

} // namespace nexmind
