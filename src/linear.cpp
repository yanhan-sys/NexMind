#include "linear.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace nexmind {

Linear::Linear(std::size_t in_features, std::size_t out_features, std::uint64_t seed)
    : in_features_(in_features),
      out_features_(out_features),
      weight_(Tensor({in_features, out_features}), "weight"),
      bias_(Tensor({1, out_features}, 0.0), "bias") {
    if (in_features == 0 || out_features == 0) {
        throw std::invalid_argument("Linear feature size must be positive");
    }
    const double limit = std::sqrt(6.0 / static_cast<double>(in_features + out_features));
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> distribution(-limit, limit);
    for (std::size_t i = 0; i < weight_.value().size(); ++i) {
        weight_.value().data()[i] = distribution(generator);
    }
}

Value Linear::forward(const Value& input) const {
    if (input.data().ndim() != 2 || input.data().shape()[1] != in_features_) {
        throw std::invalid_argument("Linear input shape mismatch");
    }
    Value projected = matmul(input, weight_.value());
    return add_bias(projected, bias_.value());
}

std::vector<Parameter*> Linear::parameters() noexcept {
    return {&weight_, &bias_};
}

} // namespace nexmind
