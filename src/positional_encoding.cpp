#include "positional_encoding.h"

#include <cmath>
#include <stdexcept>

namespace nexmind {

SinusoidalPositionalEncoding::SinusoidalPositionalEncoding(std::size_t embed_dim)
    : embed_dim_(embed_dim) {
    if (embed_dim == 0) {
        throw std::invalid_argument("Positional encoding dimension must be positive");
    }
}

Value SinusoidalPositionalEncoding::forward(const Value& input) const {
    if (input.data().ndim() != 2 || input.data().shape()[1] != embed_dim_) {
        throw std::invalid_argument("Positional encoding input shape mismatch");
    }

    const std::size_t sequence_length = input.data().shape()[0];
    Tensor encoding({sequence_length, embed_dim_}, 0.0);
    // 使用标准 Transformer 正弦/余弦公式生成固定位置编码。
    for (std::size_t position = 0; position < sequence_length; ++position) {
        for (std::size_t dimension = 0; dimension < embed_dim_; ++dimension) {
            const double exponent = static_cast<double>(dimension - (dimension % 2)) / static_cast<double>(embed_dim_);
            const double angle = static_cast<double>(position) / std::pow(10000.0, exponent);
            encoding.at({position, dimension}) = (dimension % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }

    // 加法本身由 Autograd 管理，因此位置编码不会截断 Embedding 的梯度。
    return add(input, Value(std::move(encoding), false));
}

} // namespace nexmind
