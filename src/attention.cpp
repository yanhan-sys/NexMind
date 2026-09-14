#include "attention.h"

#include <cmath>
#include <stdexcept>

namespace nexmind {

Attention::Attention(std::size_t embed_dim, std::uint64_t seed)
    : embed_dim_(embed_dim),
      query_projection_(embed_dim, embed_dim, seed),
      key_projection_(embed_dim, embed_dim, seed + 1),
      value_projection_(embed_dim, embed_dim, seed + 2),
      output_projection_(embed_dim, embed_dim, seed + 3) {
    if (embed_dim == 0) {
        throw std::invalid_argument("Attention embedding size must be positive");
    }
}

Value Attention::forward(const Value& input) const {
    // 第一版采用标准单头自注意力：Q、K、V 都来自同一个序列。
    if (input.data().ndim() != 2 || input.data().shape()[1] != embed_dim_) {
        throw std::invalid_argument("Attention input shape mismatch");
    }
    const Value query = query_projection_.forward(input);
    const Value key = key_projection_.forward(input);
    const Value value = value_projection_.forward(input);
    const double scale = 1.0 / std::sqrt(static_cast<double>(embed_dim_));
    const Value attended = scaled_dot_product_attention(query, key, value, scale);
    return output_projection_.forward(attended);
}

std::vector<Parameter*> Attention::parameters() {
    std::vector<Parameter*> result;
    const auto append = [&result](Linear& projection) {
        const auto projection_parameters = projection.parameters();
        result.insert(result.end(), projection_parameters.begin(), projection_parameters.end());
    };
    append(query_projection_);
    append(key_projection_);
    append(value_projection_);
    append(output_projection_);
    return result;
}

} // namespace nexmind
