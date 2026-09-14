#include "multi_head_attention.h"

#include "autograd.h"

#include <cmath>
#include <stdexcept>

namespace nexmind {

MultiHeadAttention::MultiHeadAttention(std::size_t embed_dim, std::size_t num_heads, std::uint64_t seed, bool causal)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      causal_(causal),
      output_projection_(embed_dim, embed_dim, seed + 3000) {
    if (embed_dim == 0 || num_heads == 0 || embed_dim % num_heads != 0) {
        throw std::invalid_argument("MultiHeadAttention dimensions must be positive and divisible");
    }

    query_projections_.reserve(num_heads_);
    key_projections_.reserve(num_heads_);
    value_projections_.reserve(num_heads_);
    for (std::size_t head = 0; head < num_heads_; ++head) {
        const std::uint64_t head_seed = seed + static_cast<std::uint64_t>(head) * 3;
        query_projections_.emplace_back(embed_dim_, head_dim_, head_seed);
        key_projections_.emplace_back(embed_dim_, head_dim_, head_seed + 1);
        value_projections_.emplace_back(embed_dim_, head_dim_, head_seed + 2);
    }
}

Value MultiHeadAttention::forward(const Value& input) const {
    // 标准多头自注意力：语言模型模式下启用因果掩码，阻止当前位置看到未来 token。
    if (input.data().ndim() != 2 || input.data().shape()[1] != embed_dim_) {
        throw std::invalid_argument("MultiHeadAttention input shape mismatch");
    }

    std::vector<Value> heads;
    heads.reserve(num_heads_);
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim_));
    for (std::size_t head = 0; head < num_heads_; ++head) {
        const Value query = query_projections_[head].forward(input);
        const Value key = key_projections_[head].forward(input);
        const Value value = value_projections_[head].forward(input);
        if (causal_) {
            heads.push_back(scaled_dot_product_attention_causal(query, key, value, scale));
        } else {
            heads.push_back(scaled_dot_product_attention(query, key, value, scale));
        }
    }

    const Value combined = concat_columns(heads);
    return output_projection_.forward(combined);
}

std::vector<Parameter*> MultiHeadAttention::parameters() {
    std::vector<Parameter*> result;
    result.reserve(num_heads_ * 6 + 2);
    for (std::size_t head = 0; head < num_heads_; ++head) {
        const auto query_parameters = query_projections_[head].parameters();
        const auto key_parameters = key_projections_[head].parameters();
        const auto value_parameters = value_projections_[head].parameters();
        result.insert(result.end(), query_parameters.begin(), query_parameters.end());
        result.insert(result.end(), key_parameters.begin(), key_parameters.end());
        result.insert(result.end(), value_parameters.begin(), value_parameters.end());
    }
    const auto output_parameters = output_projection_.parameters();
    result.insert(result.end(), output_parameters.begin(), output_parameters.end());
    return result;
}

} // namespace nexmind
