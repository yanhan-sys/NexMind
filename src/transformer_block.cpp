#include "transformer_block.h"

#include <stdexcept>

namespace nexmind {

TransformerBlock::TransformerBlock(std::size_t embed_dim,
                                   std::size_t num_heads,
                                   std::size_t feed_forward_dim,
                                   std::uint64_t seed,
                                   bool causal)
    : attention_norm_(embed_dim),
      attention_(embed_dim, num_heads, seed, causal),
      feed_forward_norm_(embed_dim),
      feed_forward_(embed_dim, feed_forward_dim, embed_dim, seed + 1) {
    // Transformer Block 的特征维度必须有效，具体整除关系由 MultiHeadAttention 校验。
    if (embed_dim == 0 || feed_forward_dim == 0) {
        throw std::invalid_argument("TransformerBlock dimensions must be positive");
    }
}

Value TransformerBlock::forward(const Value& input) const {
    // Pre-Norm：先归一化，再做注意力，最后与原输入做残差相加。
    const Value normalized_attention_input = attention_norm_.forward(input);
    const Value attention_output = attention_.forward(normalized_attention_input);
    const Value attention_residual = add(input, attention_output);

    // 第二个 Pre-Norm 子层同样采用残差连接。
    const Value normalized_feed_forward_input = feed_forward_norm_.forward(attention_residual);
    const Value feed_forward_output = feed_forward_.forward(normalized_feed_forward_input);
    return add(attention_residual, feed_forward_output);
}

std::vector<Parameter*> TransformerBlock::parameters() {
    std::vector<Parameter*> result;
    const auto attention_norm_parameters = attention_norm_.parameters();
    const auto attention_parameters = attention_.parameters();
    const auto feed_forward_norm_parameters = feed_forward_norm_.parameters();
    const auto feed_forward_parameters = feed_forward_.parameters();
    result.reserve(attention_norm_parameters.size() + attention_parameters.size() + feed_forward_norm_parameters.size() + feed_forward_parameters.size());
    result.insert(result.end(), attention_norm_parameters.begin(), attention_norm_parameters.end());
    result.insert(result.end(), attention_parameters.begin(), attention_parameters.end());
    result.insert(result.end(), feed_forward_norm_parameters.begin(), feed_forward_norm_parameters.end());
    result.insert(result.end(), feed_forward_parameters.begin(), feed_forward_parameters.end());
    return result;
}

} // namespace nexmind
