#include "transformer_lm.h"

#include <stdexcept>
#include <utility>

namespace nexmind {

TransformerLanguageModel::TransformerLanguageModel(std::size_t vocab_size,
                                                   std::size_t embed_dim,
                                                   std::size_t num_layers,
                                                   std::size_t num_heads,
                                                   std::size_t feed_forward_dim,
                                                   std::uint64_t seed)
    : vocab_size_(vocab_size),
      embed_dim_(embed_dim),
      embedding_(vocab_size, embed_dim, seed),
      encoder_(num_layers, embed_dim, num_heads, feed_forward_dim, seed + 1),
      lm_head_(embed_dim, vocab_size, seed + 2) {
    // 词表必须非空，隐藏维度由 Embedding 和 Encoder 保持一致。
    if (vocab_size == 0 || embed_dim == 0) {
        throw std::invalid_argument("TransformerLanguageModel dimensions must be positive");
    }
}

Value TransformerLanguageModel::forward(const Value& token_ids) const {
    // Token ID → Embedding → Transformer Encoder → Vocabulary logits。
    const Value embedded = embedding_.forward(token_ids);
    const Value encoded = encoder_.forward(embedded);
    return lm_head_.forward(encoded);
}

std::vector<Parameter*> TransformerLanguageModel::parameters() {
    std::vector<Parameter*> result;
    const auto embedding_parameters = embedding_.parameters();
    const auto encoder_parameters = encoder_.parameters();
    const auto lm_head_parameters = lm_head_.parameters();
    result.reserve(embedding_parameters.size() + encoder_parameters.size() + lm_head_parameters.size());
    result.insert(result.end(), embedding_parameters.begin(), embedding_parameters.end());
    result.insert(result.end(), encoder_parameters.begin(), encoder_parameters.end());
    result.insert(result.end(), lm_head_parameters.begin(), lm_head_parameters.end());
    return result;
}

} // namespace nexmind
