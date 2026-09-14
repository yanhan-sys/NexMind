#pragma once

#include "embedding.h"
#include "linear.h"
#include "transformer_encoder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nexmind {

// Decoder-free Transformer Language Model：Token Embedding + Encoder + LM Head。
class TransformerLanguageModel final : public Module {
public:
    TransformerLanguageModel(std::size_t vocab_size,
                             std::size_t embed_dim,
                             std::size_t num_layers,
                             std::size_t num_heads,
                             std::size_t feed_forward_dim,
                             std::uint64_t seed = 42);

    Value forward(const Value& token_ids) const;
    std::vector<Parameter*> parameters() override;

    std::size_t vocab_size() const noexcept { return vocab_size_; }
    std::size_t embed_dim() const noexcept { return embed_dim_; }
    const Embedding& embedding() const noexcept { return embedding_; }
    const TransformerEncoder& encoder() const noexcept { return encoder_; }
    const Linear& lm_head() const noexcept { return lm_head_; }

private:
    std::size_t vocab_size_;
    std::size_t embed_dim_;
    Embedding embedding_;
    TransformerEncoder encoder_;
    Linear lm_head_;
};

} // namespace nexmind
