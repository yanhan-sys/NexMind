#pragma once

#include "embedding.h"
#include "linear.h"
#include "positional_encoding.h"
#include "transformer_encoder.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace nexmind {

// 自回归 Transformer Language Model：Token Embedding + Position Encoding + Causal Transformer + LM Head。
class TransformerLanguageModel final : public Module {
public:
    TransformerLanguageModel(std::size_t vocab_size,
                             std::size_t embed_dim,
                             std::size_t num_layers,
                             std::size_t num_heads,
                             std::size_t feed_forward_dim,
                             std::uint64_t seed = 42,
                             bool causal = true);

    Value forward(const Value& token_ids) const;
    std::vector<Parameter*> parameters() override;

    std::size_t vocab_size() const noexcept { return vocab_size_; }
    std::size_t embed_dim() const noexcept { return embed_dim_; }
    bool causal() const noexcept { return causal_; }
    const Embedding& embedding() const noexcept { return embedding_; }
    const SinusoidalPositionalEncoding& positional_encoding() const noexcept { return positional_encoding_; }
    const TransformerEncoder& encoder() const noexcept { return encoder_; }
    const Linear& lm_head() const noexcept { return lm_head_; }

private:
    std::size_t vocab_size_;
    std::size_t embed_dim_;
    bool causal_;
    Embedding embedding_;
    SinusoidalPositionalEncoding positional_encoding_;
    TransformerEncoder encoder_;
    Linear lm_head_;
};

// Greedy：每一步选择当前 logits 中概率最大的 token。
std::vector<std::size_t> generate_greedy(const TransformerLanguageModel& model,
                                         std::vector<std::size_t> tokens,
                                         std::size_t max_new_tokens);

// Temperature sampling：根据温度缩放后的 softmax 概率随机采样 token。
std::vector<std::size_t> generate_temperature(const TransformerLanguageModel& model,
                                              std::vector<std::size_t> tokens,
                                              std::size_t max_new_tokens,
                                              double temperature,
                                              std::uint64_t seed = 42);

} // namespace nexmind
