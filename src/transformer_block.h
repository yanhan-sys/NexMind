#pragma once

#include "layer_norm.h"
#include "mlp.h"
#include "multi_head_attention.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nexmind {

// 标准 Transformer Encoder Block：Pre-Norm + 残差连接。
class TransformerBlock final : public Module {
public:
    TransformerBlock(std::size_t embed_dim,
                     std::size_t num_heads,
                     std::size_t feed_forward_dim,
                     std::uint64_t seed = 42);

    Value forward(const Value& input) const;
    std::vector<Parameter*> parameters() override;

    const MultiHeadAttention& attention() const noexcept { return attention_; }
    const MLP& feed_forward() const noexcept { return feed_forward_; }
    const LayerNorm& attention_norm() const noexcept { return attention_norm_; }
    const LayerNorm& feed_forward_norm() const noexcept { return feed_forward_norm_; }

private:
    LayerNorm attention_norm_;
    MultiHeadAttention attention_;
    LayerNorm feed_forward_norm_;
    MLP feed_forward_;
};

} // namespace nexmind
