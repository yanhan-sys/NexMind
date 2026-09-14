#pragma once

#include "transformer_block.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nexmind {

// Transformer Encoder：由多个 Transformer Block 顺序堆叠而成，可选择因果注意力。
class TransformerEncoder final : public Module {
public:
    TransformerEncoder(std::size_t num_layers,
                       std::size_t embed_dim,
                       std::size_t num_heads,
                       std::size_t feed_forward_dim,
                       std::uint64_t seed = 42,
                       bool causal = false);

    Value forward(const Value& input) const;
    std::vector<Parameter*> parameters() override;

    std::size_t num_layers() const noexcept { return blocks_.size(); }
    const TransformerBlock& block(std::size_t index) const;

private:
    std::vector<TransformerBlock> blocks_;
};

} // namespace nexmind
