#pragma once

#include "linear.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nexmind {

class MultiHeadAttention final : public Module {
public:
    MultiHeadAttention(std::size_t embed_dim, std::size_t num_heads, std::uint64_t seed = 42, bool causal = false);

    std::size_t embed_dim() const noexcept { return embed_dim_; }
    std::size_t num_heads() const noexcept { return num_heads_; }
    std::size_t head_dim() const noexcept { return head_dim_; }
    bool causal() const noexcept { return causal_; }

    Value forward(const Value& input) const;

    std::vector<Parameter*> parameters() override;

private:
    std::size_t embed_dim_;
    std::size_t num_heads_;
    std::size_t head_dim_;
    bool causal_;
    std::vector<Linear> query_projections_;
    std::vector<Linear> key_projections_;
    std::vector<Linear> value_projections_;
    Linear output_projection_;
};

} // namespace nexmind
