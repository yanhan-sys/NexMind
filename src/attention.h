#pragma once

#include "linear.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nexmind {

class Attention final : public Module {
public:
    explicit Attention(std::size_t embed_dim, std::uint64_t seed = 42);

    std::size_t embed_dim() const noexcept { return embed_dim_; }

    Value forward(const Value& input) const;

    Linear& query_projection() noexcept { return query_projection_; }
    Linear& key_projection() noexcept { return key_projection_; }
    Linear& value_projection() noexcept { return value_projection_; }
    Linear& output_projection() noexcept { return output_projection_; }

    const Linear& query_projection() const noexcept { return query_projection_; }
    const Linear& key_projection() const noexcept { return key_projection_; }
    const Linear& value_projection() const noexcept { return value_projection_; }
    const Linear& output_projection() const noexcept { return output_projection_; }

    std::vector<Parameter*> parameters() override;

private:
    std::size_t embed_dim_;
    Linear query_projection_;
    Linear key_projection_;
    Linear value_projection_;
    Linear output_projection_;
};

} // namespace nexmind
