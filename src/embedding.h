#pragma once

#include "module.h"

#include <cstddef>
#include <cstdint>

namespace nexmind {

class Embedding final : public Module {
public:
    Embedding(std::size_t num_embeddings, std::size_t embedding_dim, std::uint64_t seed = 42);

    Value forward(const Value& indices) const;
    std::vector<Parameter*> parameters() override;

    std::size_t num_embeddings() const noexcept { return num_embeddings_; }
    std::size_t embedding_dim() const noexcept { return embedding_dim_; }
    Parameter& weight() noexcept { return weight_; }
    const Parameter& weight() const noexcept { return weight_; }

private:
    std::size_t num_embeddings_;
    std::size_t embedding_dim_;
    Parameter weight_;
};

} // namespace nexmind
