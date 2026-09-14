#include "embedding.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace nexmind {

Embedding::Embedding(std::size_t num_embeddings, std::size_t embedding_dim, std::uint64_t seed)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim), weight_(Tensor({num_embeddings, embedding_dim}), "embedding_weight") {
    if (num_embeddings == 0 || embedding_dim == 0) {
        throw std::invalid_argument("Embedding dimensions must be positive");
    }
    const double limit = std::sqrt(6.0 / static_cast<double>(embedding_dim));
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> distribution(-limit, limit);
    for (std::size_t i = 0; i < weight_.value().data().size(); ++i) {
        weight_.value().data()[i] = distribution(generator);
    }
}

Value Embedding::forward(const Value& indices) const {
    if (indices.data().ndim() != 1) {
        throw std::invalid_argument("Embedding indices must be 1D");
    }
    const std::size_t sequence_length = indices.data().shape()[0];
    Tensor output({sequence_length, embedding_dim_});
    std::vector<std::size_t> selected(sequence_length);
    for (std::size_t row = 0; row < sequence_length; ++row) {
        const double raw = indices.data()[row];
        const std::size_t index = static_cast<std::size_t>(raw);
        if (raw < 0.0 || static_cast<double>(index) != raw || index >= num_embeddings_) {
            throw std::out_of_range("Embedding index out of range");
        }
        selected[row] = index;
        for (std::size_t column = 0; column < embedding_dim_; ++column) {
            output.at({row, column}) = weight_.value().data().at(index * embedding_dim_ + column);
        }
    }

    Value result(std::move(output), true);
    const auto weight_node = weight_.value().node_;
    const auto result_node = result.node_;
    result_node->parents = {weight_node};
    result_node->backward = [weight_node, selected, embedding_dim = embedding_dim_](Value::Node& node) {
        for (std::size_t row = 0; row < selected.size(); ++row) {
            const std::size_t offset = selected[row] * embedding_dim;
            for (std::size_t column = 0; column < embedding_dim; ++column) {
                weight_node->grad[offset + column] += node.grad[row * embedding_dim + column];
            }
        }
    };
    return result;
}

std::vector<Parameter*> Embedding::parameters() {
    return {&weight_};
}

} // namespace nexmind
