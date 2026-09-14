#include "autograd.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace nexmind {

namespace {

Tensor zeros_like(const Tensor& input) {
    return Tensor(input.shape(), 0.0);
}

void add_inplace(Tensor& dst, const Tensor& src) {
    if (dst.shape() != src.shape()) {
        throw std::invalid_argument("Causal attention gradient shape mismatch");
    }
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] += src[i];
    }
}

} // namespace

Value scaled_dot_product_attention_causal(const Value& query,
                                          const Value& key,
                                          const Value& value,
                                          double scale) {
    if (!query.node_ || !key.node_ || !value.node_) {
        throw std::logic_error("Invalid autograd Value");
    }
    if (query.node_->data.ndim() != 2 || key.node_->data.ndim() != 2 || value.node_->data.ndim() != 2) {
        throw std::invalid_argument("Causal attention requires 2D tensors");
    }
    const auto& query_shape = query.node_->data.shape();
    const auto& key_shape = key.node_->data.shape();
    const auto& value_shape = value.node_->data.shape();
    if (query_shape[1] != key_shape[1] || key_shape[0] != value_shape[0] || !std::isfinite(scale) || scale <= 0.0) {
        throw std::invalid_argument("Causal attention shape or scale mismatch");
    }

    const std::size_t query_rows = query_shape[0];
    const std::size_t key_rows = key_shape[0];
    const std::size_t key_dim = key_shape[1];
    const std::size_t value_dim = value_shape[1];
    Tensor probabilities({query_rows, key_rows}, 0.0);
    Tensor output({query_rows, value_dim}, 0.0);

    // 因果掩码：第 i 个位置只能看到第 0..i 个位置。
    for (std::size_t row = 0; row < query_rows; ++row) {
        const std::size_t visible = std::min(row + 1, key_rows);
        double max_score = -std::numeric_limits<double>::infinity();
        for (std::size_t column = 0; column < visible; ++column) {
            double score = 0.0;
            for (std::size_t dim = 0; dim < key_dim; ++dim) {
                score += query.node_->data.at({row, dim}) * key.node_->data.at({column, dim});
            }
            max_score = std::max(max_score, score * scale);
        }
        double sum_exp = 0.0;
        for (std::size_t column = 0; column < visible; ++column) {
            double score = 0.0;
            for (std::size_t dim = 0; dim < key_dim; ++dim) {
                score += query.node_->data.at({row, dim}) * key.node_->data.at({column, dim});
            }
            const double probability = std::exp(score * scale - max_score);
            probabilities.at({row, column}) = probability;
            sum_exp += probability;
        }
        for (std::size_t column = 0; column < visible; ++column) {
            probabilities.at({row, column}) /= sum_exp;
            for (std::size_t dim = 0; dim < value_dim; ++dim) {
                output.at({row, dim}) += probabilities.at({row, column}) * value.node_->data.at({column, dim});
            }
        }
    }

    auto node = std::make_shared<Value::Node>();
    node->data = std::move(output);
    node->requires_grad = query.requires_grad() || key.requires_grad() || value.requires_grad();
    node->parents = {query.node_, key.node_, value.node_};
    if (node->requires_grad) {
        node->grad = zeros_like(node->data);
    }

    node->backward = [query_node = query.node_, key_node = key.node_, value_node = value.node_, probabilities = std::move(probabilities), scale](Value::Node& self) {
        const std::size_t query_rows = query_node->data.shape()[0];
        const std::size_t key_rows = key_node->data.shape()[0];
        const std::size_t key_dim = key_node->data.shape()[1];
        const std::size_t value_dim = value_node->data.shape()[1];
        Tensor query_grad(query_node->data.shape(), 0.0);
        Tensor key_grad(key_node->data.shape(), 0.0);
        Tensor value_grad(value_node->data.shape(), 0.0);
        Tensor probability_grad({query_rows, key_rows}, 0.0);
        Tensor score_grad({query_rows, key_rows}, 0.0);

        for (std::size_t row = 0; row < query_rows; ++row) {
            const std::size_t visible = std::min(row + 1, key_rows);
            for (std::size_t column = 0; column < visible; ++column) {
                for (std::size_t dim = 0; dim < value_dim; ++dim) {
                    probability_grad.at({row, column}) += self.grad.at({row, dim}) * value_node->data.at({column, dim});
                    value_grad.at({column, dim}) += probabilities.at({row, column}) * self.grad.at({row, dim});
                }
            }
            double weighted_sum = 0.0;
            for (std::size_t column = 0; column < visible; ++column) {
                weighted_sum += probability_grad.at({row, column}) * probabilities.at({row, column});
            }
            for (std::size_t column = 0; column < visible; ++column) {
                score_grad.at({row, column}) = probabilities.at({row, column}) * (probability_grad.at({row, column}) - weighted_sum) * scale;
            }
        }

        for (std::size_t row = 0; row < query_rows; ++row) {
            const std::size_t visible = std::min(row + 1, key_rows);
            for (std::size_t column = 0; column < visible; ++column) {
                for (std::size_t dim = 0; dim < key_dim; ++dim) {
                    query_grad.at({row, dim}) += score_grad.at({row, column}) * key_node->data.at({column, dim});
                    key_grad.at({column, dim}) += score_grad.at({row, column}) * query_node->data.at({row, dim});
                }
            }
        }

        if (query_node->requires_grad) {
            add_inplace(query_node->grad, query_grad);
        }
        if (key_node->requires_grad) {
            add_inplace(key_node->grad, key_grad);
        }
        if (value_node->requires_grad) {
            add_inplace(value_node->grad, value_grad);
        }
    };
    return Value(std::move(node));
}

} // namespace nexmind
