#include "autograd.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace nexmind {

namespace {

Tensor zeros_like(const Tensor& input) {
    return Tensor(input.shape(), 0.0);
}

void add_inplace(Tensor& dst, const Tensor& src) {
    if (dst.shape() != src.shape()) {
        throw std::invalid_argument("Autograd gradient shape mismatch");
    }
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] += src[i];
    }
}

void backward_node(const std::shared_ptr<Value::Node>& node, std::unordered_set<Value::Node*>& visited, std::vector<std::shared_ptr<Value::Node>>& order) {
    if (!node || visited.find(node.get()) != visited.end()) {
        return;
    }
    visited.insert(node.get());
    for (const auto& parent : node->parents) {
        backward_node(parent, visited, order);
    }
    order.push_back(node);
}

} // namespace

Value::Value(Tensor data, bool requires_grad, std::string name) : node_(std::make_shared<Node>()) {
    node_->data = std::move(data);
    node_->requires_grad = requires_grad;
    node_->name = std::move(name);
    if (requires_grad) {
        node_->grad = zeros_like(node_->data);
    }
}

Value::Value(std::shared_ptr<Node> node) : node_(std::move(node)) {}

const Tensor& Value::data() const {
    if (!node_) {
        throw std::logic_error("Invalid autograd Value");
    }
    return node_->data;
}

Tensor& Value::data() {
    if (!node_) {
        throw std::logic_error("Invalid autograd Value");
    }
    return node_->data;
}

const Tensor& Value::grad() const {
    if (!node_ || !node_->requires_grad) {
        throw std::logic_error("Gradient is not enabled");
    }
    return node_->grad;
}

bool Value::requires_grad() const noexcept {
    return node_ && node_->requires_grad;
}

const std::string& Value::name() const noexcept {
    static const std::string empty;
    return node_ ? node_->name : empty;
}

void Value::zero_grad() {
    if (requires_grad()) {
        node_->grad.fill(0.0);
    }
}

void Value::backward() {
    if (!node_ || !node_->requires_grad) {
        throw std::logic_error("Backward requires a gradient-enabled output");
    }
    if (node_->data.size() != 1) {
        throw std::invalid_argument("Backward output must contain exactly one value");
    }
    std::unordered_set<Node*> visited;
    std::vector<std::shared_ptr<Node>> order;
    backward_node(node_, visited, order);
    for (const auto& current : order) {
        if (current->requires_grad) {
            current->grad.fill(0.0);
        }
    }
    node_->grad[0] = 1.0;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if ((*it)->backward) {
            (*it)->backward(*(*it));
        }
    }
}

Value add(const Value& lhs, const Value& rhs) {
    if (!lhs.node_ || !rhs.node_) {
        throw std::logic_error("Invalid autograd Value");
    }
    auto node = std::make_shared<Value::Node>();
    node->data = nexmind::add(lhs.node_->data, rhs.node_->data);
    node->requires_grad = lhs.requires_grad() || rhs.requires_grad();
    node->parents = {lhs.node_, rhs.node_};
    if (node->requires_grad) {
        node->grad = zeros_like(node->data);
    }
    node->backward = [left = lhs.node_, right = rhs.node_](Value::Node& self) {
        if (left->requires_grad) {
            add_inplace(left->grad, self.grad);
        }
        if (right->requires_grad) {
            add_inplace(right->grad, self.grad);
        }
    };
    return Value(std::move(node));
}

Value add_bias(const Value& input, const Value& bias) {
    if (!input.node_ || !bias.node_) {
        throw std::logic_error("Invalid autograd Value");
    }
    auto node = std::make_shared<Value::Node>();
    node->data = nexmind::add_row_bias(input.node_->data, bias.node_->data);
    node->requires_grad = input.requires_grad() || bias.requires_grad();
    node->parents = {input.node_, bias.node_};
    if (node->requires_grad) {
        node->grad = zeros_like(node->data);
    }
    node->backward = [input_node = input.node_, bias_node = bias.node_](Value::Node& self) {
        if (input_node->requires_grad) {
            add_inplace(input_node->grad, self.grad);
        }
        if (bias_node->requires_grad) {
            Tensor bias_grad(bias_node->data.shape(), 0.0);
            for (std::size_t row = 0; row < self.grad.shape()[0]; ++row) {
                for (std::size_t column = 0; column < self.grad.shape()[1]; ++column) {
                    bias_grad.at({0, column}) += self.grad.at({row, column});
                }
            }
            add_inplace(bias_node->grad, bias_grad);
        }
    };
    return Value(std::move(node));
}

Value multiply(const Value& lhs, const Value& rhs) {
    if (!lhs.node_ || !rhs.node_) {
        throw std::logic_error("Invalid autograd Value");
    }
    auto node = std::make_shared<Value::Node>();
    node->data = nexmind::multiply(lhs.node_->data, rhs.node_->data);
    node->requires_grad = lhs.requires_grad() || rhs.requires_grad();
    node->parents = {lhs.node_, rhs.node_};
    if (node->requires_grad) {
        node->grad = zeros_like(node->data);
    }
    node->backward = [left = lhs.node_, right = rhs.node_](Value::Node& self) {
        if (left->requires_grad) {
            add_inplace(left->grad, nexmind::multiply(self.grad, right->data));
        }
        if (right->requires_grad) {
            add_inplace(right->grad, nexmind::multiply(self.grad, left->data));
        }
    };
    return Value(std::move(node));
}

Value matmul(const Value& lhs, const Value& rhs) {
    if (!lhs.node_ || !rhs.node_) {
        throw std::logic_error("Invalid autograd Value");
    }
    auto node = std::make_shared<Value::Node>();
    node->data = nexmind::matmul(lhs.node_->data, rhs.node_->data);
    node->requires_grad = lhs.requires_grad() || rhs.requires_grad();
    node->parents = {lhs.node_, rhs.node_};
    if (node->requires_grad) {
        node->grad = zeros_like(node->data);
    }
    node->backward = [left = lhs.node_, right = rhs.node_](Value::Node& self) {
        if (left->requires_grad) {
            add_inplace(left->grad, nexmind::matmul(self.grad, nexmind::transpose_2d(right->data)));
        }
        if (right->requires_grad) {
            add_inplace(right->grad, nexmind::matmul(nexmind::transpose_2d(left->data), self.grad));
        }
    };
    return Value(std::move(node));
}

Value mean(const Value& input) {
    if (!input.node_ || input.node_->data.size() == 0) {
        throw std::invalid_argument("Mean requires a non-empty Tensor");
    }
    auto node = std::make_shared<Value::Node>();
    const Tensor& input_data = input.node_->data;
    node->data = Tensor({1}, std::accumulate(input_data.data(), input_data.data() + input_data.size(), 0.0) / static_cast<double>(input_data.size()));
    node->requires_grad = input.requires_grad();
    node->parents = {input.node_};
    if (node->requires_grad) {
        node->grad = zeros_like(node->data);
    }
    const double scale = 1.0 / static_cast<double>(input.node_->data.size());
    node->backward = [parent = input.node_, scale](Value::Node& self) {
        if (parent->requires_grad) {
            Tensor grad(parent->data.shape(), self.grad[0] * scale);
            add_inplace(parent->grad, grad);
        }
    };
    return Value(std::move(node));
}

Value cross_entropy(const Value& logits, const std::vector<std::size_t>& targets) {
    if (!logits.node_ || logits.node_->data.ndim() != 2) {
        throw std::invalid_argument("Cross entropy requires a 2D logits Tensor");
    }
    const auto& shape = logits.node_->data.shape();
    const std::size_t batch = shape[0];
    const std::size_t classes = shape[1];
    if (batch == 0 || classes == 0 || targets.size() != batch) {
        throw std::invalid_argument("Cross entropy batch or target shape mismatch");
    }
    for (std::size_t i = 0; i < batch; ++i) {
        if (targets[i] >= classes) {
            throw std::out_of_range("Cross entropy target out of range");
        }
    }

    auto node = std::make_shared<Value::Node>();
    Tensor probabilities(shape, 0.0);
    double loss = 0.0;
    for (std::size_t row = 0; row < batch; ++row) {
        double max_logit = -std::numeric_limits<double>::infinity();
        for (std::size_t col = 0; col < classes; ++col) {
            max_logit = std::max(max_logit, logits.node_->data.at({row, col}));
        }
        double sum_exp = 0.0;
        for (std::size_t col = 0; col < classes; ++col) {
            const double value = std::exp(logits.node_->data.at({row, col}) - max_logit);
            probabilities.at({row, col}) = value;
            sum_exp += value;
        }
        for (std::size_t col = 0; col < classes; ++col) {
            probabilities.at({row, col}) /= sum_exp;
        }
        loss += -std::log(probabilities.at({row, targets[row]}));
    }
    loss /= static_cast<double>(batch);
    node->data = Tensor({1}, loss);
    node->requires_grad = logits.requires_grad();
    node->parents = {logits.node_};
    if (node->requires_grad) {
        node->grad = zeros_like(node->data);
    }
    node->backward = [parent = logits.node_, probabilities = std::move(probabilities), targets, batch](Value::Node& self) {
        if (!parent->requires_grad) {
            return;
        }
        Tensor grad = probabilities;
        for (std::size_t row = 0; row < batch; ++row) {
            grad.at({row, targets[row]}) -= 1.0;
        }
        const double scale = self.grad[0] / static_cast<double>(batch);
        for (std::size_t i = 0; i < grad.size(); ++i) {
            grad[i] *= scale;
        }
        add_inplace(parent->grad, grad);
    };
    return Value(std::move(node));
}

Value relu(const Value& input) {
    if (!input.node_) {
        throw std::logic_error("Invalid autograd Value");
    }
    auto node = std::make_shared<Value::Node>();
    node->data = input.node_->data;
    for (std::size_t i = 0; i < node->data.size(); ++i) {
        node->data[i] = std::max(0.0, node->data[i]);
    }
    node->requires_grad = input.requires_grad();
    node->parents = {input.node_};
    if (node->requires_grad) {
        node->grad = zeros_like(node->data);
    }
    node->backward = [parent = input.node_](Value::Node& self) {
        if (!parent->requires_grad) {
            return;
        }
        Tensor grad(parent->data.shape(), 0.0);
        for (std::size_t i = 0; i < grad.size(); ++i) {
            if (parent->data[i] > 0.0) {
                grad[i] = self.grad[i];
            }
        }
        add_inplace(parent->grad, grad);
    };
    return Value(std::move(node));
}

Value layer_norm(const Value& input, const Value& gamma, const Value& beta, double epsilon) {
    if (!input.node_ || !gamma.node_ || !beta.node_ || input.node_->data.ndim() != 2 || gamma.node_->data.shape().size() != 2 || beta.node_->data.shape().size() != 2) {
        throw std::invalid_argument("LayerNorm requires 2D input and parameters");
    }
    const auto& input_shape = input.node_->data.shape();
    const auto& gamma_shape = gamma.node_->data.shape();
    const auto& beta_shape = beta.node_->data.shape();
    const std::size_t rows = input_shape[0];
    const std::size_t features = input_shape[1];
    if (gamma_shape[0] != 1 || beta_shape[0] != 1 || gamma_shape[1] != features || beta_shape[1] != features || epsilon <= 0.0) {
        throw std::invalid_argument("LayerNorm shape mismatch");
    }

    Tensor normalized(input_shape, 0.0);
    Tensor inverse_std({rows, 1}, 0.0);
    Tensor output(input_shape, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        double mean_value = 0.0;
        for (std::size_t column = 0; column < features; ++column) {
            mean_value += input.node_->data.at({row, column});
        }
        mean_value /= static_cast<double>(features);
        double variance = 0.0;
        for (std::size_t column = 0; column < features; ++column) {
            const double centered = input.node_->data.at({row, column}) - mean_value;
            variance += centered * centered;
        }
        variance /= static_cast<double>(features);
        const double inv = 1.0 / std::sqrt(variance + epsilon);
        inverse_std.at({row, 0}) = inv;
        for (std::size_t column = 0; column < features; ++column) {
            const double normalized_value = (input.node_->data.at({row, column}) - mean_value) * inv;
            normalized.at({row, column}) = normalized_value;
            output.at({row, column}) = normalized_value * gamma.node_->data.at({0, column}) + beta.node_->data.at({0, column});
        }
    }

    auto node = std::make_shared<Value::Node>();
    node->data = std::move(output);
    node->requires_grad = input.requires_grad() || gamma.requires_grad() || beta.requires_grad();
    node->parents = {input.node_, gamma.node_, beta.node_};
    if (node->requires_grad) {
        node->grad = zeros_like(node->data);
    }
    node->backward = [input_node = input.node_, gamma_node = gamma.node_, beta_node = beta.node_, normalized = std::move(normalized), inverse_std = std::move(inverse_std)](Value::Node& self) {
        const std::size_t rows = input_node->data.shape()[0];
        const std::size_t features = input_node->data.shape()[1];
        Tensor input_grad(input_node->data.shape(), 0.0);
        Tensor gamma_grad(gamma_node->data.shape(), 0.0);
        Tensor beta_grad(beta_node->data.shape(), 0.0);
        for (std::size_t row = 0; row < rows; ++row) {
            double sum_dy = 0.0;
            double sum_dy_xhat = 0.0;
            for (std::size_t column = 0; column < features; ++column) {
                const double dy = self.grad.at({row, column}) * gamma_node->data.at({0, column});
                sum_dy += dy;
                sum_dy_xhat += dy * normalized.at({row, column});
                gamma_grad.at({0, column}) += self.grad.at({row, column}) * normalized.at({row, column});
                beta_grad.at({0, column}) += self.grad.at({row, column});
            }
            const double scale = inverse_std.at({row, 0}) / static_cast<double>(features);
            for (std::size_t column = 0; column < features; ++column) {
                const double dy = self.grad.at({row, column}) * gamma_node->data.at({0, column});
                input_grad.at({row, column}) = scale * (static_cast<double>(features) * dy - sum_dy - normalized.at({row, column}) * sum_dy_xhat);
            }
        }
        if (input_node->requires_grad) {
            add_inplace(input_node->grad, input_grad);
        }
        if (gamma_node->requires_grad) {
            add_inplace(gamma_node->grad, gamma_grad);
        }
        if (beta_node->requires_grad) {
            add_inplace(beta_node->grad, beta_grad);
        }
    };
    return Value(std::move(node));
}

Value scaled_dot_product_attention(const Value& query, const Value& key, const Value& value, double scale) {
    // 第一版只支持单头、二维序列，先把计算和反向传播做正确。
    if (!query.node_ || !key.node_ || !value.node_) {
        throw std::logic_error("Invalid autograd Value");
    }
    if (query.node_->data.ndim() != 2 || key.node_->data.ndim() != 2 || value.node_->data.ndim() != 2) {
        throw std::invalid_argument("Attention requires 2D query, key and value");
    }
    const auto& query_shape = query.node_->data.shape();
    const auto& key_shape = key.node_->data.shape();
    const auto& value_shape = value.node_->data.shape();
    if (query_shape[1] != key_shape[1] || key_shape[0] != value_shape[0] || query_shape[1] == 0 || key_shape[0] == 0 || scale <= 0.0) {
        throw std::invalid_argument("Attention shape or scale mismatch");
    }

    const std::size_t query_length = query_shape[0];
    const std::size_t key_length = key_shape[0];
    const std::size_t key_dim = key_shape[1];
    const std::size_t value_dim = value_shape[1];

    Tensor probabilities({query_length, key_length}, 0.0);
    for (std::size_t row = 0; row < query_length; ++row) {
        double max_score = -std::numeric_limits<double>::infinity();
        Tensor scores({1, key_length}, 0.0);
        for (std::size_t column = 0; column < key_length; ++column) {
            double score_value = 0.0;
            for (std::size_t feature = 0; feature < key_dim; ++feature) {
                score_value += query.node_->data.at({row, feature}) * key.node_->data.at({column, feature});
            }
            score_value *= scale;
            scores.at({0, column}) = score_value;
            max_score = std::max(max_score, score_value);
        }
        double sum_exp = 0.0;
        for (std::size_t column = 0; column < key_length; ++column) {
            const double value_exp = std::exp(scores.at({0, column}) - max_score);
            probabilities.at({row, column}) = value_exp;
            sum_exp += value_exp;
        }
        for (std::size_t column = 0; column < key_length; ++column) {
            probabilities.at({row, column}) /= sum_exp;
        }
    }

    Tensor output({query_length, value_dim}, 0.0);
    for (std::size_t row = 0; row < query_length; ++row) {
        for (std::size_t column = 0; column < value_dim; ++column) {
            for (std::size_t source = 0; source < key_length; ++source) {
                output.at({row, column}) += probabilities.at({row, source}) * value.node_->data.at({source, column});
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
        const std::size_t query_length = query_node->data.shape()[0];
        const std::size_t key_length = key_node->data.shape()[0];
        const std::size_t key_dim = key_node->data.shape()[1];
        const std::size_t value_dim = value_node->data.shape()[1];
        Tensor probability_grad({query_length, key_length}, 0.0);
        Tensor value_grad(value_node->data.shape(), 0.0);

        // 先计算 dV 和 dP，再通过 softmax 的雅可比矩阵得到 dScore。
        for (std::size_t row = 0; row < query_length; ++row) {
            for (std::size_t source = 0; source < key_length; ++source) {
                for (std::size_t column = 0; column < value_dim; ++column) {
                    probability_grad.at({row, source}) += self.grad.at({row, column}) * value_node->data.at({source, column});
                    value_grad.at({source, column}) += probabilities.at({row, source}) * self.grad.at({row, column});
                }
            }
        }

        Tensor score_grad({query_length, key_length}, 0.0);
        for (std::size_t row = 0; row < query_length; ++row) {
            double weighted_sum = 0.0;
            for (std::size_t source = 0; source < key_length; ++source) {
                weighted_sum += probability_grad.at({row, source}) * probabilities.at({row, source});
            }
            for (std::size_t source = 0; source < key_length; ++source) {
                score_grad.at({row, source}) = probabilities.at({row, source}) * (probability_grad.at({row, source}) - weighted_sum);
            }
        }

        Tensor query_grad(query_node->data.shape(), 0.0);
        Tensor key_grad(key_node->data.shape(), 0.0);
        for (std::size_t row = 0; row < query_length; ++row) {
            for (std::size_t source = 0; source < key_length; ++source) {
                const double score = score_grad.at({row, source}) * scale;
                for (std::size_t feature = 0; feature < key_dim; ++feature) {
                    query_grad.at({row, feature}) += score * key_node->data.at({source, feature});
                    key_grad.at({source, feature}) += score * query_node->data.at({row, feature});
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
