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

} // namespace nexmind
