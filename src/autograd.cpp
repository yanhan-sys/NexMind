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
            (*it)->backward(*it);
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
    node->data = Tensor({1}, 0.0);
    node->data[0] = std::accumulate(input.node_->data.data(), input.node_->data.data() + input.node_->data.size(), 0.0) / static_cast<double>(input.node_->data.size());
    node->requires_grad = input.requires_grad();
    node->parents = {input.node_};
    if (node->requires_grad) {
        node->grad = zeros_like(node->data);
    }
    node->backward = [parent = input.node_](Value::Node& self) {
        if (parent->requires_grad) {
            const double factor = self.grad[0] / static_cast<double>(parent->data.size());
            add_inplace(parent->grad, Tensor(parent->data.shape(), factor));
        }
    };
    return Value(std::move(node));
}

Value cross_entropy(const Value& logits, const std::vector<std::size_t>& targets) {
    if (!logits.node_ || logits.node_->data.ndim() != 2) {
        throw std::invalid_argument("Cross entropy requires a 2D logits Tensor");
    }
    const std::size_t batch = logits.node_->data.shape()[0];
    const std::size_t classes = logits.node_->data.shape()[1];
    if (batch == 0 || targets.size() != batch) {
        throw std::invalid_argument("Cross entropy target size mismatch");
    }
    auto node = std::make_shared<Value::Node>();
    node->data = Tensor({1}, 0.0);
    node->requires_grad = logits.requires_grad();
    node->parents = {logits.node_};
    if (node->requires_grad) {
        node->grad = zeros_like(node->data);
    }

    std::vector<double> probabilities(batch * classes);
    double loss = 0.0;
    for (std::size_t i = 0; i < batch; ++i) {
        if (targets[i] >= classes) {
            throw std::out_of_range("Cross entropy target out of range");
        }
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < classes; ++j) {
            maximum = std::max(maximum, logits.node_->data.at({i, j}));
        }
        double sum = 0.0;
        for (std::size_t j = 0; j < classes; ++j) {
            probabilities[i * classes + j] = std::exp(logits.node_->data.at({i, j}) - maximum);
            sum += probabilities[i * classes + j];
        }
        for (std::size_t j = 0; j < classes; ++j) {
            probabilities[i * classes + j] /= sum;
        }
        loss -= std::log(std::max(probabilities[i * classes + targets[i]], 1e-300));
    }
    node->data[0] = loss / static_cast<double>(batch);
    node->backward = [parent = logits.node_, probabilities = std::move(probabilities), targets, batch, classes](Value::Node& self) {
        if (!parent->requires_grad) {
            return;
        }
        Tensor gradient(parent->data.shape());
        const double scale_factor = self.grad[0] / static_cast<double>(batch);
        for (std::size_t i = 0; i < batch; ++i) {
            for (std::size_t j = 0; j < classes; ++j) {
                double value = probabilities[i * classes + j];
                if (j == targets[i]) {
                    value -= 1.0;
                }
                gradient.at({i, j}) = value * scale_factor;
            }
        }
        add_inplace(parent->grad, gradient);
    };
    return Value(std::move(node));
}

} // namespace nexmind
