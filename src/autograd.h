#pragma once

#include "tensor.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nexmind {

class Value {
public:
    struct Node {
        Tensor data;
        Tensor grad;
        bool requires_grad = false;
        std::string name;
        std::vector<std::shared_ptr<Node>> parents;
        std::function<void(Node&)> backward;
    };

    Value() = default;
    explicit Value(Tensor data, bool requires_grad = false, std::string name = {});

    const Tensor& data() const;
    Tensor& data();
    const Tensor& grad() const;
    bool requires_grad() const noexcept;
    const std::string& name() const noexcept;

    void zero_grad();
    void backward();

private:
    explicit Value(std::shared_ptr<Node> node);
    std::shared_ptr<Node> node_;

    friend Value add(const Value& lhs, const Value& rhs);
    friend Value add_bias(const Value& input, const Value& bias);
    friend Value multiply(const Value& lhs, const Value& rhs);
    friend Value matmul(const Value& lhs, const Value& rhs);
    friend Value mean(const Value& input);
    friend Value cross_entropy(const Value& logits, const std::vector<std::size_t>& targets);
    friend Value relu(const Value& input);
};

Value add(const Value& lhs, const Value& rhs);
Value add_bias(const Value& input, const Value& bias);
Value multiply(const Value& lhs, const Value& rhs);
Value matmul(const Value& lhs, const Value& rhs);
Value mean(const Value& input);
Value cross_entropy(const Value& logits, const std::vector<std::size_t>& targets);
Value relu(const Value& input);

} // namespace nexmind
