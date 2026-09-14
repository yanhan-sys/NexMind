#include "relu.h"

#include <algorithm>
#include <stdexcept>

namespace nexmind {

Value relu(const Value& input) {
    const Tensor& input_data = input.data();
    Tensor output(input_data.shape());
    for (std::size_t i = 0; i < input_data.size(); ++i) {
        output[i] = std::max(0.0, input_data[i]);
    }

    if (!input.requires_grad()) {
        return Value(std::move(output));
    }

    Value result(std::move(output), true);
    const auto input_node = input.node_;
    const auto result_node = result.node_;
    result_node->parents = {input_node};
    result_node->backward = [input_node, result_node](Value::Node& node) {
        if (!input_node->requires_grad) {
            return;
        }
        if (input_node->grad.size() == 0) {
            input_node->grad = Tensor(input_node->data.shape(), 0.0);
        }
        for (std::size_t i = 0; i < input_node->data.size(); ++i) {
            input_node->grad[i] += input_node->data[i] > 0.0 ? node.grad[i] : 0.0;
        }
    };
    return result;
}

} // namespace nexmind
