#include "relu.h"

#include <algorithm>
#include <utility>

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
    result.node_->parents = {input_node};
    result.node_->backward = [input_node](Value::Node& self) {
        if (!input_node->requires_grad) {
            return;
        }
        for (std::size_t i = 0; i < input_node->data.size(); ++i) {
            if (input_node->data[i] > 0.0) {
                input_node->grad[i] += self.grad[i];
            }
        }
    };
    return result;
}

} // namespace nexmind
