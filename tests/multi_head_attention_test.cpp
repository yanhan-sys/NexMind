#include "multi_head_attention.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>

int main() {
    // 用四个 token、两个头验证多头注意力的形状和完整反向传播。
    nexmind::MultiHeadAttention attention(4, 2, 321);
    nexmind::Tensor input_tensor({4, 4});
    for (std::size_t i = 0; i < input_tensor.size(); ++i) {
        input_tensor[i] = 0.05 * static_cast<double>(i + 1);
    }
    nexmind::Value input(std::move(input_tensor), true, "input");

    const nexmind::Value output = attention.forward(input);
    assert(output.data().shape().size() == 2);
    assert(output.data().shape()[0] == 4);
    assert(output.data().shape()[1] == 4);

    nexmind::Value loss = nexmind::mean(output);
    loss.backward();

    bool input_gradient_nonzero = false;
    for (std::size_t i = 0; i < input.grad().size(); ++i) {
        assert(std::isfinite(input.grad()[i]));
        input_gradient_nonzero = input_gradient_nonzero || std::abs(input.grad()[i]) > 1e-12;
    }
    assert(input_gradient_nonzero);

    const auto parameters = attention.parameters();
    // 每个头有 Q/K/V 三个 Linear，共 6 个参数；输出投影再提供 2 个。
    assert(parameters.size() == 14);
    for (const auto* parameter : parameters) {
        bool gradient_nonzero = false;
        for (std::size_t i = 0; i < parameter->value().grad().size(); ++i) {
            assert(std::isfinite(parameter->value().grad()[i]));
            gradient_nonzero = gradient_nonzero || std::abs(parameter->value().grad()[i]) > 1e-12;
        }
        assert(gradient_nonzero);
    }

    return 0;
}
