#include "attention.h"

#include <cassert>
#include <cmath>
#include <cstddef>

int main() {
    // 用三个 token 的小序列验证注意力输出形状和完整反向传播链路。
    nexmind::Attention attention(4, 123);
    nexmind::Tensor input_tensor({3, 4});
    for (std::size_t i = 0; i < input_tensor.size(); ++i) {
        input_tensor[i] = 0.1 * static_cast<double>(i + 1);
    }
    nexmind::Value input(std::move(input_tensor), true, "input");

    const nexmind::Value output = attention.forward(input);
    assert(output.data().shape().size() == 2);
    assert(output.data().shape()[0] == 3);
    assert(output.data().shape()[1] == 4);

    const nexmind::Value loss = nexmind::mean(output);
    loss.backward();

    bool input_gradient_nonzero = false;
    for (std::size_t i = 0; i < input.grad().size(); ++i) {
        assert(std::isfinite(input.grad()[i]));
        input_gradient_nonzero = input_gradient_nonzero || std::abs(input.grad()[i]) > 1e-12;
    }
    assert(input_gradient_nonzero);

    const auto parameters = attention.parameters();
    assert(parameters.size() == 8);
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
