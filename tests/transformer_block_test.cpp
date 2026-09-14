#include "transformer_block.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace nexmind;

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    try {
        // 使用小尺寸验证完整 Transformer Block 的前向和反向传播。
        TransformerBlock block(8, 2, 16, 42);
        Tensor input_data({3, 8}, 0.0);
        for (std::size_t i = 0; i < input_data.size(); ++i) {
            input_data[i] = 0.05 * static_cast<double>(i + 1);
        }
        Value input(std::move(input_data), true);

        const Value output = block.forward(input);
        expect(output.data().shape() == std::vector<std::size_t>({3, 8}), "Transformer Block output shape mismatch");
        expect(block.parameters().size() == 22, "Transformer Block parameter count mismatch");

        const Value loss = mean(output);
        loss.backward();

        bool parameter_gradient_found = false;
        for (Parameter* parameter : block.parameters()) {
            expect(parameter != nullptr, "Transformer Block returned null parameter");
            expect(parameter->value().requires_grad(), "Transformer Block parameter must require gradients");
            const Tensor& gradient = parameter->value().grad();
            for (std::size_t i = 0; i < gradient.size(); ++i) {
                expect(std::isfinite(gradient[i]), "Transformer Block gradient must be finite");
                if (std::abs(gradient[i]) > 1e-12) {
                    parameter_gradient_found = true;
                }
            }
        }
        expect(parameter_gradient_found, "Transformer Block parameters did not receive gradients");
        expect(input.grad().size() == input.data().size(), "Transformer Block input gradient shape mismatch");
        for (std::size_t i = 0; i < input.grad().size(); ++i) {
            expect(std::isfinite(input.grad()[i]), "Transformer Block input gradient must be finite");
        }

        std::cout << "Transformer Block test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Transformer Block test failed: " << error.what() << '\n';
        return 1;
    }
}
