#include "transformer_encoder.h"

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
        // 使用两层 Encoder 验证堆叠、参数注册和完整反向传播。
        TransformerEncoder encoder(2, 8, 2, 16, 42);
        Tensor input_data({3, 8}, 0.0);
        for (std::size_t i = 0; i < input_data.size(); ++i) {
            input_data[i] = 0.02 * static_cast<double>(i + 1);
        }
        Value input(std::move(input_data), true);

        const Value output = encoder.forward(input);
        expect(encoder.num_layers() == 2, "Transformer Encoder layer count mismatch");
        expect(output.data().shape() == std::vector<std::size_t>({3, 8}), "Transformer Encoder output shape mismatch");
        expect(encoder.parameters().size() == 44, "Transformer Encoder parameter count mismatch");

        Value loss = mean(output);
        loss.backward();

        bool parameter_gradient_found = false;
        for (Parameter* parameter : encoder.parameters()) {
            expect(parameter != nullptr, "Transformer Encoder returned null parameter");
            expect(parameter->value().requires_grad(), "Transformer Encoder parameter must require gradients");
            const Tensor& gradient = parameter->value().grad();
            for (std::size_t i = 0; i < gradient.size(); ++i) {
                expect(std::isfinite(gradient[i]), "Transformer Encoder gradient must be finite");
                if (std::abs(gradient[i]) > 1e-12) {
                    parameter_gradient_found = true;
                }
            }
        }
        expect(parameter_gradient_found, "Transformer Encoder parameters did not receive gradients");
        expect(input.grad().size() == input.data().size(), "Transformer Encoder input gradient shape mismatch");
        for (std::size_t i = 0; i < input.grad().size(); ++i) {
            expect(std::isfinite(input.grad()[i]), "Transformer Encoder input gradient must be finite");
        }

        std::cout << "Transformer Encoder test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Transformer Encoder test failed: " << error.what() << '\n';
        return 1;
    }
}
