#include "positional_encoding.h"

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
        // 零输入便于直接观察固定位置编码的数值变化。
        SinusoidalPositionalEncoding encoding(4);
        Value input(Tensor({2, 4}, 0.0), true);
        const Value output = encoding.forward(input);

        expect(std::abs(output.data().at({0, 0})) < 1e-12, "Position 0 sine mismatch");
        expect(std::abs(output.data().at({0, 1}) - 1.0) < 1e-12, "Position 0 cosine mismatch");
        expect(std::abs(output.data().at({1, 0}) - std::sin(1.0)) < 1e-12, "Position 1 sine mismatch");
        expect(std::abs(output.data().at({1, 1}) - std::cos(1.0)) < 1e-12, "Position 1 cosine mismatch");
        expect(std::abs(output.data().at({0, 0}) - output.data().at({1, 0})) > 1e-6, "Positions are indistinguishable");

        Value loss = mean(output);
        loss.backward();
        expect(std::isfinite(input.grad()[0]), "Positional encoding gradient is not finite");

        std::cout << "Positional Encoding test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Positional Encoding test failed: " << error.what() << '\n';
        return 1;
    }
}
