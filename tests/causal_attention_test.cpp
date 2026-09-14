#include "autograd.h"

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
        // 用相同 score 构造均匀注意力，验证每个位置只能看到当前及之前的位置。
        Tensor query_data({3, 1}, 0.0);
        Tensor key_data({3, 1}, 0.0);
        Tensor value_data({3, 1}, 0.0);
        value_data[0] = 1.0;
        value_data[1] = 10.0;
        value_data[2] = 100.0;
        Value query(std::move(query_data), true);
        Value key(std::move(key_data), true);
        Value value(std::move(value_data), true);

        Value output = scaled_dot_product_attention_causal(query, key, value, 1.0);
        expect(std::abs(output.data().at({0, 0}) - 1.0) < 1e-12, "Causal row 0 can see a future value");
        expect(std::abs(output.data().at({1, 0}) - 5.5) < 1e-12, "Causal row 1 mask mismatch");
        expect(std::abs(output.data().at({2, 0}) - 37.0) < 1e-12, "Causal row 2 mask mismatch");

        Value loss = mean(output);
        loss.backward();
        expect(std::isfinite(query.grad()[0]), "Causal query gradient is not finite");
        expect(std::isfinite(key.grad()[0]), "Causal key gradient is not finite");
        expect(std::isfinite(value.grad()[0]), "Causal value gradient is not finite");

        std::cout << "Causal Attention test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Causal Attention test failed: " << error.what() << '\n';
        return 1;
    }
}
