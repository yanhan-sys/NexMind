#include "mlp.h"
#include "trainer.h"

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
        // 用 XOR 数据验证通用 Trainer 的完整训练闭环。
        MLP model(2, 8, 1, 42);
        Trainer trainer(model.parameters(), 0.05);

        Tensor x_data({4, 2}, 0.0);
        x_data[0] = 0.0; x_data[1] = 0.0;
        x_data[2] = 0.0; x_data[3] = 1.0;
        x_data[4] = 1.0; x_data[5] = 0.0;
        x_data[6] = 1.0; x_data[7] = 1.0;
        Value inputs(std::move(x_data), false);

        Tensor y_data({4, 1}, 0.0);
        y_data[0] = 0.0;
        y_data[1] = 1.0;
        y_data[2] = 1.0;
        y_data[3] = 0.0;
        Value targets(std::move(y_data), false);

        const auto make_loss = [&]() {
            const Value prediction = model.forward(inputs);
            const Value error = multiply(add(prediction, multiply(targets, Value(Tensor({4, 1}, 2.0), false))), Value(Tensor({4, 1}, 0.5), false));
            return mean(multiply(error, error));
        };

        const double initial_loss = make_loss().data()[0];
        double final_loss = initial_loss;
        for (int iteration = 0; iteration < 300; ++iteration) {
            final_loss = trainer.train_step(make_loss);
        }

        expect(trainer.step_count() == 300, "Trainer step count mismatch");
        expect(std::isfinite(final_loss), "Trainer loss must be finite");
        expect(final_loss < initial_loss, "Trainer did not reduce loss");
        expect(final_loss < 0.1, "Trainer final loss is too high");

        std::cout << "Trainer test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Trainer test failed: " << error.what() << '\n';
        return 1;
    }
}
