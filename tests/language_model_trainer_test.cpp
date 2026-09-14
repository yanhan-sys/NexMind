#include "language_model_trainer.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

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
        // 极小语言模型学习重复 token 转移：0 -> 1 -> 2 -> 0。
        TransformerLanguageModel model(3, 6, 1, 2, 12, 42);
        LanguageModelTrainer trainer(model, 0.03);
        const std::vector<std::size_t> inputs{0, 1, 2};
        const std::vector<std::size_t> targets{1, 2, 0};

        Tensor input_data({3}, 0.0);
        input_data[0] = 0.0;
        input_data[1] = 1.0;
        input_data[2] = 2.0;
        Value input(std::move(input_data), false);
        const Value initial_logits = model.forward(input);
        const double initial_loss = cross_entropy(initial_logits, targets).data()[0];

        double final_loss = initial_loss;
        for (int iteration = 0; iteration < 120; ++iteration) {
            final_loss = trainer.train_step(inputs, targets);
        }

        expect(trainer.step_count() == 120, "Language model trainer step count mismatch");
        expect(std::isfinite(final_loss), "Language model training loss must be finite");
        expect(final_loss < initial_loss, "Language model training did not reduce loss");
        expect(final_loss < 0.2, "Language model final loss is too high");

        std::cout << "Language Model Trainer test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Language Model Trainer test failed: " << error.what() << '\n';
        return 1;
    }
}
