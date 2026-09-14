#include "language_model_trainer.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

using namespace nexmind;

int main() {
    TransformerLanguageModel model(4, 4, 1, 2, 8, 123, true);
    LanguageModelTrainer trainer(model, 0.01);

    const std::vector<std::vector<std::size_t>> inputs{
        {0, 1, 2},
        {1, 2, 3},
    };
    const std::vector<std::vector<std::size_t>> targets{
        {1, 2, 3},
        {2, 3, 0},
    };

    const double first_loss = trainer.train_batch(inputs, targets);
    assert(std::isfinite(first_loss));
    assert(trainer.step_count() == 1);

    double last_loss = first_loss;
    for (std::size_t step = 0; step < 30; ++step) {
        last_loss = trainer.train_batch(inputs, targets);
    }

    assert(std::isfinite(last_loss));
    assert(last_loss < first_loss);
    assert(trainer.step_count() == 31);
    return 0;
}
