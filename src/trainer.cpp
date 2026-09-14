#include "trainer.h"

#include <stdexcept>
#include <utility>

namespace nexmind {

Trainer::Trainer(std::vector<Parameter*> parameters, double learning_rate)
    : optimizer_(std::move(parameters), learning_rate) {
}

double Trainer::train_step(const std::function<Value()>& loss_function) {
    if (!loss_function) {
        throw std::invalid_argument("Trainer requires a loss function");
    }

    // 每一步训练先清理上一轮梯度，避免梯度跨 batch 累积。
    optimizer_.zero_grad();
    Value loss = loss_function();
    if (!loss.requires_grad()) {
        throw std::invalid_argument("Training loss must require gradients");
    }

    loss.backward();
    const double loss_value = loss.data()[0];
    optimizer_.step();
    return loss_value;
}

void Trainer::zero_grad() {
    optimizer_.zero_grad();
}

} // namespace nexmind
