#include "optimizer.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace nexmind {

Adam::Adam(std::vector<Parameter*> parameters,
           double learning_rate,
           double beta1,
           double beta2,
           double epsilon)
    : parameters_(std::move(parameters)),
      learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Adam learning rate must be positive");
    }
    if (beta1 < 0.0 || beta1 >= 1.0 || beta2 < 0.0 || beta2 >= 1.0) {
        throw std::invalid_argument("Adam beta values must be in [0, 1)");
    }
    if (epsilon <= 0.0) {
        throw std::invalid_argument("Adam epsilon must be positive");
    }
    for (Parameter* parameter : parameters_) {
        if (parameter == nullptr) {
            throw std::invalid_argument("Adam parameter must not be null");
        }
        states_.push_back({Tensor(parameter->value().data().shape(), 0.0),
                           Tensor(parameter->value().data().shape(), 0.0)});
    }
}

void Adam::step() {
    ++step_count_;
    const double bias_correction1 = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
    const double bias_correction2 = 1.0 - std::pow(beta2_, static_cast<double>(step_count_));

    for (std::size_t parameter_index = 0; parameter_index < parameters_.size(); ++parameter_index) {
        Parameter& parameter = *parameters_[parameter_index];
        Tensor& data = parameter.value().data();
        const Tensor& grad = parameter.value().grad();
        State& state = states_[parameter_index];
        if (data.shape() != grad.shape()) {
            throw std::runtime_error("Adam parameter and gradient shapes differ");
        }
        for (std::size_t i = 0; i < data.size(); ++i) {
            state.first_moment[i] = beta1_ * state.first_moment[i] + (1.0 - beta1_) * grad[i];
            state.second_moment[i] = beta2_ * state.second_moment[i] + (1.0 - beta2_) * grad[i] * grad[i];
            const double first_hat = state.first_moment[i] / bias_correction1;
            const double second_hat = state.second_moment[i] / bias_correction2;
            data[i] -= learning_rate_ * first_hat / (std::sqrt(second_hat) + epsilon_);
        }
    }
}

void Adam::zero_grad() {
    for (Parameter* parameter : parameters_) {
        parameter->zero_grad();
    }
}

} // namespace nexmind
