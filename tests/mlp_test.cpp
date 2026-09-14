#include "mlp.h"
#include "optimizer.h"
#include "relu.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using nexmind::Adam;
    using nexmind::MLP;
    using nexmind::Tensor;
    using nexmind::Value;
    using nexmind::add;
    using nexmind::mean;
    using nexmind::multiply;
    using nexmind::relu;

    Value activation_input(Tensor({1, 4}), true);
    activation_input.data().at({0, 0}) = -2.0;
    activation_input.data().at({0, 1}) = 0.0;
    activation_input.data().at({0, 2}) = 3.0;
    activation_input.data().at({0, 3}) = -1.0;
    Value activation = relu(activation_input);
    assert(activation.data().at({0, 0}) == 0.0);
    assert(activation.data().at({0, 1}) == 0.0);
    assert(activation.data().at({0, 2}) == 3.0);
    assert(activation.data().at({0, 3}) == 0.0);
    mean(activation).backward();
    assert(std::abs(activation_input.grad().at({0, 0})) < 1e-12);
    assert(std::abs(activation_input.grad().at({0, 1})) < 1e-12);
    assert(std::abs(activation_input.grad().at({0, 2}) - 0.25) < 1e-12);
    assert(std::abs(activation_input.grad().at({0, 3})) < 1e-12);

    MLP model(1, 4, 1, 123);
    assert(model.parameters().size() == 4);
    model.input_layer().weight().value().data().fill(0.0);
    model.input_layer().bias().value().data().fill(0.0);
    model.input_layer().weight().value().data().at({0, 0}) = 1.0;
    model.output_layer().weight().value().data().fill(0.0);
    model.output_layer().bias().value().data().fill(0.0);

    Value input(Tensor({4, 1}));
    input.data().at({0, 0}) = 1.0;
    input.data().at({1, 0}) = 2.0;
    input.data().at({2, 0}) = 3.0;
    input.data().at({3, 0}) = 4.0;
    Value target(Tensor({4, 1}));
    target.data().at({0, 0}) = 3.0;
    target.data().at({1, 0}) = 5.0;
    target.data().at({2, 0}) = 7.0;
    target.data().at({3, 0}) = 9.0;

    Adam optimizer(model.parameters(), 0.03);
    double initial_loss = 0.0;
    double final_loss = 0.0;
    for (int iteration = 0; iteration < 300; ++iteration) {
        optimizer.zero_grad();
        Value prediction = model.forward(input);
        Value error = add(prediction, Value(Tensor({4, 1}, 0.0)));
        for (std::size_t i = 0; i < error.data().size(); ++i) {
            error.data()[i] -= target.data()[i];
        }
        Value loss = mean(multiply(error, error));
        if (iteration == 0) {
            initial_loss = loss.data()[0];
        }
        loss.backward();
        optimizer.step();
        final_loss = loss.data()[0];
    }

    assert(optimizer.step_count() == 300);
    assert(initial_loss > final_loss);
    assert(final_loss < 1e-3);
    std::cout << "initial_loss=" << initial_loss << " final_loss=" << final_loss << '\n';
    return 0;
}
