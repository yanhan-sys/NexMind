#include "linear.h"
#include "optimizer.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    nexmind::Linear model(1, 1, 123);
    model.weight().value().data()[0] = 0.0;
    model.bias().value().data()[0] = 0.0;

    nexmind::Adam optimizer(model.parameters(), 0.05);
    nexmind::Value input(nexmind::Tensor({1, 1}, 1.0));
    nexmind::Value target(nexmind::Tensor({1, 1}, 3.0));

    double initial_loss = 0.0;
    double final_loss = 0.0;
    for (int iteration = 0; iteration < 100; ++iteration) {
        optimizer.zero_grad();
        nexmind::Value prediction = model.forward(input);
        nexmind::Value error = nexmind::add(prediction, nexmind::Value(nexmind::Tensor({1, 1}, -3.0)));
        nexmind::Value loss = nexmind::mean(nexmind::multiply(error, error));
        if (iteration == 0) {
            initial_loss = loss.data()[0];
        }
        loss.backward();
        optimizer.step();
        final_loss = loss.data()[0];
    }

    assert(optimizer.step_count() == 100);
    assert(initial_loss > final_loss);
    assert(final_loss < 1e-3);
    assert(std::abs(model.weight().value().data()[0]) > 0.1);
    assert(std::abs(model.bias().value().data()[0]) > 0.1);

    std::cout << "initial_loss=" << initial_loss << " final_loss=" << final_loss << '\n';
    return 0;
}
