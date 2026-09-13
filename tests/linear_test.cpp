#include "linear.h"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
    using nexmind::Linear;
    using nexmind::Tensor;
    using nexmind::Value;
    using nexmind::mean;
    using nexmind::multiply;

    Linear linear(2, 2, 123);
    const auto parameters = linear.parameters();
    assert(parameters.size() == 2);
    assert(parameters[0] == &linear.weight());
    assert(parameters[1] == &linear.bias());
    assert(linear.weight().value().requires_grad());
    assert(linear.bias().value().requires_grad());
    assert(linear.weight().value().data().shape() == std::vector<std::size_t>({2, 2}));
    assert(linear.bias().value().data().shape() == std::vector<std::size_t>({1, 2}));

    linear.weight().value().data().at({0, 0}) = 1.0;
    linear.weight().value().data().at({0, 1}) = 2.0;
    linear.weight().value().data().at({1, 0}) = 3.0;
    linear.weight().value().data().at({1, 1}) = 4.0;
    linear.bias().value().data().at({0, 0}) = 0.5;
    linear.bias().value().data().at({0, 1}) = -1.0;

    Value x(Tensor({2, 2}), true, "x");
    x.data().at({0, 0}) = 1.0;
    x.data().at({0, 1}) = 2.0;
    x.data().at({1, 0}) = 3.0;
    x.data().at({1, 1}) = 4.0;

    Value y = linear.forward(x);
    assert(std::abs(y.data().at({0, 0}) - 7.5) < 1e-12);
    assert(std::abs(y.data().at({0, 1}) - 9.0) < 1e-12);
    assert(std::abs(y.data().at({1, 0}) - 15.5) < 1e-12);
    assert(std::abs(y.data().at({1, 1}) - 21.0) < 1e-12);

    Value loss = mean(multiply(y, y));
    loss.backward();

    assert(std::abs(linear.weight().value().grad().at({0, 0}) - 27.0) < 1e-12);
    assert(std::abs(linear.weight().value().grad().at({0, 1}) - 36.0) < 1e-12);
    assert(std::abs(linear.weight().value().grad().at({1, 0}) - 38.5) < 1e-12);
    assert(std::abs(linear.weight().value().grad().at({1, 1}) - 51.0) < 1e-12);
    assert(std::abs(linear.bias().value().grad().at({0, 0}) - 11.5) < 1e-12);
    assert(std::abs(linear.bias().value().grad().at({0, 1}) - 15.0) < 1e-12);
    assert(std::abs(x.grad().at({0, 0}) - 12.75) < 1e-12);
    assert(std::abs(x.grad().at({0, 1}) - 25.5) < 1e-12);
    assert(std::abs(x.grad().at({1, 0}) - 28.75) < 1e-12);
    assert(std::abs(x.grad().at({1, 1}) - 57.5) < 1e-12);

    linear.weight().zero_grad();
    linear.bias().zero_grad();
    assert(std::abs(linear.weight().value().grad()[0]) < 1e-12);
    assert(std::abs(linear.bias().value().grad()[0]) < 1e-12);

    return 0;
}
