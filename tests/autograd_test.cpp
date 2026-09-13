#include "autograd.h"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
    using nexmind::Tensor;
    using nexmind::Value;
    using nexmind::add;
    using nexmind::cross_entropy;
    using nexmind::matmul;
    using nexmind::mean;
    using nexmind::multiply;

    Value x(Tensor({1, 2}), true, "x");
    x.data().at({0, 0}) = 2.0;
    x.data().at({0, 1}) = 3.0;
    Value w(Tensor({2, 1}), true, "w");
    w.data().at({0, 0}) = 4.0;
    w.data().at({1, 0}) = 5.0;

    Value y = matmul(x, w);
    Value loss = mean(multiply(y, y));
    loss.backward();

    const double y_value = 23.0;
    assert(std::abs(loss.data()[0] - y_value * y_value) < 1e-12);
    assert(std::abs(x.grad().at({0, 0}) - 184.0) < 1e-12);
    assert(std::abs(x.grad().at({0, 1}) - 230.0) < 1e-12);
    assert(std::abs(w.grad().at({0, 0}) - 184.0) < 1e-12);
    assert(std::abs(w.grad().at({1, 0}) - 276.0) < 1e-12);

    Value logits(Tensor({2, 3}), true, "logits");
    logits.data().at({0, 0}) = 2.0;
    logits.data().at({0, 1}) = 1.0;
    logits.data().at({0, 2}) = 0.0;
    logits.data().at({1, 0}) = 0.0;
    logits.data().at({1, 1}) = 1.0;
    logits.data().at({1, 2}) = 2.0;

    Value ce = cross_entropy(logits, {0, 2});
    ce.backward();
    assert(ce.data()[0] > 0.0);
    assert(logits.grad().shape() == std::vector<std::size_t>({2, 3}));
    assert(logits.grad().at({0, 0}) < 0.0);
    assert(logits.grad().at({0, 1}) > 0.0);
    assert(logits.grad().at({1, 2}) < 0.0);

    Value a(Tensor({1, 2}), true);
    a.data().at({0, 0}) = 1.0;
    a.data().at({0, 1}) = 2.0;
    Value b(Tensor({1, 2}), true);
    b.data().at({0, 0}) = 3.0;
    b.data().at({0, 1}) = 4.0;
    Value c = add(a, b);
    mean(c).backward();
    assert(std::abs(a.grad().at({0, 0}) - 0.5) < 1e-12);
    assert(std::abs(b.grad().at({0, 1}) - 0.5) < 1e-12);

    return 0;
}
