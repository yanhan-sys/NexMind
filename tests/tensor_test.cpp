#include "tensor.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

int main() {
    using nexmind::Tensor;
    using nexmind::add;
    using nexmind::matmul;
    using nexmind::multiply;
    using nexmind::transpose_2d;

    Tensor a({2, 3});
    Tensor b({2, 3}, 2.0);
    a.at({0, 0}) = 1.0;
    a.at({0, 1}) = 2.0;
    a.at({0, 2}) = 3.0;
    a.at({1, 0}) = 4.0;
    a.at({1, 1}) = 5.0;
    a.at({1, 2}) = 6.0;

    const Tensor sum = add(a, b);
    assert(sum.shape() == std::vector<std::size_t>({2, 3}));
    assert(sum.at({1, 2}) == 8.0);

    const Tensor product = multiply(a, b);
    assert(product.at({0, 2}) == 6.0);

    const Tensor transposed = transpose_2d(a);
    assert(transposed.shape() == std::vector<std::size_t>({3, 2}));
    assert(transposed.at({2, 1}) == 6.0);

    Tensor weights({3, 2});
    weights.at({0, 0}) = 1.0;
    weights.at({0, 1}) = 2.0;
    weights.at({1, 0}) = 3.0;
    weights.at({1, 1}) = 4.0;
    weights.at({2, 0}) = 5.0;
    weights.at({2, 1}) = 6.0;

    const Tensor output = matmul(a, weights);
    assert(output.shape() == std::vector<std::size_t>({2, 2}));
    assert(std::abs(output.at({0, 0}) - 22.0) < 1e-12);
    assert(std::abs(output.at({0, 1}) - 28.0) < 1e-12);
    assert(std::abs(output.at({1, 0}) - 49.0) < 1e-12);
    assert(std::abs(output.at({1, 1}) - 64.0) < 1e-12);

    bool threw = false;
    try {
        (void)matmul(a, a);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    return 0;
}
