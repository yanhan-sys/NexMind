#include "embedding.h"
#include "layer_norm.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    nexmind::Embedding embedding(4, 3, 123);
    nexmind::Value indices(nexmind::Tensor({2}), false);
    indices.data()[0] = 1.0;
    indices.data()[1] = 3.0;
    nexmind::Value embedded = embedding.forward(indices);
    assert(embedded.data().shape() == std::vector<std::size_t>({2, 3}));
    assert(embedded.requires_grad());

    nexmind::Value embedding_loss = nexmind::mean(nexmind::multiply(embedded, embedded));
    embedding_loss.backward();
    assert(embedding.weight().value().grad().size() == 12);
    double row_one_grad = 0.0;
    double row_three_grad = 0.0;
    for (std::size_t column = 0; column < 3; ++column) {
        row_one_grad += std::abs(embedding.weight().value().grad().at({1, column}));
        row_three_grad += std::abs(embedding.weight().value().grad().at({3, column}));
    }
    assert(row_one_grad > 0.0);
    assert(row_three_grad > 0.0);

    nexmind::LayerNorm layer_norm(3);
    nexmind::Value input(nexmind::Tensor({2, 3}), false);
    input.data().at({0, 0}) = 1.0;
    input.data().at({0, 1}) = 2.0;
    input.data().at({0, 2}) = 4.0;
    input.data().at({1, 0}) = -1.0;
    input.data().at({1, 1}) = 0.0;
    input.data().at({1, 2}) = 2.0;
    nexmind::Value normalized = layer_norm.forward(input);
    for (std::size_t row = 0; row < 2; ++row) {
        double mean = 0.0;
        for (std::size_t column = 0; column < 3; ++column) {
            mean += normalized.data().at({row, column});
        }
        assert(std::abs(mean) < 1e-9);
    }

    nexmind::Value norm_loss = nexmind::mean(nexmind::multiply(normalized, normalized));
    norm_loss.backward();
    assert(std::isfinite(norm_loss.data()[0]));
    assert(layer_norm.gamma().value().grad().size() == 3);
    assert(layer_norm.beta().value().grad().size() == 3);
    assert(std::abs(layer_norm.gamma().value().grad()[0]) > 0.0);

    std::cout << "embedding_and_layer_norm_ok\n";
    return 0;
}
