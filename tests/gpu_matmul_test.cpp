#include "gpu_matmul.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    if (!nexmind::gpu_matmul_available()) {
        std::cout << "GPU matmul unavailable; test skipped\n";
        return 0;
    }

    // 使用非 16 对齐尺寸，覆盖 tile 边界和补零路径。
    constexpr std::size_t rows = 65;
    constexpr std::size_t inner = 67;
    constexpr std::size_t columns = 63;
    std::vector<double> lhs(rows * inner);
    std::vector<double> rhs(inner * columns);
    std::vector<double> result(rows * columns, 0.0);

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] = static_cast<double>((i % 7) + 1) * 0.25;
    }
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        rhs[i] = static_cast<double>((i % 5) + 1) * 0.5;
    }

    assert(nexmind::gpu_matmul(lhs.data(), rhs.data(), result.data(), rows, inner, columns));

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            double expected = 0.0;
            for (std::size_t k = 0; k < inner; ++k) {
                expected += lhs[row * inner + k] * rhs[k * columns + column];
            }
            assert(std::abs(result[row * columns + column] - expected) < 1e-10);
        }
    }

    std::cout << "GPU matmul test passed\n";
    return 0;
}