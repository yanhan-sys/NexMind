#include "tensor.h"
#include "gpu_matmul.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace nexmind {

std::size_t Tensor::checked_size(const std::vector<std::size_t>& shape) {
    std::size_t total = 1;
    for (const std::size_t dimension : shape) {
        if (dimension == 0) {
            return 0;
        }
        if (total > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("Tensor size overflow");
        }
        total *= dimension;
    }
    return total;
}

Tensor::Tensor(std::vector<std::size_t> shape) : shape_(std::move(shape)), data_(checked_size(shape_)) {
    strides_.resize(shape_.size());
    std::size_t stride = 1;
    for (std::size_t i = shape_.size(); i-- > 0;) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

Tensor::Tensor(std::vector<std::size_t> shape, double value) : Tensor(std::move(shape)) {
    fill(value);
}

double& Tensor::at(std::initializer_list<std::size_t> indices) {
    return data_.at(offset(indices));
}

const double& Tensor::at(std::initializer_list<std::size_t> indices) const {
    return data_.at(offset(indices));
}

std::size_t Tensor::offset(std::initializer_list<std::size_t> indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Tensor index rank mismatch");
    }
    std::size_t result = 0;
    std::size_t dimension = 0;
    for (const std::size_t index : indices) {
        if (index >= shape_[dimension]) {
            throw std::out_of_range("Tensor index out of range");
        }
        result += index * strides_[dimension];
        ++dimension;
    }
    return result;
}

void Tensor::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

Tensor add(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw std::invalid_argument("Tensor add shape mismatch");
    }
    Tensor result(lhs.shape());
    #pragma omp parallel for if(lhs.size() >= 4096)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(lhs.size()); ++i) {
        result[static_cast<std::size_t>(i)] = lhs[static_cast<std::size_t>(i)] + rhs[static_cast<std::size_t>(i)];
    }
    return result;
}

Tensor add_row_bias(const Tensor& input, const Tensor& bias) {
    if (input.ndim() != 2 || bias.ndim() != 2 || bias.shape()[0] != 1 || input.shape()[1] != bias.shape()[1]) {
        throw std::invalid_argument("Tensor row bias shape mismatch");
    }
    Tensor result(input.shape());
    #pragma omp parallel for if(input.size() >= 4096)
    for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(input.shape()[0]); ++row) {
        for (std::size_t column = 0; column < input.shape()[1]; ++column) {
            result.at({static_cast<std::size_t>(row), column}) = input.at({static_cast<std::size_t>(row), column}) + bias.at({0, column});
        }
    }
    return result;
}

Tensor multiply(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw std::invalid_argument("Tensor multiply shape mismatch");
    }
    Tensor result(lhs.shape());
    #pragma omp parallel for if(lhs.size() >= 4096)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(lhs.size()); ++i) {
        result[static_cast<std::size_t>(i)] = lhs[static_cast<std::size_t>(i)] * rhs[static_cast<std::size_t>(i)];
    }
    return result;
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.ndim() != 2 || rhs.ndim() != 2 || lhs.shape()[1] != rhs.shape()[0]) {
        throw std::invalid_argument("Tensor matmul shape mismatch");
    }
    const std::size_t rows = lhs.shape()[0];
    const std::size_t inner = lhs.shape()[1];
    const std::size_t columns = rhs.shape()[1];
    Tensor result({rows, columns}, 0.0);

    // 大矩阵优先尝试 GPU；GPU 不可用或执行失败时自动回退 CPU。
    if (rows * inner * columns >= 262144 &&
        gpu_matmul(lhs.data(), rhs.data(), result.data(), rows, inner, columns)) {
        return result;
    }

    #pragma omp parallel for if(rows * inner * columns >= 16384)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(rows); ++i) {
        for (std::size_t k = 0; k < inner; ++k) {
            const double value = lhs.at({static_cast<std::size_t>(i), k});
            for (std::size_t j = 0; j < columns; ++j) {
                result.at({static_cast<std::size_t>(i), j}) += value * rhs.at({k, j});
            }
        }
    }
    return result;
}

Tensor transpose_2d(const Tensor& input) {
    if (input.ndim() != 2) {
        throw std::invalid_argument("Tensor transpose requires a 2D tensor");
    }
    Tensor result({input.shape()[1], input.shape()[0]});
    #pragma omp parallel for if(input.size() >= 4096)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(input.shape()[0]); ++i) {
        for (std::size_t j = 0; j < input.shape()[1]; ++j) {
            result.at({j, static_cast<std::size_t>(i)}) = input.at({static_cast<std::size_t>(i), j});
        }
    }
    return result;
}

} // namespace nexmind
