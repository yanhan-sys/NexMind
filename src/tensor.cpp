#include "tensor.h"

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
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

Tensor multiply(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw std::invalid_argument("Tensor multiply shape mismatch");
    }
    Tensor result(lhs.shape());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] * rhs[i];
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
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t k = 0; k < inner; ++k) {
            const double value = lhs.at({i, k});
            for (std::size_t j = 0; j < columns; ++j) {
                result.at({i, j}) += value * rhs.at({k, j});
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
    for (std::size_t i = 0; i < input.shape()[0]; ++i) {
        for (std::size_t j = 0; j < input.shape()[1]; ++j) {
            result.at({j, i}) = input.at({i, j});
        }
    }
    return result;
}

} // namespace nexmind
