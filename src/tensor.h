#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace nexmind {

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape);
    Tensor(std::vector<std::size_t> shape, double value);

    const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    std::size_t ndim() const noexcept { return shape_.size(); }
    std::size_t size() const noexcept { return data_.size(); }

    double* data() noexcept { return data_.data(); }
    const double* data() const noexcept { return data_.data(); }

    double& operator[](std::size_t index) { return data_.at(index); }
    const double& operator[](std::size_t index) const { return data_.at(index); }

    double& at(std::initializer_list<std::size_t> indices);
    const double& at(std::initializer_list<std::size_t> indices) const;

    void fill(double value);

private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<double> data_;

    static std::size_t checked_size(const std::vector<std::size_t>& shape);
    std::size_t offset(std::initializer_list<std::size_t> indices) const;
};

Tensor add(const Tensor& lhs, const Tensor& rhs);
Tensor add_row_bias(const Tensor& input, const Tensor& bias);
Tensor multiply(const Tensor& lhs, const Tensor& rhs);
Tensor matmul(const Tensor& lhs, const Tensor& rhs);
Tensor transpose_2d(const Tensor& input);

} // namespace nexmind
