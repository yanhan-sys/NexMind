#pragma once

#include <cstddef>

namespace nexmind {

// 查询 Windows D3D11 双精度计算后端是否可用。
bool gpu_matmul_available();

// 使用 GPU 执行双精度矩阵乘法；失败时返回 false，由 CPU 路径接管。
bool gpu_matmul(const double* lhs,
                const double* rhs,
                double* result,
                std::size_t rows,
                std::size_t inner,
                std::size_t columns);

} // namespace nexmind
