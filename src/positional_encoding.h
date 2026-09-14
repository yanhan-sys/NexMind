#pragma once

#include "autograd.h"

#include <cstddef>

namespace nexmind {

// 正弦位置编码：不增加可训练参数，为序列中的每个位置提供位置信息。
class SinusoidalPositionalEncoding {
public:
    explicit SinusoidalPositionalEncoding(std::size_t embed_dim);

    Value forward(const Value& input) const;

private:
    std::size_t embed_dim_;
};

} // namespace nexmind
