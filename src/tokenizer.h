#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace nexmind {

// ByteTokenizer：直接使用 UTF-8 原始字节作为 token，零词表训练成本，适合最小可运行语言模型。
class ByteTokenizer {
public:
    static constexpr std::size_t vocabulary_size = 256;

    std::vector<std::size_t> encode(const std::string& text) const;
    std::string decode(const std::vector<std::size_t>& tokens) const;
};

} // namespace nexmind
