#include "tokenizer.h"

#include <stdexcept>

namespace nexmind {

std::vector<std::size_t> ByteTokenizer::encode(const std::string& text) const {
    std::vector<std::size_t> tokens;
    tokens.reserve(text.size());
    // 保留 UTF-8 字节原样，decode 后可以无损恢复原文本。
    for (const unsigned char byte : text) {
        tokens.push_back(static_cast<std::size_t>(byte));
    }
    return tokens;
}

std::string ByteTokenizer::decode(const std::vector<std::size_t>& tokens) const {
    std::string text;
    text.reserve(tokens.size());
    for (const std::size_t token : tokens) {
        if (token >= vocabulary_size) {
            throw std::out_of_range("ByteTokenizer token is outside vocabulary");
        }
        text.push_back(static_cast<char>(static_cast<unsigned char>(token)));
    }
    return text;
}

} // namespace nexmind
