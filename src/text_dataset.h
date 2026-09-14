#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace nexmind {

// TextDataset：将连续 token 文本切成固定长度的 next-token 训练样本。
class TextDataset {
public:
    TextDataset(std::vector<std::size_t> tokens, std::size_t sequence_length);

    std::size_t sequence_length() const noexcept { return sequence_length_; }
    std::size_t size() const noexcept;
    bool next(std::vector<std::size_t>& inputs, std::vector<std::size_t>& targets);
    void reset() noexcept { position_ = 0; }

private:
    std::vector<std::size_t> tokens_;
    std::size_t sequence_length_;
    std::size_t position_ = 0;
};

TextDataset load_text_dataset(const std::string& path, std::size_t sequence_length);

} // namespace nexmind
