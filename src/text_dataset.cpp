#include "text_dataset.h"

#include "tokenizer.h"

#include <fstream>
#include <stdexcept>
#include <utility>

namespace nexmind {

TextDataset::TextDataset(std::vector<std::size_t> tokens, std::size_t sequence_length)
    : tokens_(std::move(tokens)), sequence_length_(sequence_length) {
    if (sequence_length_ == 0) {
        throw std::invalid_argument("TextDataset sequence length must be positive");
    }
    if (tokens_.size() < sequence_length_ + 1) {
        throw std::invalid_argument("TextDataset requires at least sequence_length + 1 tokens");
    }
}

std::size_t TextDataset::size() const noexcept {
    return tokens_.size() - sequence_length_;
}

bool TextDataset::next(std::vector<std::size_t>& inputs, std::vector<std::size_t>& targets) {
    if (position_ >= size()) {
        return false;
    }
    inputs.assign(tokens_.begin() + static_cast<std::ptrdiff_t>(position_),
                  tokens_.begin() + static_cast<std::ptrdiff_t>(position_ + sequence_length_));
    targets.assign(tokens_.begin() + static_cast<std::ptrdiff_t>(position_ + 1),
                   tokens_.begin() + static_cast<std::ptrdiff_t>(position_ + sequence_length_ + 1));
    ++position_;
    return true;
}

TextDataset load_text_dataset(const std::string& path, std::size_t sequence_length) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open training text: " + path);
    }
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    return TextDataset(ByteTokenizer().encode(text), sequence_length);
}

} // namespace nexmind
