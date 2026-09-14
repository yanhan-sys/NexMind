#pragma once

#include "text_dataset.h"

#include <cstddef>
#include <vector>

namespace nexmind {

// TextDataLoader：从 TextDataset 中按固定 batch 大小取出连续训练样本。
class TextDataLoader {
public:
    TextDataLoader(TextDataset& dataset, std::size_t batch_size);

    std::size_t batch_size() const noexcept { return batch_size_; }
    bool next(std::vector<std::vector<std::size_t>>& inputs,
              std::vector<std::vector<std::size_t>>& targets);
    void reset() noexcept { dataset_.reset(); }

private:
    TextDataset& dataset_;
    std::size_t batch_size_;
};

} // namespace nexmind
