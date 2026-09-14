#include "text_dataloader.h"

#include <stdexcept>

namespace nexmind {

TextDataLoader::TextDataLoader(TextDataset& dataset, std::size_t batch_size)
    : dataset_(dataset), batch_size_(batch_size) {
    if (batch_size_ == 0) {
        throw std::invalid_argument("TextDataLoader batch size must be positive");
    }
}

bool TextDataLoader::next(std::vector<std::vector<std::size_t>>& inputs,
                          std::vector<std::vector<std::size_t>>& targets) {
    inputs.clear();
    targets.clear();
    inputs.reserve(batch_size_);
    targets.reserve(batch_size_);

    std::vector<std::size_t> input;
    std::vector<std::size_t> target;
    while (inputs.size() < batch_size_ && dataset_.next(input, target)) {
        // 拷贝当前样本，避免下一次 dataset.next() 覆盖 batch 中的数据。
        inputs.push_back(input);
        targets.push_back(target);
    }
    return !inputs.empty();
}

} // namespace nexmind
