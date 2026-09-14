#include "transformer_encoder.h"

#include <stdexcept>
#include <utility>

namespace nexmind {

TransformerEncoder::TransformerEncoder(std::size_t num_layers,
                                       std::size_t embed_dim,
                                       std::size_t num_heads,
                                       std::size_t feed_forward_dim,
                                       std::uint64_t seed) {
    // Encoder 至少需要一层 Block，避免空网络产生无意义的前向结果。
    if (num_layers == 0) {
        throw std::invalid_argument("TransformerEncoder requires at least one layer");
    }

    blocks_.reserve(num_layers);
    for (std::size_t index = 0; index < num_layers; ++index) {
        // 每层使用不同随机种子，避免所有层初始化完全相同。
        blocks_.emplace_back(embed_dim, num_heads, feed_forward_dim, seed + index);
    }
}

Value TransformerEncoder::forward(const Value& input) const {
    // 按顺序执行所有 Transformer Block。
    Value output = input;
    for (const auto& block : blocks_) {
        output = block.forward(output);
    }
    return output;
}

std::vector<Parameter*> TransformerEncoder::parameters() {
    std::vector<Parameter*> result;
    std::size_t total = 0;
    for (auto& block : blocks_) {
        total += block.parameters().size();
    }
    result.reserve(total);
    for (auto& block : blocks_) {
        const auto block_parameters = block.parameters();
        result.insert(result.end(), block_parameters.begin(), block_parameters.end());
    }
    return result;
}

const TransformerBlock& TransformerEncoder::block(std::size_t index) const {
    return blocks_.at(index);
}

} // namespace nexmind
