#include "language_model_trainer.h"

#include <stdexcept>
#include <utility>

namespace nexmind {

LanguageModelTrainer::LanguageModelTrainer(TransformerLanguageModel& model,
                                           double learning_rate)
    : model_(model),
      trainer_(model.parameters(), learning_rate) {
}

double LanguageModelTrainer::train_step(const std::vector<std::size_t>& input_tokens,
                                         const std::vector<std::size_t>& target_tokens) {
    if (input_tokens.empty()) {
        throw std::invalid_argument("Language model input tokens must not be empty");
    }
    if (input_tokens.size() != target_tokens.size()) {
        throw std::invalid_argument("Input and target token counts must match");
    }

    // 将 token ID 转成 Tensor，Embedding 层负责实际的索引检查。
    Tensor input_data({input_tokens.size()}, 0.0);
    for (std::size_t i = 0; i < input_tokens.size(); ++i) {
        input_data[i] = static_cast<double>(input_tokens[i]);
    }
    Value input_tokens_value(std::move(input_data), false);

    return trainer_.train_step([this, input_tokens_value, target_tokens]() mutable {
        // 每个位置预测对应的下一个目标 token。
        const Value logits = model_.forward(input_tokens_value);
        return cross_entropy(logits, target_tokens);
    });
}

} // namespace nexmind
