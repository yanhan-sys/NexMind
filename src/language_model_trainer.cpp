#include "language_model_trainer.h"

#include <stdexcept>
#include <utility>

namespace nexmind {

namespace {

Value make_token_value(const std::vector<std::size_t>& tokens) {
    // 将 token ID 转成 Tensor，Embedding 层负责实际的索引检查。
    Tensor input_data({tokens.size()}, 0.0);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        input_data[i] = static_cast<double>(tokens[i]);
    }
    return Value(std::move(input_data), false);
}

} // namespace

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

    const Value input_tokens_value = make_token_value(input_tokens);
    return trainer_.train_step([this, input_tokens_value, target_tokens]() mutable {
        // 每个位置预测对应的下一个目标 token。
        const Value logits = model_.forward(input_tokens_value);
        return cross_entropy(logits, target_tokens);
    });
}

double LanguageModelTrainer::train_batch(
    const std::vector<std::vector<std::size_t>>& input_batch,
    const std::vector<std::vector<std::size_t>>& target_batch) {
    if (input_batch.empty()) {
        throw std::invalid_argument("Language model batch must not be empty");
    }
    if (input_batch.size() != target_batch.size()) {
        throw std::invalid_argument("Input and target batch sizes must match");
    }
    for (std::size_t i = 0; i < input_batch.size(); ++i) {
        if (input_batch[i].empty()) {
            throw std::invalid_argument("Language model batch sample must not be empty");
        }
        if (input_batch[i].size() != target_batch[i].size()) {
            throw std::invalid_argument("Input and target sequence lengths must match");
        }
    }

    return trainer_.train_step([this, input_batch, target_batch]() {
        // 将一个 batch 的样本 loss 求平均，再只执行一次反向传播和参数更新。
        Value batch_loss;
        for (std::size_t i = 0; i < input_batch.size(); ++i) {
            const Value input_tokens = make_token_value(input_batch[i]);
            const Value logits = model_.forward(input_tokens);
            const Value sample_loss = cross_entropy(logits, target_batch[i]);
            batch_loss = batch_loss.requires_grad() ? add(batch_loss, sample_loss) : sample_loss;
        }
        return mean(batch_loss);
    });
}

} // namespace nexmind
