#pragma once

#include "trainer.h"
#include "transformer_lm.h"

#include <cstddef>
#include <vector>

namespace nexmind {

// Language Model Trainer：封装 token 序列的 next-token 训练步骤和 batch 训练。
class LanguageModelTrainer {
public:
    LanguageModelTrainer(TransformerLanguageModel& model,
                         double learning_rate = 1e-3);

    double train_step(const std::vector<std::size_t>& input_tokens,
                      const std::vector<std::size_t>& target_tokens);
    double train_batch(const std::vector<std::vector<std::size_t>>& input_batch,
                       const std::vector<std::vector<std::size_t>>& target_batch);
    std::size_t step_count() const noexcept { return trainer_.step_count(); }

private:
    TransformerLanguageModel& model_;
    Trainer trainer_;
};

} // namespace nexmind
