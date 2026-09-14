#pragma once

#include "optimizer.h"

#include <cstddef>
#include <functional>
#include <vector>

namespace nexmind {

// 通用训练循环：负责清梯度、构建 Loss、反向传播和参数更新。
class Trainer {
public:
    Trainer(std::vector<Parameter*> parameters,
            double learning_rate = 1e-3);

    double train_step(const std::function<Value()>& loss_function);
    void zero_grad();
    std::size_t step_count() const noexcept { return optimizer_.step_count(); }

private:
    Adam optimizer_;
};

} // namespace nexmind
