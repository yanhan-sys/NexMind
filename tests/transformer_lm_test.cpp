#include "transformer_lm.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace nexmind;

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    try {
        // 小型语言模型：验证 Token → Encoder → Vocabulary logits 的完整计算图。
        TransformerLanguageModel model(16, 8, 2, 2, 16, 42);
        Tensor token_data({3}, 0.0);
        token_data[0] = 1.0;
        token_data[1] = 4.0;
        token_data[2] = 7.0;
        Value token_ids(std::move(token_data), false);

        const Value logits = model.forward(token_ids);
        expect(logits.data().shape() == std::vector<std::size_t>({3, 16}), "Language model logits shape mismatch");
        expect(model.vocab_size() == 16, "Language model vocabulary size mismatch");
        expect(model.parameters().size() == 47, "Language model parameter count mismatch");

        // 每个位置只取一个目标词，验证 CrossEntropy 能直接接收 LM Head 输出。
        const std::vector<std::size_t> targets{4, 7, 1};
        Value loss = cross_entropy(logits, targets);
        loss.backward();

        expect(std::isfinite(loss.data()[0]), "Language model loss must be finite");
        bool parameter_gradient_found = false;
        for (Parameter* parameter : model.parameters()) {
            expect(parameter != nullptr, "Language model returned null parameter");
            expect(parameter->value().requires_grad(), "Language model parameter must require gradients");
            const Tensor& gradient = parameter->value().grad();
            for (std::size_t i = 0; i < gradient.size(); ++i) {
                expect(std::isfinite(gradient[i]), "Language model gradient must be finite");
                if (std::abs(gradient[i]) > 1e-12) {
                    parameter_gradient_found = true;
                }
            }
        }
        expect(parameter_gradient_found, "Language model parameters did not receive gradients");

        std::cout << "Transformer Language Model test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Transformer Language Model test failed: " << error.what() << '\n';
        return 1;
    }
}
