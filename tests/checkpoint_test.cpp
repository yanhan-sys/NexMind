#include "checkpoint.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace nexmind;

namespace {
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    try {
        // 保存前后比较全部参数，验证 checkpoint 能完整恢复模型权重和结构。
        TransformerLanguageModel model(16, 8, 1, 2, 16, 42, true);
        Tensor input_data({3}, 0.0);
        input_data[0] = 1.0;
        input_data[1] = 2.0;
        input_data[2] = 3.0;
        const Value before = model.forward(Value(std::move(input_data), false));

        const std::filesystem::path path = std::filesystem::temp_directory_path() / "nexmind_checkpoint_test.nxm";
        save_checkpoint(model, path.string());
        TransformerLanguageModel restored = load_checkpoint(path.string());

        expect(restored.vocab_size() == model.vocab_size(), "Checkpoint vocab size mismatch");
        expect(restored.embed_dim() == model.embed_dim(), "Checkpoint embed dim mismatch");
        expect(restored.num_layers() == model.num_layers(), "Checkpoint layer count mismatch");
        expect(restored.num_heads() == model.num_heads(), "Checkpoint head count mismatch");
        expect(restored.feed_forward_dim() == model.feed_forward_dim(), "Checkpoint FFN size mismatch");
        expect(restored.causal(), "Checkpoint causal flag mismatch");

        Tensor restored_input({3}, 0.0);
        restored_input[0] = 1.0;
        restored_input[1] = 2.0;
        restored_input[2] = 3.0;
        const Value after = restored.forward(Value(std::move(restored_input), false));
        expect(before.data().shape() == after.data().shape(), "Checkpoint output shape mismatch");
        for (std::size_t index = 0; index < before.data().size(); ++index) {
            expect(std::abs(before.data()[index] - after.data()[index]) < 1e-12, "Checkpoint weight mismatch");
        }

        std::filesystem::remove(path);
        std::cout << "Checkpoint test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Checkpoint test failed: " << error.what() << '\n';
        return 1;
    }
}
