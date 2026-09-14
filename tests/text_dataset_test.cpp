#include "text_dataset.h"

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
        // 序列 0 1 2 3 4 生成长度为 3 的连续 next-token 样本。
        TextDataset dataset({0, 1, 2, 3, 4}, 3);
        expect(dataset.size() == 2, "Dataset size mismatch");

        std::vector<std::size_t> inputs;
        std::vector<std::size_t> targets;
        expect(dataset.next(inputs, targets), "Dataset first sample missing");
        expect(inputs == std::vector<std::size_t>({0, 1, 2}), "Dataset first inputs mismatch");
        expect(targets == std::vector<std::size_t>({1, 2, 3}), "Dataset first targets mismatch");
        expect(dataset.next(inputs, targets), "Dataset second sample missing");
        expect(inputs == std::vector<std::size_t>({1, 2, 3}), "Dataset second inputs mismatch");
        expect(targets == std::vector<std::size_t>({2, 3, 4}), "Dataset second targets mismatch");
        expect(!dataset.next(inputs, targets), "Dataset should be exhausted");
        dataset.reset();
        expect(dataset.next(inputs, targets), "Dataset reset failed");

        std::cout << "Text Dataset test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Text Dataset test failed: " << error.what() << '\n';
        return 1;
    }
}
