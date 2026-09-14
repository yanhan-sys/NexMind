#include "transformer_lm.h"

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

void validate_tokens(const std::vector<std::size_t>& tokens, std::size_t vocab_size) {
    for (const std::size_t token : tokens) {
        expect(token < vocab_size, "Generated token is outside vocabulary");
    }
}

} // namespace

int main() {
    try {
        // 使用极小模型验证 Greedy 和 Temperature 两条生成路径。
        TransformerLanguageModel model(8, 8, 1, 2, 16, 42);
        const std::vector<std::size_t> prompt{1, 3, 2};

        const auto greedy = generate_greedy(model, prompt, 5);
        expect(greedy.size() == prompt.size() + 5, "Greedy generation length mismatch");
        validate_tokens(greedy, model.vocab_size());
        expect(std::equal(prompt.begin(), prompt.end(), greedy.begin()), "Greedy generation changed prompt");

        const auto sampled = generate_temperature(model, prompt, 7, 1.0, 42);
        expect(sampled.size() == prompt.size() + 7, "Temperature generation length mismatch");
        validate_tokens(sampled, model.vocab_size());
        expect(std::equal(prompt.begin(), prompt.end(), sampled.begin()), "Temperature generation changed prompt");

        bool rejected_temperature = false;
        try {
            (void)generate_temperature(model, prompt, 1, 0.0, 42);
        } catch (const std::invalid_argument&) {
            rejected_temperature = true;
        }
        expect(rejected_temperature, "Invalid temperature was not rejected");

        bool rejected_token = false;
        try {
            (void)generate_greedy(model, {model.vocab_size()}, 1);
        } catch (const std::out_of_range&) {
            rejected_token = true;
        }
        expect(rejected_token, "Out-of-range prompt token was not rejected");

        std::cout << "Generation test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Generation test failed: " << error.what() << '\n';
        return 1;
    }
}
