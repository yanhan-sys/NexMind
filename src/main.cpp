#include "checkpoint.h"
#include "language_model_trainer.h"
#include "text_dataset.h"
#include "tokenizer.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace nexmind;

namespace {

std::size_t parse_size(std::string_view value, const char* name) {
    std::size_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result == 0) {
        throw std::invalid_argument(std::string("Invalid ") + name);
    }
    return result;
}

double parse_double(std::string_view value, const char* name) {
    std::string text(value);
    std::size_t consumed = 0;
    const double result = std::stod(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument(std::string("Invalid ") + name);
    }
    return result;
}

void print_usage() {
    std::cout << "NexMind\n"
              << "  train <text> <checkpoint> [steps] [sequence_length] [embed_dim] [layers] [heads] [ff_dim] [lr]\n"
              << "  generate <checkpoint> <prompt> [max_new_tokens] [temperature]\n";
}

int train_command(int argc, char** argv) {
    if (argc < 4) {
        print_usage();
        return 1;
    }
    const std::string text_path = argv[2];
    const std::string checkpoint_path = argv[3];
    const std::size_t steps = argc > 4 ? parse_size(argv[4], "steps") : 1000;
    const std::size_t sequence_length = argc > 5 ? parse_size(argv[5], "sequence_length") : 64;
    const std::size_t embed_dim = argc > 6 ? parse_size(argv[6], "embed_dim") : 64;
    const std::size_t layers = argc > 7 ? parse_size(argv[7], "layers") : 2;
    const std::size_t heads = argc > 8 ? parse_size(argv[8], "heads") : 4;
    const std::size_t feed_forward_dim = argc > 9 ? parse_size(argv[9], "ff_dim") : 128;
    const double learning_rate = argc > 10 ? parse_double(argv[10], "learning rate") : 1e-3;

    // 当前开源第一版采用 ByteTokenizer，因此词表固定为 256 个字节。
    TextDataset dataset = load_text_dataset(text_path, sequence_length);
    TransformerLanguageModel model(ByteTokenizer::vocabulary_size, embed_dim, layers, heads, feed_forward_dim, 42, true);
    LanguageModelTrainer trainer(model, learning_rate);

    std::vector<std::size_t> inputs;
    std::vector<std::size_t> targets;
    double loss = 0.0;
    for (std::size_t step = 0; step < steps; ++step) {
        if (!dataset.next(inputs, targets)) {
            dataset.reset();
            if (!dataset.next(inputs, targets)) {
                throw std::runtime_error("Training dataset contains no samples");
            }
        }
        loss = trainer.train_step(inputs, targets);
        if ((step + 1) % 100 == 0 || step + 1 == steps) {
            std::cout << "step=" << (step + 1) << " loss=" << loss << '\n';
        }
    }

    save_checkpoint(model, checkpoint_path);
    std::cout << "saved=" << checkpoint_path << '\n';
    return 0;
}

int generate_command(int argc, char** argv) {
    if (argc < 4) {
        print_usage();
        return 1;
    }
    const TransformerLanguageModel model = load_checkpoint(argv[2]);
    const ByteTokenizer tokenizer;
    const std::vector<std::size_t> prompt = tokenizer.encode(argv[3]);
    const std::size_t max_new_tokens = argc > 4 ? parse_size(argv[4], "max_new_tokens") : 64;
    const double temperature = argc > 5 ? parse_double(argv[5], "temperature") : 0.0;

    // temperature <= 0 使用确定性的 greedy generation，否则使用随机采样。
    const auto output = temperature > 0.0
        ? generate_temperature(model, prompt, max_new_tokens, temperature, 42)
        : generate_greedy(model, prompt, max_new_tokens);
    std::cout << tokenizer.decode(output) << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage();
            return 0;
        }
        const std::string command = argv[1];
        if (command == "train") {
            return train_command(argc, argv);
        }
        if (command == "generate") {
            return generate_command(argc, argv);
        }
        print_usage();
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "NexMind error: " << error.what() << '\n';
        return 1;
    }
}
