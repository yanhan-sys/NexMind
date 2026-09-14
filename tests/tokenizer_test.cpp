#include "tokenizer.h"

#include <iostream>
#include <stdexcept>

using namespace nexmind;

namespace {
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    try {
        // UTF-8 文本按原始字节编码，编码后再解码必须完全一致。
        const std::string text = "NexMind 中文\n";
        ByteTokenizer tokenizer;
        const auto tokens = tokenizer.encode(text);
        expect(!tokens.empty(), "Tokenizer returned empty tokens");
        expect(tokens.size() == text.size(), "Byte token count mismatch");
        expect(tokenizer.decode(tokens) == text, "Tokenizer round trip mismatch");
        expect(ByteTokenizer::vocabulary_size == 256, "Byte vocabulary size mismatch");
        std::cout << "Tokenizer test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Tokenizer test failed: " << error.what() << '\n';
        return 1;
    }
}
