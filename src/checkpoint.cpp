#include "checkpoint.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace nexmind {

namespace {

constexpr char kMagic[] = "NXMIND01";
constexpr std::uint32_t kVersion = 1;

void write_exact(std::ofstream& output, const void* data, std::size_t size) {
    output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!output) {
        throw std::runtime_error("Failed to write checkpoint");
    }
}

void read_exact(std::ifstream& input, void* data, std::size_t size) {
    input.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!input) {
        throw std::runtime_error("Failed to read checkpoint");
    }
}

template <typename T>
void write_value(std::ofstream& output, const T& value) {
    write_exact(output, &value, sizeof(T));
}

template <typename T>
T read_value(std::ifstream& input) {
    T value{};
    read_exact(input, &value, sizeof(T));
    return value;
}

} // namespace

void save_checkpoint(const TransformerLanguageModel& model, const std::string& path) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Failed to open checkpoint for writing: " + path);
    }

    // Header 保存模型结构，后续参数按照 parameters() 的稳定顺序写入。
    write_exact(output, kMagic, sizeof(kMagic) - 1);
    write_value(output, kVersion);
    write_value(output, static_cast<std::uint64_t>(model.vocab_size()));
    write_value(output, static_cast<std::uint64_t>(model.embed_dim()));
    write_value(output, static_cast<std::uint64_t>(model.num_layers()));
    write_value(output, static_cast<std::uint64_t>(model.num_heads()));
    write_value(output, static_cast<std::uint64_t>(model.feed_forward_dim()));
    write_value(output, static_cast<std::uint8_t>(model.causal() ? 1 : 0));

    const auto parameters = const_cast<TransformerLanguageModel&>(model).parameters();
    write_value(output, static_cast<std::uint64_t>(parameters.size()));
    for (const Parameter* parameter : parameters) {
        const Tensor& tensor = parameter->value().data();
        write_value(output, static_cast<std::uint64_t>(tensor.ndim()));
        for (const std::size_t dimension : tensor.shape()) {
            write_value(output, static_cast<std::uint64_t>(dimension));
        }
        write_value(output, static_cast<std::uint64_t>(tensor.size()));
        if (tensor.size() != 0) {
            write_exact(output, tensor.data(), tensor.size() * sizeof(double));
        }
    }
}

TransformerLanguageModel load_checkpoint(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open checkpoint: " + path);
    }

    char magic[sizeof(kMagic) - 1]{};
    read_exact(input, magic, sizeof(magic));
    if (std::string(magic, sizeof(magic)) != std::string(kMagic, sizeof(kMagic) - 1)) {
        throw std::runtime_error("Invalid NexMind checkpoint magic");
    }
    const std::uint32_t version = read_value<std::uint32_t>(input);
    if (version != kVersion) {
        throw std::runtime_error("Unsupported NexMind checkpoint version");
    }

    const auto vocab_size = static_cast<std::size_t>(read_value<std::uint64_t>(input));
    const auto embed_dim = static_cast<std::size_t>(read_value<std::uint64_t>(input));
    const auto num_layers = static_cast<std::size_t>(read_value<std::uint64_t>(input));
    const auto num_heads = static_cast<std::size_t>(read_value<std::uint64_t>(input));
    const auto feed_forward_dim = static_cast<std::size_t>(read_value<std::uint64_t>(input));
    const bool causal = read_value<std::uint8_t>(input) != 0;
    const std::size_t parameter_count = static_cast<std::size_t>(read_value<std::uint64_t>(input));

    // 固定 seed 只用于创建参数容器，随后立即覆盖为 checkpoint 中保存的权重。
    TransformerLanguageModel model(vocab_size, embed_dim, num_layers, num_heads, feed_forward_dim, 42, causal);
    auto parameters = model.parameters();
    if (parameters.size() != parameter_count) {
        throw std::runtime_error("Checkpoint parameter count mismatch");
    }

    for (std::size_t index = 0; index < parameters.size(); ++index) {
        const std::size_t ndim = static_cast<std::size_t>(read_value<std::uint64_t>(input));
        std::vector<std::size_t> shape(ndim);
        for (std::size_t dimension = 0; dimension < ndim; ++dimension) {
            shape[dimension] = static_cast<std::size_t>(read_value<std::uint64_t>(input));
        }
        const std::size_t size = static_cast<std::size_t>(read_value<std::uint64_t>(input));
        Tensor& target = parameters[index]->value().data();
        if (target.shape() != shape || target.size() != size) {
            throw std::runtime_error("Checkpoint parameter shape mismatch");
        }
        if (size != 0) {
            read_exact(input, target.data(), size * sizeof(double));
        }
    }
    return model;
}

} // namespace nexmind
