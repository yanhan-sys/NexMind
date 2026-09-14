#include "transformer_lm.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace nexmind {

TransformerLanguageModel::TransformerLanguageModel(std::size_t vocab_size,
                                                   std::size_t embed_dim,
                                                   std::size_t num_layers,
                                                   std::size_t num_heads,
                                                   std::size_t feed_forward_dim,
                                                   std::uint64_t seed)
    : vocab_size_(vocab_size),
      embed_dim_(embed_dim),
      embedding_(vocab_size, embed_dim, seed),
      encoder_(num_layers, embed_dim, num_heads, feed_forward_dim, seed + 1),
      lm_head_(embed_dim, vocab_size, seed + 2) {
    // 词表必须非空，隐藏维度由 Embedding 和 Encoder 保持一致。
    if (vocab_size == 0 || embed_dim == 0) {
        throw std::invalid_argument("TransformerLanguageModel dimensions must be positive");
    }
}

Value TransformerLanguageModel::forward(const Value& token_ids) const {
    // Token ID → Embedding → Transformer Encoder → Vocabulary logits。
    const Value embedded = embedding_.forward(token_ids);
    const Value encoded = encoder_.forward(embedded);
    return lm_head_.forward(encoded);
}

std::vector<Parameter*> TransformerLanguageModel::parameters() {
    std::vector<Parameter*> result;
    const auto embedding_parameters = embedding_.parameters();
    const auto encoder_parameters = encoder_.parameters();
    const auto lm_head_parameters = lm_head_.parameters();
    result.reserve(embedding_parameters.size() + encoder_parameters.size() + lm_head_parameters.size());
    result.insert(result.end(), embedding_parameters.begin(), embedding_parameters.end());
    result.insert(result.end(), encoder_parameters.begin(), encoder_parameters.end());
    result.insert(result.end(), lm_head_parameters.begin(), lm_head_parameters.end());
    return result;
}

namespace {

void validate_generation_input(const TransformerLanguageModel& model,
                               const std::vector<std::size_t>& tokens) {
    if (tokens.empty()) {
        throw std::invalid_argument("Generation prompt must not be empty");
    }
    for (const std::size_t token : tokens) {
        if (token >= model.vocab_size()) {
            throw std::out_of_range("Generation token ID is outside vocabulary");
        }
    }
}

std::size_t select_greedy_token(const Tensor& logits, std::size_t row, std::size_t vocab_size) {
    const std::size_t offset = row * vocab_size;
    std::size_t best_token = 0;
    double best_value = -std::numeric_limits<double>::infinity();
    for (std::size_t token = 0; token < vocab_size; ++token) {
        const double value = logits[offset + token];
        if (value > best_value) {
            best_value = value;
            best_token = token;
        }
    }
    return best_token;
}

std::size_t select_temperature_token(const Tensor& logits,
                                     std::size_t row,
                                     std::size_t vocab_size,
                                     double temperature,
                                     std::mt19937_64& generator) {
    const std::size_t offset = row * vocab_size;
    double max_logit = -std::numeric_limits<double>::infinity();
    for (std::size_t token = 0; token < vocab_size; ++token) {
        max_logit = std::max(max_logit, logits[offset + token]);
    }

    std::vector<double> probabilities(vocab_size, 0.0);
    double sum = 0.0;
    for (std::size_t token = 0; token < vocab_size; ++token) {
        const double probability = std::exp((logits[offset + token] - max_logit) / temperature);
        probabilities[token] = probability;
        sum += probability;
    }

    if (!std::isfinite(sum) || sum <= 0.0) {
        return select_greedy_token(logits, row, vocab_size);
    }

    for (double& probability : probabilities) {
        probability /= sum;
    }
    std::discrete_distribution<std::size_t> distribution(probabilities.begin(), probabilities.end());
    return distribution(generator);
}

} // namespace

std::vector<std::size_t> generate_greedy(const TransformerLanguageModel& model,
                                         std::vector<std::size_t> tokens,
                                         std::size_t max_new_tokens) {
    validate_generation_input(model, tokens);
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        Tensor input_data({tokens.size()}, 0.0);
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            input_data[i] = static_cast<double>(tokens[i]);
        }
        const Value logits = model.forward(Value(std::move(input_data), false));
        const std::size_t next_token = select_greedy_token(logits.data(), tokens.size() - 1, model.vocab_size());
        tokens.push_back(next_token);
    }
    return tokens;
}

std::vector<std::size_t> generate_temperature(const TransformerLanguageModel& model,
                                              std::vector<std::size_t> tokens,
                                              std::size_t max_new_tokens,
                                              double temperature,
                                              std::uint64_t seed) {
    validate_generation_input(model, tokens);
    if (!std::isfinite(temperature) || temperature <= 0.0) {
        throw std::invalid_argument("Generation temperature must be positive");
    }

    std::mt19937_64 generator(seed);
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        Tensor input_data({tokens.size()}, 0.0);
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            input_data[i] = static_cast<double>(tokens[i]);
        }
        const Value logits = model.forward(Value(std::move(input_data), false));
        const std::size_t next_token = select_temperature_token(logits.data(), tokens.size() - 1,
                                                                model.vocab_size(), temperature, generator);
        tokens.push_back(next_token);
    }
    return tokens;
}

} // namespace nexmind
