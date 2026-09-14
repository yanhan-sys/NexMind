#pragma once

#include "transformer_lm.h"

#include <string>

namespace nexmind {

// Checkpoint：保存模型结构信息和全部参数，使用标准二进制格式，不依赖第三方库。
void save_checkpoint(const TransformerLanguageModel& model, const std::string& path);
TransformerLanguageModel load_checkpoint(const std::string& path);

} // namespace nexmind
