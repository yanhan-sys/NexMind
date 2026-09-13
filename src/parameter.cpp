#include "parameter.h"

#include <utility>

namespace nexmind {

Parameter::Parameter(Tensor data, std::string name) : value_(std::move(data), true, std::move(name)) {}

} // namespace nexmind
