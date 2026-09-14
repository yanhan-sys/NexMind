#pragma once

#include "parameter.h"

#include <cstddef>
#include <vector>

namespace nexmind {

class Adam {
public:
    explicit Adam(std::vector<Parameter*> parameters,
                  double learning_rate = 1e-3,
                  double beta1 = 0.9,
                  double beta2 = 0.999,
                  double epsilon = 1e-8);

    void step();
    void zero_grad();

    std::size_t step_count() const noexcept { return step_count_; }

private:
    struct State {
        Tensor first_moment;
        Tensor second_moment;
    };

    std::vector<Parameter*> parameters_;
    std::vector<State> states_;
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    std::size_t step_count_ = 0;
};

} // namespace nexmind
