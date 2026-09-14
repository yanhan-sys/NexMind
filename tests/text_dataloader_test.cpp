#include "text_dataloader.h"

#include <cassert>
#include <cstddef>
#include <vector>

using namespace nexmind;

int main() {
    TextDataset dataset({0, 1, 2, 3, 4, 5}, 2);
    TextDataLoader loader(dataset, 2);

    std::vector<std::vector<std::size_t>> inputs;
    std::vector<std::vector<std::size_t>> targets;

    assert(loader.next(inputs, targets));
    assert(inputs.size() == 2);
    assert(targets.size() == 2);
    assert(inputs[0] == std::vector<std::size_t>({0, 1}));
    assert(targets[0] == std::vector<std::size_t>({1, 2}));
    assert(inputs[1] == std::vector<std::size_t>({1, 2}));
    assert(targets[1] == std::vector<std::size_t>({2, 3}));

    assert(loader.next(inputs, targets));
    assert(inputs.size() == 2);
    assert(inputs[0] == std::vector<std::size_t>({2, 3}));
    assert(inputs[1] == std::vector<std::size_t>({3, 4}));

    assert(loader.next(inputs, targets));
    assert(inputs.size() == 1);
    assert(inputs[0] == std::vector<std::size_t>({4, 5}));
    assert(targets[0] == std::vector<std::size_t>({5, 0}) || targets[0] == std::vector<std::size_t>({5}));

    loader.reset();
    assert(loader.next(inputs, targets));
    assert(inputs[0] == std::vector<std::size_t>({0, 1}));
    return 0;
}
