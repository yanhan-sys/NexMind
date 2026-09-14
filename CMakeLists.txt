cmake_minimum_required(VERSION 3.20)
project(NexMind VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NEXMIND_BUILD_TESTS "Build NexMind tests" ON)

add_library(nexmind_core
    src/tensor.cpp
    src/tensor.h
    src/autograd.cpp
    src/autograd.h
    src/parameter.cpp
    src/parameter.h
    src/module.h
    src/linear.cpp
    src/linear.h
    src/optimizer.cpp
    src/optimizer.h
    src/trainer.cpp
    src/trainer.h
    src/relu.cpp
    src/relu.h
    src/mlp.cpp
    src/mlp.h
    src/embedding.cpp
    src/embedding.h
    src/layer_norm.cpp
    src/layer_norm.h
    src/attention.cpp
    src/attention.h
    src/multi_head_attention.cpp
    src/multi_head_attention.h
    src/transformer_block.cpp
    src/transformer_block.h
    src/transformer_encoder.cpp
    src/transformer_encoder.h
    src/transformer_lm.cpp
    src/transformer_lm.h
    src/language_model_trainer.cpp
    src/language_model_trainer.h
)

target_compile_features(nexmind_core PUBLIC cxx_std_20)
target_compile_options(nexmind_core PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

target_include_directories(nexmind_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(nexmind
    src/main.cpp
)

target_link_libraries(nexmind PRIVATE nexmind_core)
target_compile_features(nexmind PRIVATE cxx_std_20)
target_compile_options(nexmind PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

if(NEXMIND_BUILD_TESTS)
    enable_testing()

    add_executable(nexmind_tensor_test tests/tensor_test.cpp)
    target_link_libraries(nexmind_tensor_test PRIVATE nexmind_core)
    target_compile_features(nexmind_tensor_test PRIVATE cxx_std_20)
    target_compile_options(nexmind_tensor_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

    add_executable(nexmind_autograd_test tests/autograd_test.cpp)
    target_link_libraries(nexmind_autograd_test PRIVATE nexmind_core)
    target_compile_features(nexmind_autograd_test PRIVATE cxx_std_20)
    target_compile_options(nexmind_autograd_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

    add_executable(nexmind_linear_test tests/linear_test.cpp)
    target_link_libraries(nexmind_linear_test PRIVATE nexmind_core)
    target_compile_features(nexmind_linear_test PRIVATE cxx_std_20)
    target_compile_options(nexmind_linear_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

    add_executable(nexmind_optimizer_test tests/optimizer_test.cpp)
    target_link_libraries(nexmind_optimizer_test PRIVATE nexmind_core)
    target_compile_features(nexmind_optimizer_test PRIVATE cxx_std_20)
    target_compile_options(nexmind_optimizer_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

    add_executable(nexmind_trainer_test tests/trainer_test.cpp)
    target_link_libraries(nexmind_trainer_test PRIVATE nexmind_core)
    target_compile_features(nexmind_trainer_test PRIVATE cxx_std_20)
    target_compile_options(nexmind_trainer_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

    add_executable(nexmind_mlp_test tests/mlp_test.cpp)
    target_link_libraries(nexmind_mlp_test PRIVATE nexmind_core)
    target_compile_features(nexmind_mlp_test PRIVATE cxx_std_20)
    target_compile_options(nexmind_mlp_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

    add_executable(nexmind_transformer_components_test tests/transformer_components_test.cpp)
    target_link_libraries(nexmind_transformer_components_test PRIVATE nexmind_core)
    target_compile_features(nexmind_transformer_components_test PRIVATE cxx_std_20)
    target_compile_options(nexmind_transformer_components_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

    add_executable(nexmind_attention_test tests/attention_test.cpp)
    target_link_libraries(nexmind_attention_test PRIVATE nexmind_core)
    target_compile_features(nexmind_attention_test PRIVATE cxx_std_20)
    target_compile_options(nexmind_attention_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

    add_executable(nexmind_multi_head_attention_test tests/multi_head_attention_test.cpp)
    target_link_libraries(nexmind_multi_head_attention_test PRIVATE nexmind_core)
    target_compile_features(nexmind_multi_head_attention_test PRIVATE cxx_std_20)
    target_compile_options(nexmind_multi_head_attention_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

    add_executable(nexmind_transformer_block_test tests/transformer_block_test.cpp)
    target_link_libraries(nexmind_transformer_block_test PRIVATE nexmind_core)
    target_compile_features(nexmind_transformer_block_test PRIVATE cxx_std_20)
    target_compile_options(nexmind_transformer_block_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

    add_executable(nexmind_transformer_encoder_test tests/transformer_encoder_test.cpp)
    target_link_libraries(nexmind_transformer_encoder_test PRIVATE nexmind_core)
    target_compile_features(nexmind_transformer_encoder_test PRIVATE cxx_std_20)
    target_compile_options(nexmind_transformer_encoder_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

    add_executable(nexmind_transformer_lm_test tests/transformer_lm_test.cpp)
    target_link_libraries(nexmind_transformer_lm_test PRIVATE nexmind_core)
    target_compile_features(nexmind_transformer_lm_test PRIVATE cxx_std_20)
    target_compile_options(nexmind_transformer_lm_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

    add_executable(nexmind_language_model_trainer_test tests/language_model_trainer_test.cpp)
    target_link_libraries(nexmind_language_model_trainer_test PRIVATE nexmind_core)
    target_compile_features(nexmind_language_model_trainer_test PRIVATE cxx_std_20)
    target_compile_options(nexmind_language_model_trainer_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

    add_test(NAME nexmind_tensor COMMAND nexmind_tensor_test)
    add_test(NAME nexmind_autograd COMMAND nexmind_autograd_test)
    add_test(NAME nexmind_linear COMMAND nexmind_linear_test)
    add_test(NAME nexmind_optimizer COMMAND nexmind_optimizer_test)
    add_test(NAME nexmind_trainer COMMAND nexmind_trainer_test)
    add_test(NAME nexmind_mlp COMMAND nexmind_mlp_test)
    add_test(NAME nexmind_transformer_components COMMAND nexmind_transformer_components_test)
    add_test(NAME nexmind_attention COMMAND nexmind_attention_test)
    add_test(NAME nexmind_multi_head_attention COMMAND nexmind_multi_head_attention_test)
    add_test(NAME nexmind_transformer_block COMMAND nexmind_transformer_block_test)
    add_test(NAME nexmind_transformer_encoder COMMAND nexmind_transformer_encoder_test)
    add_test(NAME nexmind_transformer_lm COMMAND nexmind_transformer_lm_test)
    add_test(NAME nexmind_language_model_trainer COMMAND nexmind_language_model_trainer_test)
    add_test(NAME nexmind_smoke COMMAND nexmind)
endif()