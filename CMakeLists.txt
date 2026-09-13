cmake_minimum_required(VERSION 3.20)
project(NexMind VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
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
)

target_compile_features(nexmind_core PUBLIC cxx_std_17)

target_include_directories(nexmind_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(nexmind
    src/main.cpp
)

target_link_libraries(nexmind PRIVATE nexmind_core)
target_compile_features(nexmind PRIVATE cxx_std_17)

if(NEXMIND_BUILD_TESTS)
    enable_testing()

    add_executable(nexmind_tensor_test
        tests/tensor_test.cpp
    )
    target_link_libraries(nexmind_tensor_test PRIVATE nexmind_core)
    target_compile_features(nexmind_tensor_test PRIVATE cxx_std_17)

    add_executable(nexmind_autograd_test
        tests/autograd_test.cpp
    )
    target_link_libraries(nexmind_autograd_test PRIVATE nexmind_core)
    target_compile_features(nexmind_autograd_test PRIVATE cxx_std_17)

    add_executable(nexmind_linear_test
        tests/linear_test.cpp
    )
    target_link_libraries(nexmind_linear_test PRIVATE nexmind_core)
    target_compile_features(nexmind_linear_test PRIVATE cxx_std_17)

    add_test(NAME nexmind_tensor COMMAND nexmind_tensor_test)
    add_test(NAME nexmind_autograd COMMAND nexmind_autograd_test)
    add_test(NAME nexmind_linear COMMAND nexmind_linear_test)
    add_test(NAME nexmind_smoke COMMAND nexmind)
endif()
