cmake_minimum_required(VERSION 3.20)
project(NexMind VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NEXMIND_BUILD_TESTS "Build NexMind tests" ON)

add_executable(nexmind
    src/main.cpp
)

target_compile_features(nexmind PRIVATE cxx_std_17)

target_include_directories(nexmind PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(NEXMIND_BUILD_TESTS)
    enable_testing()
    add_test(NAME nexmind_smoke COMMAND nexmind)
endif()
