cmake_minimum_required(VERSION 3.24)

project(llm_serving_simulator LANGUAGES CXX)

option(SERVING_BUILD_CUDA "Build the optional PagedAttention CUDA benchmark" ON)
option(SERVING_BUILD_TESTS "Build unit tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(serving_core
    src/kv_cache.cpp
    src/latency_model.cpp
    src/metrics.cpp
    src/simulator.cpp
    src/workload.cpp
)

target_include_directories(serving_core PUBLIC include)

if(MSVC)
    target_compile_options(serving_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(serving_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

add_executable(serving_sim src/main.cpp)
target_link_libraries(serving_sim PRIVATE serving_core)

if(SERVING_BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        add_library(paged_attention_cuda cuda/paged_attention.cu)
        target_include_directories(paged_attention_cuda PUBLIC include)
        target_compile_features(paged_attention_cuda PUBLIC cxx_std_20)
        set_target_properties(paged_attention_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )

        target_compile_definitions(serving_sim PRIVATE SERVING_HAS_CUDA=1)
        target_link_libraries(serving_sim PRIVATE paged_attention_cuda)
    else()
        message(STATUS "CUDA compiler was not found; building simulator without the CUDA benchmark")
        target_sources(serving_sim PRIVATE src/cuda_bench_stub.cpp)
    endif()
else()
    target_sources(serving_sim PRIVATE src/cuda_bench_stub.cpp)
endif()

if(SERVING_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_executable(serving_tests tests/serving_tests.cpp)
        target_link_libraries(serving_tests PRIVATE serving_core)
        add_test(NAME serving_tests COMMAND serving_tests)
    endif()
endif()
