cmake_minimum_required(VERSION 3.22)
project(rnn2_gpu_probe LANGUAGES CXX)

find_package(Torch REQUIRED)
add_executable(rnn2_probe src/probe.cpp)
target_compile_features(rnn2_probe PRIVATE cxx_std_20)
target_link_libraries(rnn2_probe PRIVATE "${TORCH_LIBRARIES}" ${CMAKE_DL_LIBS})
separate_arguments(TORCH_EXTRA_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(rnn2_probe PRIVATE ${TORCH_EXTRA_FLAGS} -Wall -Wextra -Wpedantic)

# Deliberately host-only C++: tensor kernels come from the selected LibTorch build.
# Configure with that installation's CMAKE_PREFIX_PATH; do not silently substitute
# a CPU-only build when the requested runtime device is CUDA/HIP.
