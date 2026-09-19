cmake_minimum_required(VERSION 3.18)
project(GTS_G6 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories(include)
add_executable(GTS_G6 src/main_g6.cu)
target_compile_options(GTS_G6 PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O2>)
