cmake_minimum_required(VERSION 3.16)
project(RT-HiSS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architecture override")

enable_language(CUDA)

set(OptiX_INSTALL_DIR "/path/to/NVIDIA-OptiX-SDK" CACHE PATH "Path to the OptiX SDK")
set(CMAKE_PREFIX_PATH "${OptiX_INSTALL_DIR}" ${CMAKE_PREFIX_PATH})

find_package(OpenMP REQUIRED)

set(DIM 18 CACHE STRING "Dataset Dimension")
if(DEFINED DIM)
    set(DIM ${DIM})
endif()

# Options to select batching/batching and memory access scheme
option(USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING_OPT "primitive=shared, query=global batching" OFF)
option(USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING_OPT "primitive=shared, query=shared batching(default)" ON)
option(USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING_OPT "primitive=global, query=shared batching" OFF)
option(USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING_OPT "primitive=global, query=global batching" OFF)
option(USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING_OPT "primitive=global, query=global Non-batching (baseline)" OFF)


option(USE_UNCOMPRESSED_MASK_OPT "Use uncompressed mask to copy results" OFF)
option(USE_PAGEABLE_MEMORY_OPT "Use pageable memory instead of pinned memory to copy" OFF)
option(USE_FIXED_WORK_PER_BLOCK_OPT "Use fixed work per thread block" ON)
option(USE_SORT_BY_WORKLOAD_OPT "Sort the primitive by workload/number of primitives" ON)
option(USE_USE_CANDIDATE_POINT_COPY_OPT "Copy packed key value pairs" OFF)


# Number of kd tree levels. -1 (the default) means "auto": the height is chosen
# at runtime as floor(ln(|D|)). Set an explicit positive value to override.
set(MAX_KD_LEVELS_OPT -1 CACHE STRING "kd tree height; -1 = auto floor(ln(|D|))")
if(DEFINED MAX_KD_LEVELS_OPT)
    set(MAX_KD_LEVELS_OPT ${MAX_KD_LEVELS_OPT})
endif()

set(THREADS_TO_COPY_OPT 8 CACHE STRING "NUmber of threads to copy results to CPU")
if(DEFINED THREADS_TO_COPY_OPT)
    set(THREADS_TO_COPY_OPT ${THREADS_TO_COPY_OPT})
endif()

set(PINNED_MEMORY_SIZE_OPT 16*1024*1024 CACHE STRING "Pinned memory in bytes")
if(DEFINED PINNED_MEMORY_SIZE_OPT)
    set(PINNED_MEMORY_SIZE_OPT ${PINNED_MEMORY_SIZE_OPT})
endif()

set(NUM_CALCULATIONS_PER_THREAD_OPT 128 CACHE STRING "Number of point comparisons per thread")
if(DEFINED NUM_CALCULATIONS_PER_THREAD_OPT)
    set(NUM_CALCULATIONS_PER_THREAD_OPT ${NUM_CALCULATIONS_PER_THREAD_OPT})
endif()

# RT-HiSS has no viewer, so GLFW's windowing backends are not needed.
# Disabling them avoids requiring X11/Wayland development headers to build OWL.
set(GLFW_BUILD_WAYLAND OFF)
set(GLFW_BUILD_X11 OFF)

# Using OWL as submodule
set(owl_dir ${PROJECT_SOURCE_DIR}/OWL)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

embed_ptx(
        OUTPUT_TARGET
        RT-HiSS-ptx
        PTX_LINK_LIBRARIES
        owl::owl
        SOURCES
        deviceCode.cu
)

add_executable(RT-HiSS hostCode.cpp utility.cu utility_seq.cpp utility.h)

set(RT-HiSS_COMPILE_DEFINITIONS
        DATASET_DIM=${DIM}
        MAX_KD_LEVELS_OPT=${MAX_KD_LEVELS_OPT}
        THREADS_TO_COPY_OPT=${THREADS_TO_COPY_OPT}
        PINNED_MEMORY_SIZE_OPT=${PINNED_MEMORY_SIZE_OPT}
        NUM_CALCULATIONS_PER_THREAD_OPT=${NUM_CALCULATIONS_PER_THREAD_OPT}
        USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING_OPT=$<BOOL:${USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING_OPT}>
        USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING_OPT=$<BOOL:${USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING_OPT}>
        USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING_OPT=$<BOOL:${USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING_OPT}>
        USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING_OPT=$<BOOL:${USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING_OPT}>
        USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING_OPT=$<BOOL:${USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING_OPT}>
        USE_UNCOMPRESSED_MASK_OPT=$<BOOL:${USE_UNCOMPRESSED_MASK_OPT}>
        USE_PAGEABLE_MEMORY_OPT=$<BOOL:${USE_PAGEABLE_MEMORY_OPT}>
        USE_FIXED_WORK_PER_BLOCK_OPT=$<BOOL:${USE_FIXED_WORK_PER_BLOCK_OPT}>
        USE_SORT_BY_WORKLOAD_OPT=$<BOOL:${USE_SORT_BY_WORKLOAD_OPT}>
        USE_USE_CANDIDATE_POINT_COPY_OPT=$<BOOL:${USE_USE_CANDIDATE_POINT_COPY_OPT}>
)

if(DEFINED SAFE_SHARED_MEMORY)
    list(APPEND RT-HiSS_COMPILE_DEFINITIONS SAFE_SHARED_MEMORY=${SAFE_SHARED_MEMORY})
endif()

target_compile_definitions(RT-HiSS PRIVATE ${RT-HiSS_COMPILE_DEFINITIONS})
target_compile_definitions(RT-HiSS-ptx PRIVATE ${RT-HiSS_COMPILE_DEFINITIONS})

# Thrust/CUB headers (CUDA 13 ships them under include/cccl) for the host C++ sources
get_filename_component(CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_TOOLKIT_ROOT "${CUDA_BIN_DIR}" DIRECTORY)
if(EXISTS "${CUDA_TOOLKIT_ROOT}/include/cccl/thrust")
    target_include_directories(RT-HiSS PRIVATE "${CUDA_TOOLKIT_ROOT}/include/cccl")
endif()
target_include_directories(RT-HiSS PRIVATE "${CUDA_TOOLKIT_ROOT}/include")

target_compile_options(RT-HiSS PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-Xcompiler=-fopenmp>)
# target_compile_options(RT-HiSS PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda> -lineinfo)

target_link_libraries(RT-HiSS
        PRIVATE
        RT-HiSS-ptx
        owl::owl
        stb_image
        OpenMP::OpenMP_CXX
)
