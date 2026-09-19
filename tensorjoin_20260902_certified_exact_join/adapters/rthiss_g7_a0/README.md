# RT-HiSS — Ray Tracing Accelerated High-Dimensional Similarity Search


## Requirements
- CMake 3.16+
- CUDA toolkit (nvcc) and an NVIDIA GPU with RT cores
- OptiX SDK compatible with OWL (download from https://developer.nvidia.com/designworks/optix/downloads/legacy)
- C++17 compiler with OpenMP support

Tested on: Quadro RTX 5000 and Quadro RTX 6000 with OptiX 7.4 and 7.7.

## Quickstart
### Setup
1) Initialize the OWL submodule:
```bash
git submodule update --init --recursive
```

2) Configure paths (pass with `-D` or edit `CMakeLists.txt`):
- `OptiX_INSTALL_DIR` must point to your OptiX SDK, e.g. `-DOptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-7.x` (no default)
- `CMAKE_CUDA_COMPILER` is auto-detected from `PATH`; override with `-DCMAKE_CUDA_COMPILER=/path/to/nvcc` if needed
- `CMAKE_CUDA_ARCHITECTURES` should match your GPU (default is 75)

### Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDIM=<num_dimensions>
cmake --build build -j
```

Example for the sample dataset (2 dimensions):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDIM=2
cmake --build build -j
```

### Run
```bash
./build/RT-HiSS <dataset.txt> <epsilon> [reorder_mode]
```

Example:
```bash
./build/RT-HiSS sampleDataset/iono_57_1000.txt 0.01
```

## Dataset format
- CSV with numeric values, no header
- Each row is a point
- Number of columns must match `DIM` at build time

A sample dataset is available in `sampleDataset/`.

### Fetching the SC'26 datasets
The datasets used in SC'26 are stored in this repository with [Git LFS](https://git-lfs.com/) (Large File Storage). Git only clones a small pointer file for `scDatasets.tar.gz`, so you need Git LFS installed to download the actual contents:

```bash
sudo apt install git-lfs   
git lfs install           
git lfs pull              
```

Unpack the archive with:
```bash
tar -xvzf scDatasets.tar.gz
```
Alternatively, you can download the datasets directly from https://rcdata.nau.edu/gowanlock_lab/datasets/RT-HiSS_datasets.


The scripts in `experiments/` can be used to reproduce the results reported in the paper.

## Tunable parameters

### Runtime arguments
- `dataset.txt`: input dataset path
- `epsilon`: search distance (float)
- `reorder_mode` (optional): reorders dimensions by variance. One of:
  - `highest` — highest-variance dimensions first (default if omitted)
  - `lowest` — lowest-variance dimensions first
  - `random` — random dimension order

### CMake cache variables (set with `-D`)
- `DIM` : Number of dimensions in the dataset
- `MAX_KD_LEVELS_OPT` (default -1 = auto, floor(ln(|D|))): kd-tree height limit; affects the number of primitives in the OptiX scene
- `THREADS_TO_COPY_OPT` (default 8): Number of CPU threads for result copy
- `PINNED_MEMORY_SIZE_OPT` (default 16\*1024\*1024): Allocated pinned buffer size in bytes shared among all CPU threads
- `NUM_CALCULATIONS_PER_THREAD_OPT` (default 128): Number of point comparisons processed per thread
- `SAFE_SHARED_MEMORY` (default: auto-detected): Usable shared memory per block in bytes. If unset, it is detected at runtime.

### CMake options for point comparison variants
These options select where primitive and query points are loaded from (shared vs. global memory). The default loads both from shared memory.
`USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING_OPT` is the baseline where both are loaded from global memory with tiling disabled.

- `USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING_OPT` (default ON)
- `USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING_OPT`
- `USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING_OPT`
- `USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING_OPT`
- `USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING_OPT`

Note: enable only one batching mode at a time.

### CMake options for data copy variants
- `USE_UNCOMPRESSED_MASK_OPT` (default OFF): Use an uncompressed result mask
- `USE_PAGEABLE_MEMORY_OPT` (default OFF): Use pageable memory instead of pinned memory
- `USE_USE_CANDIDATE_POINT_COPY_OPT` (default OFF): Use key-value pairs to store the result mask

### CMake options for optimizations
- `USE_FIXED_WORK_PER_BLOCK_OPT` (default ON): Enforce a fixed workload per thread block
- `USE_SORT_BY_WORKLOAD_OPT` (default ON): Sort workloads in ascending order to reduce tail effects

### Example build with different parameter configurations
The build below uses the default point comparison variant and data copy variant, plus default values for CPU threads and pinned memory.

```bash
cmake -S . -B build \
  -DDIM=2 \
  -DUSE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING_OPT=ON \
  -DMAX_KD_LEVELS_OPT=13 \
  -DTHREADS_TO_COPY_OPT=8 \
  -DPINNED_MEMORY_SIZE_OPT=16777216 \
  -DSAFE_SHARED_MEMORY=48768 \
  -DUSE_UNCOMPRESSED_MASK_OPT=OFF \
  -DUSE_PAGEABLE_MEMORY_OPT=OFF \
  -DUSE_USE_CANDIDATE_POINT_COPY_OPT=OFF

cmake --build build -j
```

## IMPORTANT Notes
The algorithm relies on GPU shared memory (L1) for performance. Available shared memory depends on GPU architecture. By default `SAFE_SHARED_MEMORY` is detected automatically at runtime. To use a fixed value instead, override it at build time with `-DSAFE_SHARED_MEMORY=<bytes>` (for example `-DSAFE_SHARED_MEMORY=48768` for a Quadro RTX 5000 with 48 KiB of shared memory).

A good value for `MAX_KD_LEVELS_OPT` depends on dataset cardinality. We observe that floor(ln(|D|)) performs well, where |D| is the number of points in the dataset. This parameter has a significant impact on the results.

## Output
Results are appended to a `results.txt` file. Update the path if you want a different output location.

Sample output includes useful statistics to verify parameters and performance.

```text
**********************START OF RESULT**********************

=============================================================
Dataset file: sampleDataset/iono_57_1000.txt
Dataset points: 1000
Dataset dimensions: 2
Epsilon: 0.010000
Reorder mode: highest
USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING: 1
USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING: 0
USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING: 0
USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING: 0
USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING: 0
MAX_KD_LEVELS: 6
THREADS_TO_COPY: 8
USE_UNCOMPRESSED_MASK: 0
USE_PAGEABLE_MEMORY: 0
USE_FIXED_WORK_PER_BLOCK: 1
SORT_BY_WORKLOAD: 1
USE_CANDIDATE_POINT_COPY: 0
PINNED_MEMORY_OPT: 16777216
-------------------------------------------------------------
Total number of queries: 1000
Time to perform only intersection test: 0.002 seconds
Total identified neighbors: 1160
Time for KD tree construction: 0.006 seconds
CUDA kernel only time: 0.000
Time to copy results only: 0.000
Time to compress results: 0.000
Work Generation time: 0.000
Refine time: 0.001 seconds
RT time: 0.000 seconds
TOTAL TIME (REFINE + QUERY): 0.001 seconds
TOTAL TIME INCLUDING EVERYTHING: 0.057 seconds


**********************END OF RESULT**********************
```

## Cite us
If you use this work in research or as a reference implementation, please cite us.
