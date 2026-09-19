#!/usr/bin/env bash
# Reproduction commands recorded after the observed a1 build.
# Run on the verified 8P host; do not change global dependencies.
set -euo pipefail
cd @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join
cmake -S adapters/rthiss_g7_a0 -B adapters/rthiss_g7_a0/build_sample_sm120_a1 \
  -DCMAKE_BUILD_TYPE=Release -DDIM=2 -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DOptiX_INSTALL_DIR=@TENSORJOIN_ROOT@/RT-TIDE/deps/optix-9.1
cmake --build adapters/rthiss_g7_a0/build_sample_sm120_a1 --target RT-HiSS -j4
# Select a fresh attempt ID; the runner refuses to overwrite an existing run.
# python3 src/run_g7_rthiss_smoke.py --attempt a2
