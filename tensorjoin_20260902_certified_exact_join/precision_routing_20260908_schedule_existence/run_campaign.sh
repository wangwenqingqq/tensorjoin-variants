#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=@TENSORJOIN_ROOT@/isaacsim6/env/bin/python
export PYTHONDONTWRITEBYTECODE=1 NVIDIA_TF32_OVERRIDE=0
export TRITON_CACHE_DIR="$PWD/artifacts/private_triton_cache"
for tool in memcheck initcheck synccheck racecheck; do
 "$PY" src/guard_r2.py --label "${tool}_a0" --gpu 2 --target 8p_gpu2 -- \
 /usr/local/cuda-13.1/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 \
 "$PY" -u src/safety_r1.py --label "${tool}_a0"
done
"$PY" src/admit.py
for i in 0 1 2 3 4 5; do
 label=$(printf 'p%02d_a0' "$i")
 "$PY" src/guard_r2.py --label "$label" --gpu 2 --target 8p_gpu2 -- \
 "$PY" -u src/runner.py --index "$i"
done
"$PY" src/analyze.py > raw/analysis_stdout.txt
