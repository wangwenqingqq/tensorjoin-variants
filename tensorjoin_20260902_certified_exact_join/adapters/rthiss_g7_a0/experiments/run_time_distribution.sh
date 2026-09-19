#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP=time_distribution

ensure_datasets
start_experiment "${EXP}"

mapfile -t DATASET_LIST < <(active_datasets)

for name in "${DATASET_LIST[@]}"; do
  echo "=== ${name} (DIM=${DIMS[$name]}, kd=${KD_LEVELS[$name]}) ==="
  build_config "${DIMS[$name]}" \
    "MAX_KD_LEVELS_OPT=${KD_LEVELS[$name]}" \
    "${BASE_OPTS[@]}" \
    ${VARIANT_PRIM_SHARED_QUERY_SHARED}
  for eps in ${EPSILONS[$name]}; do
    run_case "${EXP}" "${name}" "${eps}"
  done
done

make_plot "${EXP}" plot_time_distribution.py
