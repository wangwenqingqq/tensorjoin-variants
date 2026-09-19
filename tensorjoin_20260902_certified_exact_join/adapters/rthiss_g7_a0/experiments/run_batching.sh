#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP=batching

VARIANT_NAMES=(
  prim_global_query_shared
  prim_shared_query_global
  prim_shared_query_shared
  prim_global_query_global
)

ensure_datasets
start_experiment "${EXP}"

mapfile -t DATASET_LIST < <(active_datasets)

for variant in "${VARIANT_NAMES[@]}"; do
  echo "=== variant: ${variant} ==="
  variant_var="VARIANT_${variant^^}"
  for name in "${DATASET_LIST[@]}"; do
    echo "--- building ${name} (DIM=${DIMS[$name]}, kd=${KD_LEVELS[$name]})"
    build_config "${DIMS[$name]}" \
      "MAX_KD_LEVELS_OPT=${KD_LEVELS[$name]}" \
      "${BASE_OPTS[@]}" \
      ${!variant_var}
    for eps in ${EPSILONS[$name]}; do
      run_case "${EXP}" "${name}" "${eps}"
    done
  done
done

make_plot "${EXP}" plot_batching.py
