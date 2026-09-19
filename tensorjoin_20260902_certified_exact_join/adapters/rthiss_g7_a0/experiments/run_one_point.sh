#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP=one_point

ensure_datasets
start_experiment "${EXP}"

mapfile -t DATASET_LIST < <(active_datasets)

for name in "${DATASET_LIST[@]}"; do
  dim="${DIMS[$name]}"

  echo "=== ${name}: proposed kd height (${KD_LEVELS[$name]}) ==="
  build_config "${dim}" \
    "MAX_KD_LEVELS_OPT=${KD_LEVELS[$name]}" \
    "${BASE_OPTS[@]}" \
    ${VARIANT_PRIM_SHARED_QUERY_SHARED}
  for eps in ${EPSILONS[$name]}; do
    run_case "${EXP}" "${name}" "${eps}"
  done

  echo "=== ${name}: one point per primitive (${KD_LEVELS_ONE_POINT[$name]}) ==="
  build_config "${dim}" \
    "MAX_KD_LEVELS_OPT=${KD_LEVELS_ONE_POINT[$name]}" \
    "${BASE_OPTS[@]}" \
    ${VARIANT_PRIM_SHARED_QUERY_SHARED}
  for eps in ${EPSILONS[$name]}; do
    run_case "${EXP}" "${name}" "${eps}"
  done
done

make_plot "${EXP}" plot_one_point.py
