#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP=data_copy

ensure_datasets
start_experiment "${EXP}"

mapfile -t DATASET_LIST < <(active_datasets)

for name in "${DATASET_LIST[@]}"; do
  dim="${DIMS[$name]}"
  kd="${KD_LEVELS[$name]}"

  echo "=== ${name}: compressed mask ==="
  build_config "${dim}" \
    "MAX_KD_LEVELS_OPT=${kd}" \
    "${BASE_OPTS[@]}" \
    ${VARIANT_PRIM_SHARED_QUERY_SHARED}
  for eps in ${EPSILONS[$name]}; do
    run_case "${EXP}" "${name}" "${eps}"
  done

  echo "=== ${name}: uncompressed mask ==="
  build_config "${dim}" \
    "MAX_KD_LEVELS_OPT=${kd}" \
    "${BASE_OPTS[@]}" \
    ${VARIANT_PRIM_SHARED_QUERY_SHARED} \
    "USE_UNCOMPRESSED_MASK_OPT=ON"
  for eps in ${EPSILONS[$name]}; do
    run_case "${EXP}" "${name}" "${eps}"
  done

  echo "=== ${name}: key-value pair copy ==="
  build_config "${dim}" \
    "MAX_KD_LEVELS_OPT=${kd}" \
    "${BASE_OPTS[@]}" \
    ${VARIANT_PRIM_SHARED_QUERY_SHARED} \
    "USE_USE_CANDIDATE_POINT_COPY_OPT=ON"
  for eps in ${EPSILONS[$name]}; do
    run_case "${EXP}" "${name}" "${eps}"
  done
done

make_plot "${EXP}" plot_data_copy.py
