#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP=default

ensure_datasets
start_experiment "${EXP}"

mapfile -t DATASET_LIST < <(active_datasets)

for name in "${DATASET_LIST[@]}"; do
  echo "=== Configuring build: dataset=${name}, DIM=${DIMS[$name]} ==="
  build_config "${DIMS[$name]}" \
    "MAX_KD_LEVELS_OPT=-1" \
    "${BASE_OPTS[@]}" \
    ${VARIANT_PRIM_SHARED_QUERY_SHARED}

  for eps in ${EPSILONS[$name]}; do
    run_case "${EXP}" "${name}" "${eps}"
  done
done

write_csv "${EXP}"
