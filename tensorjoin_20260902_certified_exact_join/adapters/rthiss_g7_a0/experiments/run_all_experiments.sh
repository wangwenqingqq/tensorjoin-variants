#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

fmt_duration() {
  local total="$1"
  printf '%02dh:%02dm:%02ds' $((total / 3600)) $((total % 3600 / 60)) $((total % 60))
}

if [[ $# -gt 0 ]]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=(default time_distribution batching data_copy one_point)
fi

declare -a TIMED_EXPERIMENTS=()
declare -a TIMED_SECONDS=()
total_elapsed=0

for exp in "${EXPERIMENTS[@]}"; do
  script="${EXP_DIR}/run_${exp}.sh"
  if [[ ! -x "${script}" ]]; then
    echo "ERROR: unknown experiment: ${exp}" >&2
    continue
  fi
  echo "==========================================================="
  echo " experiment: ${exp}   (datasets: ${DATASETS})"
  echo "==========================================================="
  exp_start=${SECONDS}
  status=0
  "${script}" || status=$?
  exp_elapsed=$((SECONDS - exp_start))
  TIMED_EXPERIMENTS+=("${exp}")
  TIMED_SECONDS+=("${exp_elapsed}")
  total_elapsed=$((total_elapsed + exp_elapsed))
  echo ">>> [${exp}] total time: $(fmt_duration "${exp_elapsed}") (${exp_elapsed}s)"
  if [[ ${status} -ne 0 ]]; then
    echo "ERROR: experiment ${exp} exited with status ${status}" >&2
  fi
done

echo
echo "==========================================================="
echo " timing summary"
echo "==========================================================="
for i in "${!TIMED_EXPERIMENTS[@]}"; do
  printf '  %-20s %s (%ss)\n' \
      "${TIMED_EXPERIMENTS[$i]}" \
      "$(fmt_duration "${TIMED_SECONDS[$i]}")" \
      "${TIMED_SECONDS[$i]}"
done
printf '  %-20s %s (%ss)\n' "TOTAL" "$(fmt_duration "${total_elapsed}")" "${total_elapsed}"

echo
echo "results: ${RESULTS_ROOT}/results_<experiment>.txt"
echo "plots:   ${PLOT_ROOT}/<experiment>.pdf"
