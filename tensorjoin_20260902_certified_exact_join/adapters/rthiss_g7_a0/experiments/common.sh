#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${EXP_DIR}/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${REPO_DIR}/build}"
RESULTS_ROOT="${RESULTS_ROOT:-${EXP_DIR}/results}"
PLOT_ROOT="${PLOT_ROOT:-${EXP_DIR}/plots}"
DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/scDatasets}"
OPTIX_DIR="${OPTIX_DIR:-${REPO_DIR}/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64}"
BINARY="${BUILD_DIR}/RT-HiSS"
VENV_DIR="${VENV_DIR:-${EXP_DIR}/.venv}"
BASE_PYTHON="${BASE_PYTHON:-python3}"
PYTHON="${PYTHON:-}"

DATASETS="${DATASETS:-wave, msd, susy, bigcross}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"

declare -A DIMS=(
  [wave]=49
  [msd]=90
  [susy]=18
  [bigcross]=57
  [higgs]=28
)

declare -A EPSILONS=(
  [wave]="0.0054 0.00702 0.008358"
  [msd]="0.0076 0.00913 0.011334"
  [susy]="0.01703 0.02078 0.025555"
  [bigcross]="0.0131 0.01994 0.0281"
  [higgs]="0.049186 0.05558 0.063117"
)

declare -A KD_LEVELS=(
  [wave]=12
  [msd]=13
  [susy]=15
  [bigcross]=16
  [higgs]=16
)

declare -A KD_LEVELS_ONE_POINT=(
  [wave]=19
  [msd]=19
  [susy]=23
  [bigcross]=24
  [higgs]=24
)

VARIANT_PRIM_SHARED_QUERY_GLOBAL="USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING_OPT=ON USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING_OPT=OFF USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING_OPT=OFF USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING_OPT=OFF USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING_OPT=OFF"
VARIANT_PRIM_SHARED_QUERY_SHARED="USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING_OPT=OFF USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING_OPT=ON USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING_OPT=OFF USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING_OPT=OFF USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING_OPT=OFF"
VARIANT_PRIM_GLOBAL_QUERY_SHARED="USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING_OPT=OFF USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING_OPT=OFF USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING_OPT=ON USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING_OPT=OFF USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING_OPT=OFF"
VARIANT_PRIM_GLOBAL_QUERY_GLOBAL="USE_PRIMITIVE_SHARED_QUERY_GLOBAL_BATCHING_OPT=OFF USE_PRIMITIVE_SHARED_QUERY_SHARED_BATCHING_OPT=OFF USE_PRIMITIVE_GLOBAL_QUERY_SHARED_BATCHING_OPT=OFF USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_BATCHING_OPT=ON USE_PRIMITIVE_GLOBAL_QUERY_GLOBAL_NON_BATCHING_OPT=OFF"

BASE_OPTS=(
  "THREADS_TO_COPY_OPT=8"
  "PINNED_MEMORY_SIZE_OPT=16777216"
  "USE_UNCOMPRESSED_MASK_OPT=OFF"
  "USE_PAGEABLE_MEMORY_OPT=OFF"
  "USE_USE_CANDIDATE_POINT_COPY_OPT=OFF"
)

active_datasets() {
  local list name
  read -r -a list <<< "${DATASETS//,/ }"
  for name in "${list[@]}"; do
    if [[ -z "${DIMS[$name]:-}" ]]; then
      echo "ERROR: unknown dataset '${name}' in DATASETS" >&2
      exit 1
    fi
  done
  printf '%s\n' "${list[@]}"
}

ensure_datasets() {
  if [[ -d "${DATASET_DIR}" ]]; then
    return 0
  fi
  if [[ -f "${REPO_DIR}/scDatasets.tar.gz" ]]; then
    echo ">>> extracting scDatasets.tar.gz"
    tar xzf "${REPO_DIR}/scDatasets.tar.gz" -C "${REPO_DIR}"
  fi
  if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "ERROR: dataset directory not found: ${DATASET_DIR}" >&2
    echo "       set DATASET_DIR=/path/to/datasets" >&2
    exit 1
  fi
}

dataset_path() {
  echo "${DATASET_DIR}/$1.txt"
}

build_config() {
  local dim="$1"
  shift
  local flags=()
  local flag
  for flag in "$@"; do
    flags+=("-D${flag}")
  done
  cmake -S "${REPO_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DOptiX_INSTALL_DIR="${OPTIX_DIR}" \
        -DDIM="${dim}" \
        "${flags[@]}" > /dev/null
  cmake --build "${BUILD_DIR}" --target RT-HiSS -j"$(nproc)" > /dev/null
}

ensure_python_env() {
  if [[ -n "${PYTHON}" ]]; then
    return 0
  fi
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo ">>> creating python virtualenv in ${VENV_DIR}"
    "${BASE_PYTHON}" -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip
    "${VENV_DIR}/bin/python" -m pip install --quiet -r "${EXP_DIR}/requirements.txt"
  fi
  PYTHON="${VENV_DIR}/bin/python"
  "${PYTHON}" -c "import matplotlib, pandas, numpy" || {
    echo "ERROR: python dependencies missing in ${VENV_DIR}" >&2
    exit 1
  }
}

experiment_workdir() {
  echo "${RESULTS_ROOT}/$1"
}

experiment_results() {
  echo "${RESULTS_ROOT}/results_$1.txt"
}

start_experiment() {
  local exp="$1"
  ensure_python_env
  mkdir -p "$(experiment_workdir "${exp}")" "${PLOT_ROOT}"
  : > "$(experiment_results "${exp}")"
  rm -f "$(experiment_workdir "${exp}")/results.txt"
}

run_case() {
  local exp="$1" name="$2" eps="$3"
  local data workdir results
  data="$(dataset_path "${name}")"
  workdir="$(experiment_workdir "${exp}")"
  results="$(experiment_results "${exp}")"
  if [[ ! -f "${data}" ]]; then
    echo "ERROR: missing dataset ${data}, skipping" >&2
    return 0
  fi
  echo ">>> [${exp}] ${name} epsilon=${eps}"
  if ( cd "${workdir}" && "${BINARY}" "${data}" "${eps}" >> run.log 2>&1 ); then
    if [[ -f "${workdir}/results.txt" ]]; then
      cat "${workdir}/results.txt" >> "${results}"
      rm -f "${workdir}/results.txt"
    fi
  else
    echo "ERROR: run failed: ${name} eps=${eps} (see ${workdir}/run.log)" >&2
    rm -f "${workdir}/results.txt"
  fi
}

write_csv() {
  local exp="$1"
  "${PYTHON}" "${EXP_DIR}/parse_results.py" \
      --results "$(experiment_results "${exp}")" \
      --out "${RESULTS_ROOT}/results_${exp}.csv"
}

make_plot() {
  local exp="$1" script="$2"
  local results
  results="$(experiment_results "${exp}")"
  if [[ ! -s "${results}" ]]; then
    echo "ERROR: no results in ${results}, skipping plot" >&2
    return 0
  fi
  "${PYTHON}" "${EXP_DIR}/${script}" --results "${results}" --out "${PLOT_ROOT}/${exp}.pdf"
  write_csv "${exp}"
}
