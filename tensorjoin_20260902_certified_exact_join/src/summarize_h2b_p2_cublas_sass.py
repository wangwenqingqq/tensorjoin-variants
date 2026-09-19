#!/usr/bin/env python3
"""Summarize immutable H2B-P2 Nsight launch and selected-function evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "raw"
ARTIFACTS = PROJECT / "artifacts" / "h2b_p2"
RESULT = PROJECT / "results" / "h2b_p2_cublas_sass_audit.json"
TEXT = RAW / "h2b_p2_cublas_sass_audit.txt"

TRACE_RUNNER_SHA256 = "83a9cc8a6341d8a23854685b462d6688c57f0764c9d3451e919a4b2fa266203e"
CUBLAS_SHA256 = "e70f38efabe986acd5eb683497c62f0f1730a6176ee291d9d24c6e339d1fbf86"
CUBLASLT_SHA256 = "656298c804f5adbb0df930545c17911b9584ab4e5101c0eeb65d1fe881d880f8"

CASES = {
    "esc50_panns": {
        "shape": [640, 2560, 2048],
        "kernel_fragment": "cutlass_80_simt_sgemm_256x128_8x4_tn_align1",
        "grid": [40, 2, 11],
        "block": [256, 1, 1],
        "registers_per_thread": 212,
        "dynamic_shared_kbyte": 49.15,
        "output_sha256": "f09c8720d37a0b20666f05530b08fe767fc9b11ce904c121756e994c921530a7",
    },
    "ucf101_r3d18": {
        "shape": [662, 2684, 512],
        "kernel_fragment": "cutlass_80_simt_sgemm_128x64_8x5_tn_align1",
        "grid": [88, 3, 3],
        "block": [128, 1, 1],
        "registers_per_thread": 130,
        "dynamic_shared_kbyte": 30.72,
        "output_sha256": "39f68b4fbb762457ecd317a2e7ca0610820831e0d94de2961a223030ab61a42f",
    },
}


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def parse_json_from_log(path: Path) -> dict:
    text = path.read_text()
    start = text.find("{\n")
    if start < 0:
        raise RuntimeError(f"JSON record missing from {path}")
    decoder = json.JSONDecoder()
    record, _ = decoder.raw_decode(text[start:])
    return record


def opcode_of(source: str) -> str:
    source = source.strip()
    source = re.sub(r"^@[!A-Z0-9.]+\s+", "", source)
    if not source:
        raise RuntimeError("empty SASS source")
    return source.split(None, 1)[0]


def parse_sass(path: Path, expected_fragment: str) -> dict:
    rows = list(csv.reader(path.open(newline="")))
    if len(rows) < 3 or rows[0][0] != "Kernel Name" or expected_fragment not in rows[0][1]:
        raise RuntimeError(f"unexpected selected function in {path}")
    instructions = []
    dynamic = Counter()
    for row in rows[2:]:
        if len(row) != 3:
            raise RuntimeError(f"malformed SASS row in {path}: {row}")
        source = " ".join(row[1].strip().split())
        opcode = opcode_of(source)
        executed = int(row[2])
        instructions.append(source)
        dynamic[opcode] += executed
    normalized = "\n".join(instructions) + "\n"
    opcodes = Counter(opcode_of(item) for item in instructions)
    family_patterns = {
        "tf32_conversion": re.compile(r"(?:^|\.)TF32(?:\.|$)"),
        "mma_any": re.compile(r"MMA"),
        "half_bf16_fp8_int_mma": re.compile(r"(?:HMMA|IMMA|QMMA|QGMMA|WGMMA)"),
        "fp32_fma": re.compile(r"^FFMA(?:\.|$)"),
        "loads": re.compile(r"^(?:LD|LDC|LDCU|LDG|LDL|LDS|LDSM|ULDC|S2R|S2UR)(?:\.|$)"),
        "stores": re.compile(r"^(?:ST|STG|STL|STS)(?:\.|$)"),
        "barriers": re.compile(r"(?:BAR|MEMBAR|DEPBAR|CGAERRBAR)"),
    }
    family_static = {
        name: sum(count for opcode, count in opcodes.items() if pattern.search(opcode))
        for name, pattern in family_patterns.items()
    }
    family_dynamic = {
        name: sum(count for opcode, count in dynamic.items() if pattern.search(opcode))
        for name, pattern in family_patterns.items()
    }
    return {
        "kernel_name": rows[0][1],
        "raw_sass_csv_sha256": sha256_file(path),
        "normalized_sass_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "static_instruction_count": len(instructions),
        "sum_ncu_instructions_executed": sum(dynamic.values()),
        "opcode_histogram_static": dict(sorted(opcodes.items())),
        "opcode_histogram_dynamic": dict(sorted(dynamic.items())),
        "family_counts_static": family_static,
        "family_counts_dynamic": family_dynamic,
        "forbidden_opcode_counts_static": {
            mnemonic: sum(count for opcode, count in opcodes.items() if mnemonic in opcode)
            for mnemonic in (
                "F2FP.TF32",
                "HMMA",
                "IMMA",
                "MMA",
                "QMMA",
                "QGMMA",
                "WGMMA",
            )
        },
        "normalization": "trim/collapse whitespace; keep predication, opcode and operands; omit address/execution count",
    }


def parse_details(path: Path, expected_fragment: str) -> dict:
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows or any(expected_fragment not in row["Kernel Name"] for row in rows):
        raise RuntimeError(f"unexpected NCU details kernel in {path}")
    metric = {row["Metric Name"]: (row["Metric Value"], row["Metric Unit"]) for row in rows}

    def value(name: str, cast):
        if name not in metric:
            raise RuntimeError(f"missing NCU metric {name} in {path}")
        return cast(metric[name][0])

    first = rows[0]
    return {
        "ncu_kernel_name": first["Kernel Name"],
        "compute_capability": first["CC"],
        "block_size_threads": value("Block Size", int),
        "grid_size_blocks": value("Grid Size", int),
        "registers_per_thread": value("Registers Per Thread", int),
        "static_shared_memory_bytes_per_block": value("Static Shared Memory Per Block", int),
        "dynamic_shared_memory_kbyte_per_block": value("Dynamic Shared Memory Per Block", float),
        "driver_shared_memory_kbyte_per_block": value("Driver Shared Memory Per Block", float),
        "local_memory_spilling_request_bytes": value("Local Memory Spilling Requests", int),
        "ncu_details_csv_sha256": sha256_file(path),
    }


def nsys_launch_line(path: Path, range_name: str, kernel_fragment: str) -> str:
    matches = [
        line.strip()
        for line in path.read_text().splitlines()
        if range_name in line and kernel_fragment in line
    ]
    if len(matches) != 2:
        # cuda_gpu_trace and cuda_gpu_kern_gb_sum each retain this launch.
        raise RuntimeError(f"expected two bound NSYS launch rows in {path}, got {len(matches)}")
    return matches[0]


def main() -> int:
    if sha256_file(PROJECT / "src/run_h2b_p2_trace.py") != TRACE_RUNNER_SHA256:
        raise RuntimeError("trace runner hash mismatch")

    summary = {
        "experiment": "tensorjoin_20260903_h2b_p2_cublas_sass",
        "decision": "PASS",
        "scope": "runtime launch attribution and selected-function SASS/resource audit only",
        "host": "gpu-host-8",
        "physical_gpu": 1,
        "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "gpu_uuid": "GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3",
        "tool_versions": {
            "nsight_systems": "2025.5.2.266-255236693005v0",
            "nsight_compute": "2025.4.1.0 build 37053803",
        },
        "cublas": {
            "version": 130100,
            "math_mode": 2,
            "compute_type": 69,
            "input_output_type": 0,
            "algorithm": -1,
            "nvidia_tf32_override": "0",
            "libcublas_sha256": CUBLAS_SHA256,
            "libcublaslt_sha256": CUBLASLT_SHA256,
        },
        "module_provenance": {
            "kind": "unresolved parent image; exact runtime-selected function exported by NCU",
            "limitation": "NCU binds the selected function to the profiled launch and exports SASS text, but the parent static/JIT image was not exposed by the trace",
        },
        "cases": {},
        "gate": {
            "both_runtime_launches_bound": True,
            "both_selected_functions_exported": True,
            "zero_tf32_conversion": True,
            "zero_mma_any": True,
            "zero_lower_precision_mma": True,
            "fp32_ffma_present": True,
        },
        "failed_attempt": {
            "path": "raw/h2b_p2_nsys_esc50_panns_attempt0.log",
            "sha256": sha256_file(RAW / "h2b_p2_nsys_esc50_panns_attempt0.log"),
            "reason": "relative env executable rejected by nsys before CUDA launch",
        },
        "commands": {
            "nsys": "CUDA_VISIBLE_DEVICES=1 nsys profile --trace=cuda,nvtx,cublas-verbose --capture-range=nvtx --capture-range-end=stop-shutdown --output=<report> /usr/bin/env NVIDIA_TF32_OVERRIDE=0 <python> src/run_h2b_p2_trace.py --dataset=<dataset>",
            "ncu": "CUDA_VISIBLE_DEVICES=1 ncu --target-processes all --launch-skip 0 --launch-count 1 --section LaunchStats --section InstructionStats --export <report> --force-overwrite /usr/bin/env NVIDIA_TF32_OVERRIDE=0 <python> src/run_h2b_p2_trace.py --dataset=<dataset>",
            "exports": "ncu --import <report> --page raw --print-source sass --csv; ncu --import <report> --page details --csv; nsys stats --report cuda_gpu_trace,cuda_api_trace,cuda_gpu_kern_gb_sum,nvtx_gpu_proj_trace --format csv --force-export=true <report>",
        },
    }

    for dataset, expected in CASES.items():
        ncu_log = RAW / f"h2b_p2_ncu_{dataset}.log"
        nsys_log = RAW / f"h2b_p2_nsys_{dataset}.log"
        sass_path = RAW / f"h2b_p2_ncu_{dataset}_sass.csv"
        details_path = RAW / f"h2b_p2_ncu_{dataset}_details.csv"
        stats_path = RAW / f"h2b_p2_nsys_{dataset}_stats.csv"
        trace = parse_json_from_log(ncu_log)
        if [trace["m_query_tokens"], trace["n_base_tokens"], trace["k_dimension"]] != expected["shape"]:
            raise RuntimeError(f"shape mismatch for {dataset}")
        if trace["output_sha256"] != expected["output_sha256"]:
            raise RuntimeError(f"output hash mismatch for {dataset}")
        if trace["cublas"]["math_mode"] != 2 or trace["cublas"]["compute_type"] != 69:
            raise RuntimeError(f"cuBLAS contract mismatch for {dataset}")
        sass = parse_sass(sass_path, expected["kernel_fragment"])
        details = parse_details(details_path, expected["kernel_fragment"])
        if details["grid_size_blocks"] != expected["grid"][0] * expected["grid"][1] * expected["grid"][2]:
            raise RuntimeError(f"grid mismatch for {dataset}")
        if details["block_size_threads"] != expected["block"][0]:
            raise RuntimeError(f"block mismatch for {dataset}")
        if details["registers_per_thread"] != expected["registers_per_thread"]:
            raise RuntimeError(f"register mismatch for {dataset}")
        if abs(details["dynamic_shared_memory_kbyte_per_block"] - expected["dynamic_shared_kbyte"]) > 0.01:
            raise RuntimeError(f"shared-memory mismatch for {dataset}")
        launch = nsys_launch_line(
            stats_path, f"H2B_P2_PEDANTIC_SGEMM_{dataset}", expected["kernel_fragment"]
        )
        case = {
            "shape_m_n_k": expected["shape"],
            "grid_xyz": expected["grid"],
            "block_xyz": expected["block"],
            "output_sha256": trace["output_sha256"],
            "output_nonfinite": trace["output_nonfinite"],
            "runtime_kernel_name": sass["kernel_name"],
            "nsys_bound_launch_row": launch,
            "resources": details,
            "sass": sass,
            "evidence_sha256": {
                "ncu_report": sha256_file(ARTIFACTS / f"ncu_{dataset}.ncu-rep"),
                "nsys_report": sha256_file(ARTIFACTS / f"nsys_{dataset}.nsys-rep"),
                "nsys_sqlite": sha256_file(ARTIFACTS / f"nsys_{dataset}.sqlite"),
                "ncu_log": sha256_file(ncu_log),
                "nsys_log": sha256_file(nsys_log),
                "nsys_stats_csv": sha256_file(stats_path),
            },
        }
        summary["cases"][dataset] = case

    all_cases = summary["cases"].values()
    if any(case["sass"]["family_counts_static"]["tf32_conversion"] for case in all_cases):
        summary["gate"]["zero_tf32_conversion"] = False
    all_cases = summary["cases"].values()
    if any(case["sass"]["family_counts_static"]["mma_any"] for case in all_cases):
        summary["gate"]["zero_mma_any"] = False
    all_cases = summary["cases"].values()
    if any(case["sass"]["family_counts_static"]["half_bf16_fp8_int_mma"] for case in all_cases):
        summary["gate"]["zero_lower_precision_mma"] = False
    all_cases = summary["cases"].values()
    if any(case["sass"]["family_counts_static"]["fp32_fma"] == 0 for case in all_cases):
        summary["gate"]["fp32_ffma_present"] = False
    if not all(summary["gate"].values()):
        summary["decision"] = "REJECT"

    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    RESULT.write_text(payload)
    lines = [
        f"decision={summary['decision']}",
        f"experiment={summary['experiment']}",
    ]
    for name, case in summary["cases"].items():
        fc = case["sass"]["family_counts_static"]
        lines.extend(
            [
                f"{name}.kernel={case['runtime_kernel_name']}",
                f"{name}.shape={case['shape_m_n_k']}",
                f"{name}.grid={case['grid_xyz']}",
                f"{name}.block={case['block_xyz']}",
                f"{name}.registers_per_thread={case['resources']['registers_per_thread']}",
                f"{name}.dynamic_shared_kbyte_per_block={case['resources']['dynamic_shared_memory_kbyte_per_block']}",
                f"{name}.static_instructions={case['sass']['static_instruction_count']}",
                f"{name}.normalized_sass_sha256={case['sass']['normalized_sass_sha256']}",
                f"{name}.tf32_conversion={fc['tf32_conversion']}",
                f"{name}.mma_any={fc['mma_any']}",
                f"{name}.lower_precision_mma={fc['half_bf16_fp8_int_mma']}",
                f"{name}.fp32_ffma={fc['fp32_fma']}",
            ]
        )
    lines.append(f"result_sha256={hashlib.sha256(payload.encode()).hexdigest()}")
    TEXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if summary["decision"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
