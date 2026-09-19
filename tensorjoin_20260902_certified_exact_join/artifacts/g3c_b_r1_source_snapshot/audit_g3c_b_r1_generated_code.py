#!/usr/bin/env python3
"""Audit exact runtime-generated G3C-B-R1 cubins without making timing claims."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RESULT = PROJECT / "results/g3c_b_r1_generated_code_audit.json"
CUDA_OBJDUMP = Path("/usr/local/cuda-13.1/bin/cuobjdump")
EXPECTED_G3B_CUBIN_SHA256 = (
    "dd32e97936d8e0c6abd3924d8fdeeb17088e7a6e1e4d8ac2f1fc492bbc2114ac"
)
EXPECTED_CANDIDATE_SOURCE_SHA256 = (
    "637e9455fdcdffcc8cddbed625ef74ef371f117876bc3f25977a79b4a95f706d"
)
ADDRESS_RE = re.compile(r"^\s*/\*\s*[0-9a-fA-F]+\s*\*/\s*(.*)$")
RAW_RE = re.compile(r"\s*/\*\s*(?:0x)?[0-9a-fA-F]{8,}\s*\*/\s*$")
RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def exactly_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def tool_output(*args: str) -> str:
    return subprocess.run(
        [str(CUDA_OBJDUMP), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def normalized_instructions(sass: str) -> str:
    output: list[str] = []
    for line in sass.splitlines():
        match = ADDRESS_RE.match(line)
        if not match:
            continue
        instruction = RAW_RE.sub("", match.group(1)).strip()
        if not instruction or instruction.startswith("/*"):
            continue
        output.append(re.sub(r"\s+", " ", instruction))
    return "\n".join(output) + "\n"


def opcode_counts(normalized: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in normalized.splitlines():
        instruction = re.sub(r"^@!?P\d+\s+", "", line)
        match = re.match(r"([A-Za-z][A-Za-z0-9_.]*)", instruction)
        if match:
            counts[match.group(1).upper()] += 1
    return counts


def audit_kernel(cubin: Path, ptx: Path, artifact_prefix: str) -> dict[str, object]:
    sass = tool_output("--dump-sass", str(cubin))
    resources = tool_output("--dump-resource-usage", str(cubin))
    normalized = normalized_instructions(sass)
    sass_path = PROJECT / f"artifacts/{artifact_prefix}.sass"
    normalized_path = PROJECT / f"artifacts/{artifact_prefix}.normalized.sass"
    resource_path = PROJECT / f"artifacts/{artifact_prefix}.resources.txt"
    sass_path.write_text(sass, encoding="utf-8")
    normalized_path.write_text(normalized, encoding="utf-8")
    resource_path.write_text(resources, encoding="utf-8")
    resource_match = RESOURCE_RE.search(resources)
    if not resource_match:
        raise RuntimeError(f"Could not parse resources for {cubin}")
    registers, stack, shared, local = map(int, resource_match.groups())
    counts = opcode_counts(normalized)
    ptx_text = ptx.read_text(encoding="utf-8", errors="replace")
    hfma2_lines = [line for line in normalized.splitlines() if "HFMA2" in line]
    hfma2_zero_only = all("-RZ, RZ, 0, 0" in line for line in hfma2_lines)
    low_precision_data_mma = sum(
        count
        for opcode, count in counts.items()
        if any(token in opcode for token in ("HMMA", "QMMA", "IMMA"))
    )
    fp64_sass = sum(
        count
        for opcode, count in counts.items()
        if opcode.startswith(("DADD", "DMUL", "DFMA", "DSETP"))
    )
    return {
        "cubin": str(cubin.relative_to(PROJECT)),
        "cubin_sha256": sha256_file(cubin),
        "ptx": str(ptx.relative_to(PROJECT)),
        "ptx_sha256": sha256_file(ptx),
        "sass": str(sass_path.relative_to(PROJECT)),
        "sass_sha256": sha256_file(sass_path),
        "normalized_sass": str(normalized_path.relative_to(PROJECT)),
        "normalized_sass_sha256": sha256_file(normalized_path),
        "normalized_instruction_count": sum(counts.values()),
        "resources": {
            "registers_per_thread": registers,
            "stack_bytes": stack,
            "shared_bytes": shared,
            "local_bytes": local,
        },
        "selected_opcode_counts": {
            key: value
            for key, value in sorted(counts.items())
            if key.startswith(
                (
                    "FADD",
                    "FMUL",
                    "FFMA",
                    "FSETP",
                    "HFMA2",
                    "HMMA",
                    "QMMA",
                    "IMMA",
                    "DADD",
                    "DMUL",
                    "DFMA",
                    "LDL",
                    "STL",
                )
            )
        },
        "sass_ldl_count": sum(v for k, v in counts.items() if k.startswith("LDL")),
        "sass_stl_count": sum(v for k, v in counts.items() if k.startswith("STL")),
        "sass_low_precision_data_mma_count": low_precision_data_mma,
        "sass_fp64_arithmetic_count": fp64_sass,
        "sass_hfma2_count": len(hfma2_lines),
        "sass_hfma2_zero_initialization_only": hfma2_zero_only,
        "ptx_target_sm120a": ".target sm_120a" in ptx_text,
        "ptx_float32_arithmetic_present": bool(
            re.search(r"\b(?:add|sub|mul|fma)(?:\.rn)?\.f32\b", ptx_text)
        ),
        "ptx_low_precision_or_fp64_arithmetic_absent": not bool(
            re.search(r"\.(?:f16|bf16|tf32|f64)\b", ptx_text)
        ),
        "resource_text": str(resource_path.relative_to(PROJECT)),
        "resource_text_sha256": sha256_file(resource_path),
    }


def main() -> int:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if not CUDA_OBJDUMP.is_file():
        raise FileNotFoundError(CUDA_OBJDUMP)
    run0 = PROJECT / "artifacts/g3c_b_r1_triton_cache_run0"
    run1 = PROJECT / "artifacts/g3c_b_r1_triton_cache_run1"
    candidate0_cubin = exactly_one(run0, "certified_fp32_filter_i64.cubin")
    candidate0_ptx = exactly_one(run0, "certified_fp32_filter_i64.ptx")
    candidate1_cubin = exactly_one(run1, "certified_fp32_filter_i64.cubin")
    candidate1_ptx = exactly_one(run1, "certified_fp32_filter_i64.ptx")
    g3b0_cubin = exactly_one(run0, "analytic_certificate_compact_i64.cubin")
    g3b0_ptx = exactly_one(run0, "analytic_certificate_compact_i64.ptx")
    g3b1_cubin = exactly_one(run1, "analytic_certificate_compact_i64.cubin")
    g3b1_ptx = exactly_one(run1, "analytic_certificate_compact_i64.ptx")

    candidate0 = audit_kernel(
        candidate0_cubin, candidate0_ptx, "g3c_b_r1_candidate_fp32_filter_run0"
    )
    candidate1 = audit_kernel(
        candidate1_cubin, candidate1_ptx, "g3c_b_r1_candidate_fp32_filter_run1"
    )
    g3b0 = audit_kernel(g3b0_cubin, g3b0_ptx, "g3c_b_r1_candidate_g3b_stage_run0")
    g3b1 = audit_kernel(g3b1_cubin, g3b1_ptx, "g3c_b_r1_candidate_g3b_stage_run1")

    u = 2.0**-24
    d = 512
    gamma = (d - 1) * u / (1.0 - (d - 1) * u)
    required_distance = (2.0 * u + gamma * (1.0 + u)) / (
        (1.0 - u) * (1.0 - gamma)
    )
    required_input = (u + u * u) / ((1.0 - u) ** 2 * (1.0 - gamma))
    implemented_distance = 2.0**-14
    implemented_input = 2.0**-22
    implemented_final = 2.0**-22
    coefficient_screen = {
        "float32_unit_roundoff": u,
        "gamma_511": gamma,
        "derived_required_distance_coefficient": required_distance,
        "implemented_distance_coefficient": implemented_distance,
        "distance_margin_ratio": implemented_distance / required_distance,
        "derived_required_input_coefficient": required_input,
        "implemented_input_coefficient": implemented_input,
        "input_margin_ratio": implemented_input / required_input,
        "implemented_final_expression_coefficient": implemented_final,
        "final_expression_margin_in_unit_roundoffs": implemented_final / u,
        "derived_ftz_tiny_operation_allowance": 2048,
        "implemented_tiny_allowance": 4096,
        "tiny_allowance_ratio": 2.0,
        "status": "analytic_coefficient_screen_not_standalone_generated_code_proof",
    }
    run_stability = bool(
        candidate0["cubin_sha256"] == candidate1["cubin_sha256"]
        and candidate0["normalized_sass_sha256"]
        == candidate1["normalized_sass_sha256"]
        and g3b0["cubin_sha256"] == g3b1["cubin_sha256"]
        and g3b0["normalized_sass_sha256"] == g3b1["normalized_sass_sha256"]
    )
    candidate_semantics_pass = bool(
        candidate0["ptx_target_sm120a"]
        and candidate0["ptx_float32_arithmetic_present"]
        and candidate0["ptx_low_precision_or_fp64_arithmetic_absent"]
        and candidate0["sass_low_precision_data_mma_count"] == 0
        and candidate0["sass_fp64_arithmetic_count"] == 0
        and candidate0["sass_hfma2_zero_initialization_only"]
        and candidate0["sass_ldl_count"] == 0
        and candidate0["sass_stl_count"] == 0
        and candidate0["resources"]["stack_bytes"] == 0
        and candidate0["resources"]["local_bytes"] == 0
        and coefficient_screen["distance_margin_ratio"] > 1.9
        and coefficient_screen["input_margin_ratio"] > 3.9
        and coefficient_screen["final_expression_margin_in_unit_roundoffs"] >= 4.0
        and coefficient_screen["tiny_allowance_ratio"] >= 2.0
    )
    g3b_preserved = bool(
        g3b0["cubin_sha256"] == EXPECTED_G3B_CUBIN_SHA256
        and g3b1["cubin_sha256"] == EXPECTED_G3B_CUBIN_SHA256
        and g3b0["normalized_sass_sha256"]
        == g3b1["normalized_sass_sha256"]
    )
    source_hash = sha256_file(PROJECT / "src/run_g3c_b_r1_gpu_cascade.py")
    gate_pass = bool(
        source_hash == EXPECTED_CANDIDATE_SOURCE_SHA256
        and run_stability
        and candidate_semantics_pass
        and g3b_preserved
    )
    result = {
        "experiment_id": "tensorjoin_20260903_g3c_b_r1_gpu_cascade",
        "measurement_status": "generated_code_semantics_audit_not_performance",
        "candidate_source_sha256": source_hash,
        "candidate_run0": candidate0,
        "candidate_run1": candidate1,
        "g3b_stage_run0": g3b0,
        "g3b_stage_run1": g3b1,
        "coefficient_screen": coefficient_screen,
        "two_process_generated_code_stable": run_stability,
        "candidate_float32_semantics_gate_pass": candidate_semantics_pass,
        "g3b_runtime_artifact_preserved": g3b_preserved,
        "g3b_preservation_basis": (
            "exact full-cubin SHA-256 equality to frozen G3B-R1 plus "
            "same-cuobjdump-normalizer equality across both G3C-B processes"
        ),
        "r1_source_delta": (
            "only final_magnitude=max(abs(d2)+R0,1) changed to "
            "final_magnitude=abs(d2)+R0"
        ),
        "generated_code_gate_pass": gate_pass,
        "performance_claim_allowed": False,
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(RESULT, result)
    print("AUDIT_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
