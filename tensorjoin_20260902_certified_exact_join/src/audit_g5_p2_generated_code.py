#!/usr/bin/env python3
"""Audit the exact full-N G5 P1 runtime-selected Triton functions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
P1 = PROJECT / "results/g5_compatibility_tensorjoin_p1_r1_a0.json"
RESULT = PROJECT / "results/g5_p2_generated_code_audit_p1.json"
CUDA_OBJDUMP = Path("/usr/local/cuda-13.1/bin/cuobjdump")
KERNELS = (
    "analytic_certificate_ragged_safe_i64",
    "certified_fp32_filter_i64",
    "refine_ambiguous_fp64_i64",
)
EXPECTED_P1_SHA256 = (
    "7fcdd2786db413ee9f56347c27eceb9542e41e8fc644e9dd3a55d9ec59bff2a5"
)
EXPECTED_SPECIALIZATION = {
    "N_": 60_000,
    "K": 512,
    "CAPACITY": 16_777_216,
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_K": 64,
    "FP32_BLOCK_K": 256,
    "FP64_BLOCK_K": 256,
    "num_warps": 4,
    "stage1_num_stages": 3,
}
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
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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
        if instruction and not instruction.startswith("/*"):
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


def audit_kernel(name: str, source: dict[str, object]) -> dict[str, object]:
    cubin = PROJECT / str(source["cubin"])
    ptx = PROJECT / str(source["ptx"])
    if sha256_file(cubin) != source["cubin_sha256"]:
        raise RuntimeError(f"P1 cubin hash changed for {name}")
    if sha256_file(ptx) != source["ptx_sha256"]:
        raise RuntimeError(f"P1 PTX hash changed for {name}")
    sass = tool_output("--dump-sass", str(cubin))
    resources = tool_output("--dump-resource-usage", str(cubin))
    normalized = normalized_instructions(sass)
    prefix = PROJECT / f"artifacts/g5_p2_{name}"
    paths = {
        "sass": prefix.with_suffix(".sass"),
        "normalized_sass": prefix.with_suffix(".normalized.sass"),
        "resources": prefix.with_suffix(".resources.txt"),
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
    paths["sass"].write_text(sass, encoding="utf-8")
    paths["normalized_sass"].write_text(normalized, encoding="utf-8")
    paths["resources"].write_text(resources, encoding="utf-8")
    match = RESOURCE_RE.search(resources)
    if not match:
        raise RuntimeError(f"Cannot parse resource usage for {name}")
    registers, stack, shared, local = map(int, match.groups())
    counts = opcode_counts(normalized)
    ptx_text = ptx.read_text(encoding="utf-8", errors="replace")
    low_precision_mma = sum(
        value
        for opcode, value in counts.items()
        if any(token in opcode for token in ("HMMA", "QMMA", "IMMA"))
    )
    fp64_arithmetic = sum(
        value
        for opcode, value in counts.items()
        if opcode.startswith(("DADD", "DMUL", "DFMA", "DSETP"))
    )
    return {
        "name": name,
        "cubin": str(cubin.relative_to(PROJECT)),
        "cubin_sha256": sha256_file(cubin),
        "cubin_bytes": cubin.stat().st_size,
        "ptx": str(ptx.relative_to(PROJECT)),
        "ptx_sha256": sha256_file(ptx),
        "ptx_bytes": ptx.stat().st_size,
        "sass": str(paths["sass"].relative_to(PROJECT)),
        "sass_sha256": sha256_file(paths["sass"]),
        "normalized_sass": str(paths["normalized_sass"].relative_to(PROJECT)),
        "normalized_sass_sha256": sha256_file(paths["normalized_sass"]),
        "resource_text": str(paths["resources"].relative_to(PROJECT)),
        "resource_text_sha256": sha256_file(paths["resources"]),
        "resources": {
            "registers_per_thread": registers,
            "stack_bytes": stack,
            "shared_bytes": shared,
            "local_bytes": local,
        },
        "normalized_instruction_count": sum(counts.values()),
        "opcode_counts": dict(sorted(counts.items())),
        "low_precision_mma_count": low_precision_mma,
        "fp64_arithmetic_count": fp64_arithmetic,
        "ldl_count": sum(v for k, v in counts.items() if k.startswith("LDL")),
        "stl_count": sum(v for k, v in counts.items() if k.startswith("STL")),
        "ptx_target_sm120a": ".target sm_120a" in ptx_text,
        "ptx_contains_f64": ".f64" in ptx_text,
    }


def main() -> int:
    if not CUDA_OBJDUMP.is_file():
        raise FileNotFoundError(CUDA_OBJDUMP)
    if sha256_file(P1) != EXPECTED_P1_SHA256:
        raise RuntimeError("G5 P1 result hash mismatch")
    p1 = json.loads(P1.read_text(encoding="utf-8"))
    if not p1["correctness"]["exact_contract_pass"]:
        raise RuntimeError("G5 P1 exactness did not pass")
    if p1["specialization_contract"] != EXPECTED_SPECIALIZATION:
        raise RuntimeError("G5 P1 specialization contract mismatch")
    if not p1["selected_cache_artifacts_unchanged_during_timer"]:
        raise RuntimeError("G5 P1 changed selected artifacts during timing")
    source = p1["selected_cache_artifacts_after_timer"]
    audited = {name: audit_kernel(name, source[name]) for name in KERNELS}
    common_resource_pass = all(
        value["ptx_target_sm120a"]
        and value["resources"]["stack_bytes"] == 0
        and value["resources"]["local_bytes"] == 0
        and value["ldl_count"] == 0
        and value["stl_count"] == 0
        for value in audited.values()
    )
    semantic_role_pass = bool(
        audited[KERNELS[0]]["low_precision_mma_count"] > 0
        and audited[KERNELS[1]]["low_precision_mma_count"] == 0
        and audited[KERNELS[1]]["fp64_arithmetic_count"] == 0
        and not audited[KERNELS[1]]["ptx_contains_f64"]
        and audited[KERNELS[2]]["fp64_arithmetic_count"] > 0
        and audited[KERNELS[2]]["ptx_contains_f64"]
    )
    gate = bool(common_resource_pass and semantic_role_pass)
    result = {
        "experiment_id": "tensorjoin_20260903_g5_unified_public_cifar60k",
        "measurement_status": "generated_code_mechanism_evidence_not_performance",
        "p1_result": str(P1.relative_to(PROJECT)),
        "p1_result_sha256": sha256_file(P1),
        "specialization_contract": p1["specialization_contract"],
        "kernels": audited,
        "common_resource_gate_pass": common_resource_pass,
        "semantic_role_gate_pass": semantic_role_pass,
        "p1_generated_code_gate_pass": gate,
        "second_fresh_cache_identity_status": "pending_P4_tensorjoin_slot",
        "performance_claim_allowed": False,
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(
            PROJECT / "PROTOCOL_G5_COMPATIBILITY_SCREEN.md"
        ),
    }
    atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())

