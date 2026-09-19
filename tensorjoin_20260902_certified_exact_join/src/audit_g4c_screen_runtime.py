#!/usr/bin/env python3
"""Audit exact runtime-selected G4C screen cubins across native dimensions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RESULT = PROJECT / "results/g4c_screen_runtime_audit.json"
CUDA_OBJDUMP = Path("/usr/local/cuda-13.1/bin/cuobjdump")
DATASETS = ("sift128", "cifar_gist512", "fashion784")
KERNELS = (
    "analytic_certificate_ragged_safe_i64",
    "certified_fp32_filter_i64",
    "refine_ambiguous_fp64_i64",
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
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
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


def audit_kernel(cubin: Path, ptx: Path, prefix: str) -> dict[str, object]:
    sass = tool_output("--dump-sass", str(cubin))
    resources = tool_output("--dump-resource-usage", str(cubin))
    normalized = normalized_instructions(sass)
    sass_path = PROJECT / f"artifacts/{prefix}.sass"
    normalized_path = PROJECT / f"artifacts/{prefix}.normalized.sass"
    resource_path = PROJECT / f"artifacts/{prefix}.resources.txt"
    for path in (sass_path, normalized_path, resource_path):
        if path.exists():
            raise FileExistsError(path)
    sass_path.write_text(sass, encoding="utf-8")
    normalized_path.write_text(normalized, encoding="utf-8")
    resource_path.write_text(resources, encoding="utf-8")
    match = RESOURCE_RE.search(resources)
    if not match:
        raise RuntimeError(f"Could not parse resources for {cubin}")
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
        "cubin": str(cubin.relative_to(PROJECT)),
        "cubin_sha256": sha256_file(cubin),
        "ptx": str(ptx.relative_to(PROJECT)),
        "ptx_sha256": sha256_file(ptx),
        "sass": str(sass_path.relative_to(PROJECT)),
        "sass_sha256": sha256_file(sass_path),
        "normalized_sass": str(normalized_path.relative_to(PROJECT)),
        "normalized_sass_sha256": sha256_file(normalized_path),
        "resource_text": str(resource_path.relative_to(PROJECT)),
        "resource_text_sha256": sha256_file(resource_path),
        "resources": {
            "registers_per_thread": registers,
            "stack_bytes": stack,
            "shared_bytes": shared,
            "local_bytes": local,
        },
        "normalized_instruction_count": sum(counts.values()),
        "low_precision_mma_count": low_precision_mma,
        "fp64_arithmetic_count": fp64_arithmetic,
        "ldl_count": sum(v for k, v in counts.items() if k.startswith("LDL")),
        "stl_count": sum(v for k, v in counts.items() if k.startswith("STL")),
        "ptx_target_sm120a": ".target sm_120a" in ptx_text,
        "ptx_contains_f64": ".f64" in ptx_text,
    }


def main() -> int:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if not CUDA_OBJDUMP.is_file():
        raise FileNotFoundError(CUDA_OBJDUMP)
    datasets: dict[str, object] = {}
    all_pass = True
    for dataset in DATASETS:
        roots = []
        manifests = []
        for process_id, order in enumerate(("kc", "ck")):
            stem = f"g4c_screen_{dataset}_n4096_k64_process_{process_id}_{order}_a0"
            manifest_path = PROJECT / f"results/{stem}_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest.get("admitted"):
                raise RuntimeError(f"Non-admitted screen manifest: {manifest_path}")
            roots.append(PROJECT / str(manifest["cache_path"]))
            manifests.append(
                {
                    "path": str(manifest_path.relative_to(PROJECT)),
                    "sha256": sha256_file(manifest_path),
                }
            )
        process_hashes: list[dict[str, str]] = []
        for root in roots:
            process_hashes.append(
                {
                    name: sha256_file(exactly_one(root, f"{name}.cubin"))
                    for name in KERNELS
                }
            )
        stable = process_hashes[0] == process_hashes[1]
        audited: dict[str, object] = {}
        for name in KERNELS:
            cubin = exactly_one(roots[0], f"{name}.cubin")
            ptx = exactly_one(roots[0], f"{name}.ptx")
            audited[name] = audit_kernel(
                cubin, ptx, f"g4c_screen_{dataset}_{name}"
            )
        common_resource_pass = all(
            value["ptx_target_sm120a"]
            and value["resources"]["stack_bytes"] == 0
            and value["resources"]["local_bytes"] == 0
            and value["ldl_count"] == 0
            and value["stl_count"] == 0
            for value in audited.values()
        )
        semantic_pass = bool(
            audited["analytic_certificate_ragged_safe_i64"][
                "low_precision_mma_count"
            ]
            > 0
            and audited["certified_fp32_filter_i64"]["low_precision_mma_count"]
            == 0
            and audited["certified_fp32_filter_i64"]["fp64_arithmetic_count"]
            == 0
            and not audited["certified_fp32_filter_i64"]["ptx_contains_f64"]
            and audited["refine_ambiguous_fp64_i64"]["fp64_arithmetic_count"] > 0
            and audited["refine_ambiguous_fp64_i64"]["ptx_contains_f64"]
        )
        dataset_pass = bool(stable and common_resource_pass and semantic_pass)
        all_pass = all_pass and dataset_pass
        datasets[dataset] = {
            "manifests": manifests,
            "process_cubin_hashes": process_hashes,
            "two_process_cubins_byte_stable": stable,
            "kernels": audited,
            "resource_gate_pass": common_resource_pass,
            "semantic_role_gate_pass": semantic_pass,
            "dataset_runtime_binary_gate_pass": dataset_pass,
        }
    result = {
        "experiment_id": "tensorjoin_20260903_g4c_public_anchor_screen",
        "measurement_status": "runtime_binary_and_generated_code_audit_not_performance",
        "datasets": datasets,
        "runtime_binary_gate_pass": all_pass,
        "performance_claim_allowed": False,
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "screen_summary_sha256": sha256_file(
            PROJECT / "results/g4c_screen_summary.json"
        ),
    }
    atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
