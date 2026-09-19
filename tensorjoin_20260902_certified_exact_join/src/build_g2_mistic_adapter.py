#!/usr/bin/env python3
"""Build an isolated SM120 canonical-output adapter for pinned MiSTIC."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "external/self-join-MiSTIC"
OUTPUT = PROJECT / "adapters/mistic_g2a"
RECEIPT = PROJECT / "receipts/g2a_mistic_adapter_build.json"
NVCC = Path("/usr/local/cuda-13.1/bin/nvcc")
REVISION = "656dd47b5594f1d9d7f961f008cfc6aedb0aed9d"
SOURCE_TREE = "0297078cb57283f91fe59bcf453cdb6b7726a627"


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def replace_define(text: str, name: str, value: str) -> str:
    updated, count = re.subn(
        rf"(?m)^#define\s+{re.escape(name)}\s+.*$", f"#define {name} {value}", text
    )
    if count != 1:
        raise ValueError(f"Expected one {name} define, found {count}")
    return updated


def one_replace(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected one {label} site, found {count}")
    return text.replace(old, new)


def run(command: list[str], cwd: Path) -> None:
    print("COMMAND " + json.dumps(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT} or {RECEIPT}")
    if not SOURCE.is_dir() or not NVCC.is_file():
        raise FileNotFoundError(f"Missing source/toolkit: {SOURCE}, {NVCC}")

    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp.{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        SOURCE,
        temporary,
        ignore=shutil.ignore_patterns(
            ".git", "build", "build_*", "*.o", "main", "main_d512", "*.ncu-rep"
        ),
    )
    commands: list[list[str]] = []
    originals: dict[str, str] = {}
    try:
        params_path = temporary / "include/params.cuh"
        launcher_path = temporary / "src/launcher.cu"
        main_path = temporary / "src/main.cu"
        for path in (params_path, launcher_path, main_path):
            originals[str(path.relative_to(temporary))] = path.read_text(encoding="utf-8")

        params = replace_define(originals["include/params.cuh"], "CUDA_DEVICE", "0")
        params_path.write_text(params, encoding="utf-8")

        launcher = originals["src/launcher.cu"]
        marker = "struct neighborTable * nodeLauncher5(double * data,"
        start = launcher.index(marker)
        end = launcher.index("struct neighborTable * launchCOSS(", start)
        prefix, node5, suffix = launcher[:start], launcher[start:end], launcher[end:]
        allocation = (
            "struct neighborTable * tables = "
            "(struct neighborTable*)malloc(sizeof(struct neighborTable)*numPoints);"
        )
        if node5.count(allocation) != 1:
            raise ValueError("Could not isolate nodeLauncher5 neighbor-table allocation")
        node5 = node5.replace(
            allocation,
            "struct neighborTable * tables = new neighborTable[numPoints];",
            1,
        )
        launcher = prefix + node5 + suffix
        launcher_path.write_text(launcher, encoding="utf-8")

        original_call = """\t#if KT == 5
\tnodeLauncher5(dimOrderedData,
\t\tdim,
\t\tnumPoints,
\t\t0, //numRP
\t\tpointArray,
\t\tepsilon);
\t#endif
"""
        adapted_call = """\t#if KT == 5
\tstruct neighborTable * table = nodeLauncher5(dimOrderedData,
\t\tdim,
\t\tnumPoints,
\t\t0, //numRP
\t\tpointArray,
\t\tepsilon);

\tconst char *g2aOutputPath = getenv("MISTIC_G2A_OUTPUT");
\tif (g2aOutputPath == nullptr) {
\t\tstd::cerr << "MISTIC_G2A_OUTPUT is required for the G2A adapter" << std::endl;
\t\treturn 3;
\t}
\tstd::vector<unsigned long long> g2aPairs;
\tstd::vector<unsigned char> g2aHasSelf(numPoints, 0);
\tunsigned long long g2aRawKernelPairs = 0;
\tunsigned long long g2aInvalidPositions = 0;
\tfor (unsigned int orderedQuery = 0; orderedQuery < numPoints; orderedQuery++) {
\t\tconst unsigned int query = pointArray[orderedQuery];
\t\tif (query >= numPoints) {
\t\t\tg2aInvalidPositions++;
\t\t\tcontinue;
\t\t}
\t\tfor (unsigned int arrayIndex = 1;
\t\t\t arrayIndex < table[orderedQuery].cntNDataArrays; arrayIndex++) {
\t\t\tfor (unsigned int valueIndex = table[orderedQuery].vectindexmin[arrayIndex];
\t\t\t\t valueIndex <= table[orderedQuery].vectindexmax[arrayIndex]; valueIndex++) {
\t\t\t\tconst unsigned int orderedCandidate =
\t\t\t\t\ttable[orderedQuery].vectdataPtr[arrayIndex][valueIndex];
\t\t\t\tg2aRawKernelPairs++;
\t\t\t\tif (orderedCandidate >= numPoints) {
\t\t\t\t\tg2aInvalidPositions++;
\t\t\t\t\tcontinue;
\t\t\t\t}
\t\t\t\tconst unsigned int candidate = pointArray[orderedCandidate];
\t\t\t\tif (candidate >= numPoints) {
\t\t\t\t\tg2aInvalidPositions++;
\t\t\t\t\tcontinue;
\t\t\t\t}
\t\t\t\tif (query == candidate) g2aHasSelf[query] = 1;
\t\t\t\tg2aPairs.push_back(
\t\t\t\t\tstatic_cast<unsigned long long>(query) * numPoints + candidate);
\t\t\t}
\t\t}
\t}
\tunsigned long long g2aSelfInsertions = 0;
\tfor (unsigned int point = 0; point < numPoints; point++) {
\t\tif (!g2aHasSelf[point]) {
\t\t\tg2aPairs.push_back(static_cast<unsigned long long>(point) * numPoints + point);
\t\t\tg2aSelfInsertions++;
\t\t}
\t}
\tstd::sort(g2aPairs.begin(), g2aPairs.end());
\tstd::ofstream g2aOutput(g2aOutputPath, std::ios::binary | std::ios::out);
\tg2aOutput.write(reinterpret_cast<const char *>(g2aPairs.data()),
\t\tstatic_cast<std::streamsize>(g2aPairs.size() * sizeof(unsigned long long)));
\tg2aOutput.close();
\tif (!g2aOutput) {
\t\tstd::cerr << "Failed to write G2A canonical pair file" << std::endl;
\t\treturn 4;
\t}
\tstd::cout << "G2A_ADAPTER raw_kernel_pairs=" << g2aRawKernelPairs
\t\t<< " normalized_self_insertions=" << g2aSelfInsertions
\t\t<< " invalid_positions=" << g2aInvalidPositions
\t\t<< " output_pairs=" << g2aPairs.size() << std::endl;
\t#endif
"""
        main_text = originals["src/main.cu"]
        if "#include <algorithm>" not in main_text:
            main_text = main_text.replace("#include <vector>\n", "#include <vector>\n#include <algorithm>\n")
        main_text = one_replace(main_text, original_call, adapted_call, "KT=5 call")
        main_path.write_text(main_text, encoding="utf-8")

        patch_parts: list[str] = []
        for relative in ("include/params.cuh", "src/launcher.cu", "src/main.cu"):
            updated = (temporary / relative).read_text(encoding="utf-8")
            patch_parts.extend(
                difflib.unified_diff(
                    originals[relative].splitlines(keepends=True),
                    updated.splitlines(keepends=True),
                    fromfile=f"upstream/{relative}",
                    tofile=f"adapter/{relative}",
                )
            )
        (temporary / "g2a_adapter.patch").write_text("".join(patch_parts), encoding="utf-8")

        build_dir = temporary / "build_g2a"
        build_dir.mkdir()
        common = [
            str(NVCC),
            "-std=c++17",
            "-O3",
            "-Xcompiler=-fopenmp",
            "-lineinfo",
            "-arch=sm_120",
            "-I.",
            "-DDIM=512",
            "-DBS=256",
            "-DKB=1024",
        ]
        objects: list[str] = []
        for stem in ("main", "launcher", "kernel", "nodes", "tree", "utils"):
            source_name = f"src/{stem}.cu"
            object_name = f"build_g2a/{stem}.o"
            command = [*common, "-c", source_name, "-o", object_name]
            commands.append(command)
            run(command, temporary)
            objects.append(object_name)
        link = [*common, *objects, "-o", "build_g2a/main_d512"]
        commands.append(link)
        run(link, temporary)
        binary = temporary / "build_g2a/main_d512"
        if not binary.is_file() or binary.stat().st_size == 0:
            raise RuntimeError("MiSTIC G2A binary was not produced")
        os.replace(temporary, OUTPUT)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    binary = OUTPUT / "build_g2a/main_d512"
    receipt = {
        "name": "MiSTIC G2A canonical-output adapter",
        "upstream_revision": REVISION,
        "upstream_tree": SOURCE_TREE,
        "upstream_path": str(SOURCE),
        "adapter_path": str(OUTPUT),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "patch_sha256": sha256_file(OUTPUT / "g2a_adapter.patch"),
        "binary_bytes": binary.stat().st_size,
        "binary_sha256": sha256_file(binary),
        "nvcc": subprocess.check_output([str(NVCC), "--version"], text=True),
        "commands": commands,
        "configuration": {
            "DIM": 512,
            "BS": 256,
            "KB": 1024,
            "CUDA_DEVICE": 0,
            "KT": 5,
            "architecture": "sm_120",
            "language_standard": "c++17",
        },
        "changes": [
            "Select physical/visible GPU0 instead of upstream CUDA_DEVICE=1",
            "Use new[] for nodeLauncher5 neighborTable objects containing std::vector",
            "Serialize mapped directed uint64 pair IDs; normalize only missing self-pairs",
        ],
        "kernel_or_distance_predicate_change": False,
    }
    atomic_json(RECEIPT, receipt)
    print("BUILD_COMPLETE " + json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
